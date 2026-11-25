package sdfx.vulkan.tools

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.util.concurrent.{CompletableFuture,ExecutionException,TimeUnit}
import sdfx.core.protocol.EngineError
import ujson.{Obj,Value}

/** The opt-in guest runs in a separate JVM so its stdout and lifetime are independent of JSONL. */
private[vulkan] object VulkanCustomCode:
  private def directory = sys.env.get("SDFX_VULKAN_SCRIPT_DIR").map(Path.of(_)).getOrElse(
    Path.of(sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR",".")).toAbsolutePath.resolveSibling("custom-code")).toAbsolutePath
  def enabled: Boolean = sys.env.get("SDFX_ALLOW_CODE").contains("1") &&
    Files.isRegularFile(directory.resolve("VulkanScriptHost.class")) &&
    Files.isRegularFile(directory.resolve("vulkan_script.py"))

  def run(raw: Value, device: String, model: Option[String], check: () => Unit, generate: Value => Value): Value =
    if !enabled then throw EngineError("unsupported","Enable Allow custom code and install the optional JVM scripting runtime")
    require(raw("code").str.length <= 1000000,"Custom code exceeds 1,000,000 characters")
    val dir = directory
    val javaExe = Path.of(System.getProperty("java.home"),"bin","java.exe").toString
    val cp = dir.toString + java.io.File.pathSeparator + dir.resolve("lib").toString + java.io.File.separator + "*"
    val proc = new ProcessBuilder(javaExe,"--enable-native-access=ALL-UNNAMED","-Xmx512m","-cp",cp,"VulkanScriptHost",
      dir.resolve("vulkan_script.py").toString).redirectError(ProcessBuilder.Redirect.INHERIT).start()
    val work = CompletableFuture.supplyAsync(() => {
      val writer = proc.outputWriter(UTF_8)
      val reader = proc.inputReader(UTF_8)
      def send(value: Value): Unit =
        writer.write(ujson.write(value))
        writer.newLine()
        writer.flush()
      try
        send(Obj("params"->raw,"device"->device,"model"->model.fold[Value](ujson.Null)(ujson.Str(_))))
        var result = Option.empty[Value]
        while result.isEmpty do
          val line = reader.readLine()
          require(line != null,"Custom script host exited before returning a result")
          val msg = ujson.read(line)
          if msg.obj.contains("error") then throw EngineError("invalid_params",msg("error").str)
          else if msg.obj.contains("generate") then
            val options = msg("generate")
            val base = raw.obj.get("generation_params").fold(Obj())(x=>Obj.from(x.obj))
            options.obj.foreach((key,v)=>base(key)=v)
            check()
            send(generate(Obj("params" -> base,"output_dir" -> raw("output_dir"))))
          else result = Some(msg("reply"))
        result.get
      finally
        writer.close()
        reader.close()
    })
    try
      while !work.isDone do
        check()
        try work.get(100,TimeUnit.MILLISECONDS)
        catch case _: java.util.concurrent.TimeoutException => ()
      check()
      work.get()
    catch
      case ex: ExecutionException => throw ex.getCause
    finally
      if proc.isAlive then proc.destroyForcibly()
      proc.waitFor()
