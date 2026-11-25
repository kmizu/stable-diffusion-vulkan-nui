package sdfx.vulkan

import java.io.{BufferedReader, InputStreamReader, OutputStream, OutputStreamWriter}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.UUID
import java.util.concurrent.{CancellationException, Executors, ThreadLocalRandom}
import javax.imageio.ImageIO
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import sdfx.core.imagebrowser.BrowserFiles
import sdfx.core.infotext.HardwareInfo
import sdfx.core.protocol.*
import ujson.{Arr, Obj, Str, Value}

object VulkanWorker:
  private[vulkan] def generation(raw: Value): VulkanProtocol.Generation = VulkanProtocol.generation(raw)

  def main(args: Array[String]): Unit =
    val dir = sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR", throw IllegalArgumentException("SDFX_VULKAN_LIBRARY_DIR is required"))
    Using.resource(new VulkanInference(Path.of(dir))) { backend =>
      new VulkanWorker(backend, sys.env.get("SDFX_VULKAN_DEVICE"), System.out)
        .serve(new BufferedReader(new InputStreamReader(System.in, UTF_8)))
    }

private final class VulkanWorker(backend: VulkanInference, selection: Option[String], out: OutputStream):
  import VulkanProtocol.*
  private val adapters = backend.devices().asScala.toVector
  private val device = select(adapters, selection)
  private val writer = new OutputStreamWriter(out, UTF_8)
  private val lock = new Object
  private val executor = Executors.newSingleThreadExecutor()
  private val jobs = mutable.LinkedHashMap.empty[String, Job]
  private var protection = InterruptionProtection()
  @volatile private var loaded = Option.empty[LoadedModel]

  private final class Job(val id: String, val kind: String, val owner: Option[String]):
    @volatile var cancelled = false
    var running = false
    val files = mutable.ArrayBuffer.empty[Path]
    def check(): Unit = if cancelled then throw CancellationException()

  private def emit(msg: Value): Unit = writer.synchronized {
    writer.write(ujson.write(msg))
    writer.write('\n')
    writer.flush()
  }
  private def result(id: String, value: Value): Unit =
    emit(Obj("id" -> id, "type" -> "result", "result" -> value))

  private def error(id: Value, err: Throwable): Unit =
    val (code, message) = err match
      case e: EngineError => (e.code, e.message)
      case _: OutOfMemoryError => ("oom", "The JVM could not allocate memory for the Vulkan worker")
      case e: VulkanInference.NativeFailure =>
        val nativeCode = e.status match
          case 1 | 2 | 3 | 4 => "invalid_params"
          case 5 => "unsupported"
          case 6 => "oom"
          case 7 => "model_load_failed"
          case _ => "internal"
        (nativeCode, s"${display(device)}: ${e.getMessage}")
      case _: CancellationException => ("interrupted", "Generation interrupted")
      case e: IllegalArgumentException => ("invalid_params", Option(e.getMessage).getOrElse("Invalid parameters"))
      case e => ("internal", Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
    emit(Obj("id" -> id, "type" -> "error", "code" -> code, "message" -> message, "detail" -> ""))

  private def hello = Obj("protocol" -> 1, "engine_version" -> "vulkan-sd.cpp-f296cb0", "cuda" -> false,
    "dry_run" -> false, "device" -> display(device), "vulkan_devices" -> Arr.from(adapters.map(display)),
    "features" -> Arr("txt2img", "list_devices", "prevent_interrupt"))

  def serve(in: BufferedReader): Unit =
    val greeting = hello
    greeting("type") = "hello"
    greeting("id") = ujson.Null
    emit(greeting)
    try
      var running = true
      while running do
        val line = in.readLine()
        if line == null then running = false
        else if line.nonEmpty then
          var id: Value = ujson.Null
          try
            val msg = decode(ujson.read(line).obj)
            val key = decode(msg("id").str)
            id = Str(key)
            val kind = decode(msg("type").str)
            val params = msg.getOrElse("params", Obj())
            if params.objOpt.isEmpty then invalid("params must be an object")
            if msg.contains("image_password") || msg.contains("allow_plaintext_video") then unsupported("protected image transport")
            val owner = decode(msg.get("owner_id").map(_.str))
            kind match
              case "shutdown" =>
                stopJobs()
                result(key, Obj())
                running = false
              case "ping" => result(key, hello)
              case "list_samplers" => result(key, Json.toJs(samplerList))
              case "list_models" => result(key, models(params))
              case "list_files" => result(key, Obj("files" -> Arr()))
              case "list_devices" => result(key, Obj("devices" -> Arr.from(adapters.map(d => Obj("id" -> d.id(), "name" -> d.name())))))
              case "system_status" => result(key, Obj("loaded" -> loaded.nonEmpty, "device" -> display(device), "dry_run" -> false))
              case "memory" => unsupported("adapter memory telemetry")
              case "interruption_protection" => lock.synchronized {
                params.obj.get("enabled").foreach { v =>
                  val enabled = decode(v.bool)
                  if protection.enabled != enabled then protection = InterruptionProtection(enabled, protection.revision + 1)
                }
                result(key, Json.toJs(protection))
              }
              case "restart_authorization" | "interrupt" => lock.synchronized {
                val target = decode(params.obj.get("target_id").map(_.str))
                val expectedOwner = decode(params.obj.get("owner_id").map(_.str))
                val job = jobs.values.find(j => target.forall(_ == j.id) && expectedOwner.forall(j.owner.contains))
                val canCancel = kind == "interrupt" && !protection.enabled
                val tooLate = canCancel && job.exists(j => j.running && j.kind != "txt2img")
                val affected = canCancel && !tooLate && job.isDefined
                if affected then job.foreach(cancel)
                val decision = InterruptionDecision(!protection.enabled && !tooLate, protection.enabled, protection,
                  affected, Option.when(protection.enabled)("Interruptions are blocked"), tooLate)
                val response = Json.toJs(decision)
                response("active_id") = job.filter(j => affected && j.running).fold[Value](ujson.Null)(j => Str(j.id))
                result(key, response)
              }
              case "txt2img" | "load_model" | "unload" | "set_vae" =>
                val job = new Job(key, kind, owner)
                lock.synchronized {
                  if jobs.contains(key) then invalid("Duplicate in-flight request id")
                  jobs(key) = job
                  executor.execute(() => execute(job, params))
                }
              case _ => unsupported(kind)
          catch case NonFatal(e) => error(id, e)
    finally
      // On EOF release the context only after all admitted native calls have returned.
      stopJobs()
      executor.close()

  private def cancel(job: Job): Unit =
    job.cancelled = true
    if job.running && job.kind == "txt2img" then backend.cancel()

  private def stopJobs(): Unit = lock.synchronized { jobs.values.foreach(cancel) }

  private def execute(job: Job, raw: Value): Unit =
    try
      lock.synchronized { job.check(); job.running = true }
      val response = job.kind match
        case "load_model" => load(raw)
        case "unload" =>
          backend.unload()
          loaded = None
          Obj()
        case "set_vae" =>
          if raw.obj.get("path").exists(_ != ujson.Null) then unsupported("external VAE")
          Json.toJs(currentModel)
        case "txt2img" => generate(job, generation(raw))
      lock.synchronized {
        job.check()
        result(job.id, response)
        jobs.remove(job.id)
      }
    catch case e if NonFatal(e) || e.isInstanceOf[OutOfMemoryError] =>
      job.files.foreach(p => try Files.deleteIfExists(p) catch case NonFatal(ex) => System.err.println(ex.getMessage))
      lock.synchronized {
        error(Str(job.id), if job.cancelled then CancellationException() else e)
        jobs.remove(job.id)
      }

  private def currentModel: LoadedModel =
    loaded.getOrElse(throw EngineError("no_model", "Load a checkpoint before generation"))

  private def load(raw: Value): Value =
    val req = decode(Json.fromJs[LoadModelRequest](raw))
    checkFields(raw, Json.toJs(req))
    if req.vaePath.nonEmpty || req.vaeTiling || req.vaeSlicing || req.vaeFp16 then unsupported("custom VAE settings")
    if req.fp8Storage then unsupported("FP8 storage")
    if !Set("gpu", "auto")(req.vramMode) then unsupported(s"VRAM mode ${req.vramMode}")
    val weights = req.dtype match
      case "fp16" => "f16"
      case "fp32" => "f32"
      case other => unsupported(s"dtype $other")
    val path = Path.of(req.path).toAbsolutePath.normalize()
    if !Files.isRegularFile(path) || !checkpoint(path) then invalid(s"A .safetensors checkpoint file is required: $path")
    val start = System.nanoTime()
    try
      val hash = BrowserFiles.sha256(path).take(10)
      backend.load(path, device.id(), weights)
      val kind = backend.modelVersion() match
        case "SD 1.x" => "sd15"
        case "SDXL" => "sdxl"
        case other => other
      val model = LoadedModel(path.getFileName.toString.dropRight(".safetensors".length), path.toString,
        kind, hash, (System.nanoTime() - start) / 1000000, modelToken = Some(UUID.randomUUID().toString))
      loaded = Some(model)
      Json.toJs(model)
    catch
      case e: VulkanInference.NativeFailure => throw e
      case NonFatal(e) => throw EngineError("model_load_failed", s"${display(device)}: ${e.getMessage}")

  private def generate(job: Job, gen: Generation): Value =
    val model = currentModel
    val req = gen.request
    val p = req.params
    val dir = Files.createDirectories(Path.of(req.outputDir).toAbsolutePath.normalize())
    val start = System.nanoTime()
    val firstSeed = if p.seed >= 0 then p.seed else ThreadLocalRandom.current().nextLong(1L << 32)
    val imgs = (0 until p.imagesPerJob).map { i =>
      job.check()
      val seed = Math.addExact(firstSeed, i.toLong)
      val image = backend.generate(p.prompt, p.negativePrompt, p.width, p.height, p.steps, p.cfgScale, seed,
        samplers(p.samplerName), schedulers(gen.scheduler), p.clipSkip, (stage, step, total) =>
          if job.cancelled then backend.cancel()
          else emit(Obj("id" -> job.id, "type" -> "event", "event" -> "progress", "step" -> step, "total" -> total,
            "pass" -> stage, "image" -> i, "images" -> p.imagesPerJob, "batch_size" -> 1)))
      job.check()
      val file = Files.createTempFile(dir, "vulkan-", ".png")
      job.files += file
      if !ImageIO.write(image, "png", file.toFile) then throw IllegalStateException("PNG writer unavailable")
      val hardware = Option.when(req.hardwareInfo.enabled)(HardwareInfo.Record(display(device),
        f"${(System.nanoTime() - start) / 1e9}%.2fs", Option(req.hardwareInfo.remark).filter(_.nonEmpty)))
      GeneratedImage(file.toString, seed, width = image.getWidth, height = image.getHeight, hardwareInfo = hardware,
        infotextFields = Seq("Vulkan device" -> display(device)))
    }
    Json.toJs(GenerationResult(imgs, (System.nanoTime() - start) / 1e9, Some(model.name), Some(model.hash),
      scheduler = Some(gen.scheduler)))
