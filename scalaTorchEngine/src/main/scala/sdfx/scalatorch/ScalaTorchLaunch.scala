package sdfx.engine.scalatorch

import sdfx.engine.{EngineBackendProvider,EngineLaunch}

import java.nio.file.{Files,Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Launch adapter for the Python-free worker, shared by desktop and MCP. */
object ScalaTorchLaunch:
  private def exeName: String = if sys.props("os.name").startsWith("Windows") then "java.exe" else "java"
  private val keys=Set("SDFX_SCALATORCH_CLASSPATH_FILE","SDFX_SCALATORCH_MODEL",
    "SDFX_SCALATORCH_SCHEDULER")
  /** `scala-torch` folder of the installation that contains the running jar (`<install>/app/<jar>` -> `<install>/scala-torch`).
    * Needed because the user data home (`--home`, MCP) can be anywhere while the bundled worker pieces stay with the program. */
  lazy val installScalaTorch: Option[Path] =
    try
      val jar=Path.of(classOf[ScalaTorchLaunchMarker].getProtectionDomain.getCodeSource.getLocation.toURI).toAbsolutePath.normalize()
      Option(jar.getParent).flatMap(a => Option(a.getParent)).map(_.resolve("scala-torch")).filter(Files.isDirectory(_))
    catch case _: Throwable => None

  /** True when an NVIDIA driver is installed (nvcuda.dll on Windows, libcuda.so on Linux). */
  lazy val cudaAvailable: Boolean =
    if sys.props.getOrElse("os.name","").startsWith("Windows") then
      Files.isRegularFile(Path.of(sys.env.getOrElse("SystemRoot","C:\\Windows"),"System32","nvcuda.dll"))
    else Seq("/usr/lib/x86_64-linux-gnu/libcuda.so.1","/usr/lib64/libcuda.so.1","/usr/lib/libcuda.so.1").exists(p => Files.exists(Path.of(p)))

  /** Model the worker starts with when none is configured: the last used checkpoint, else the first checkpoint in the model folders.
    * The model can be changed later through load_model. */
  def defaultModel(settings: sdfx.core.config.Settings,home: Option[Path]=None): Option[String] =
    val dirs=(settings.modelDirs.checkpoints.map(Path.of(_)) ++ home.toSeq.map(_.resolve("models").resolve("Stable-diffusion"))).filter(Files.isDirectory(_))
    def files(d: Path)=try Files.walk(d,3).iterator().asScala.filter(f => Files.isRegularFile(f) &&
      Vector(".safetensors",".ckpt").exists(f.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith)).toVector.sortBy(_.toString)
      catch case _: Throwable => Vector.empty[Path]
    val all=dirs.flatMap(files)
    settings.lastModel.flatMap(n => all.find(f => f.getFileName.toString==n || f.getFileName.toString.stripSuffix(".safetensors").stripSuffix(".ckpt")==n))
      .orElse(all.headOption).map(_.toString)

  def requested(env: Map[String,String]=sys.env): Boolean=keys.exists(env.contains) ||
    sys.props.get("sdfx.engine.backend").exists(_.trim.equalsIgnoreCase("scala-torch"))

  private def isJava25Home(dir: Path): Boolean =
    val name=Option(dir.getFileName).map(_.toString).getOrElse("")
    name.contains("25") || {
      val rel=dir.resolve("release")
      Files.isRegularFile(rel) && (try Files.readString(rel).contains("JAVA_VERSION=\"25") catch case _: Throwable => false)
    }

  private def resolveJavaExe(p: Path): Path =
    if Files.isDirectory(p) then
      Seq(p.resolve(exeName),p.resolve("bin").resolve(exeName),p.resolve("Contents/Home/bin").resolve(exeName))
        .find(Files.isRegularFile(_)).getOrElse(p)
    else p

  private def discoverJava25(env: Map[String,String],bundled: Seq[Path]): Path =
    bundled.map(_.resolve("jre/bin").resolve(exeName)).find(Files.isRegularFile(_))
      .orElse {
        Option.when(Runtime.version().feature()>=25)(Path.of(sys.props("java.home"),"bin",exeName))
          .filter(Files.isRegularFile(_))
      }
      .orElse {
        Seq("JAVA25_HOME","JAVA_HOME").flatMap(env.get).map(_.trim).filter(_.nonEmpty).flatMap { raw =>
          val d=Path.of(raw).toAbsolutePath.normalize()
          val exe=d.resolve("bin").resolve(exeName)
          Option.when(Files.isRegularFile(exe) && isJava25Home(d))(exe)
        }.headOption
      }
      .orElse {
        val roots=Seq(
          Path.of("C:\\dev\\Java"),
          Path.of("C:\\Program Files\\Java"),
          Path.of("C:\\Program Files\\Eclipse Adoptium"),
          Path.of("/usr/lib/jvm"),
          Path.of("/Library/Java/JavaVirtualMachines")
        ).filter(Files.isDirectory(_))
        roots.flatMap { root =>
          val stream=Files.list(root)
          try
            stream.iterator().asScala.filter(Files.isDirectory(_)).toVector
              .filter(d => d.getFileName.toString.contains("25"))
              .sortBy(_.getFileName.toString)(Ordering[String].reverse)
              .flatMap { d =>
                val home=if Files.isDirectory(d.resolve("Contents/Home")) then d.resolve("Contents/Home") else d
                val exe=home.resolve("bin").resolve(exeName)
                Option.when(Files.isRegularFile(exe) && isJava25Home(home))(exe)
              }
          catch case _: Throwable => Vector.empty
          finally stream.close()
        }.headOption
      }
      .getOrElse(Path.of(sys.props("java.home"),"bin",exeName))

  private def managedClasspathEntries(resource: String="scalatorch/models/worker/NativeWorkerMain.class",
      anchorClass: Class[?]=classOf[ScalaTorchLaunchMarker]): Option[String] =
    sdfx.engine.WorkerClasspath.forResource(resource, anchorClass)

  def dryRunLaunch(home: sdfx.core.config.AppHome): EngineLaunch =
    val bundled = (Seq(home.root.resolve("scala-torch")) ++ installScalaTorch.toSeq).distinct
    val currentJava = Path.of(sys.props("java.home"), "bin", exeName)
    val javaExe =
      if Files.isRegularFile(currentJava) then currentJava
      else discoverJava25(sys.env, bundled)
    require(Files.isRegularFile(javaExe), "Java executable is missing for dry-run worker")
    val cp=managedClasspathEntries("sdfx/scalatorch/NativeDryRunWorker.class",classOf[sdfx.scalatorch.NativeDryRunWorker.type])
      .getOrElse(throw IllegalArgumentException("NativeDryRunWorker is missing from classpath"))
    val workdir=if Files.isDirectory(home.root) then home.root else Path.of(".").toAbsolutePath.normalize()
    EngineLaunch(Seq(javaExe.toString,"-Djava.awt.headless=true","-cp",cp,"sdfx.scalatorch.NativeDryRunWorker"),
      workdir,helloTimeout=15.seconds)

  def fromEnvironment(env: Map[String,String]=sys.env,appHome: Option[Path]=None): Option[EngineLaunch]=
    if !requested(env) then None
    else
      def configured(name: String,default: Option[Path]): Path=env.get(name) match
        case Some(value) =>
          require(value.trim.nonEmpty,s"native engine requires $name")
          Path.of(value.trim).toAbsolutePath.normalize()
        case None => default.filter(Files.exists(_)).map(_.toAbsolutePath.normalize()).getOrElse(
          throw IllegalArgumentException(s"native engine requires $name"))
      val rawModel=env.getOrElse("SDFX_SCALATORCH_MODEL","").trim
      require(rawModel.nonEmpty,"native engine requires SDFX_SCALATORCH_MODEL")
      val idle=rawModel=="-"
      val modelOpt=if idle then None else Some(Path.of(rawModel).toAbsolutePath.normalize())
      val bundled=appHome.toSeq.map(_.resolve("scala-torch")) ++ installScalaTorch.toSeq
      val defaultCpFile=bundled.map(_.resolve("worker-classpath.txt")).find(Files.exists(_))
      val explicitCpFile=if env.contains("SDFX_SCALATORCH_CLASSPATH_FILE") || defaultCpFile.nonEmpty then
        Some(configured("SDFX_SCALATORCH_CLASSPATH_FILE",defaultCpFile))
      else None
      val singleFile=modelOpt.exists(model => Files.isRegularFile(model) && Vector(".safetensors",".ckpt").exists(
        model.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith))
      val config=if idle || (singleFile && !env.contains("SDFX_SCALATORCH_SCHEDULER")) then None else
        Some(configured("SDFX_SCALATORCH_SCHEDULER",modelOpt.map(_.resolve("scheduler/scheduler_config.json"))))
      explicitCpFile.foreach { cpFile =>
        require(Files.isRegularFile(cpFile) && Files.size(cpFile)<=1048576,"native classpath file is missing or too large")
      }
      modelOpt.foreach(model => require(Files.isDirectory(model) || singleFile,"native model directory or checkpoint file is missing"))
      require(config.forall(Files.isRegularFile(_)),"native scheduler config is missing")
      val tokenizer=if !singleFile then None else
        val explicit=env.get("SDFX_SCALATORCH_TOKENIZER").map { value =>
          require(value.trim.nonEmpty,"native tokenizer path is empty")
          Path.of(value.trim).toAbsolutePath.normalize()
        }
        explicit.foreach { p =>
          require(Files.isRegularFile(p.resolve("vocab.json")) && Files.isRegularFile(p.resolve("merges.txt")),
            "configured native tokenizer is missing vocabulary or merges")
        }
        explicit.orElse {
          val candidates=bundled.map(_.resolve("tokenizer")) ++ modelOpt.map(_.getParent.resolve("tokenizer")).toSeq
          candidates.find(path => Files.isRegularFile(path.resolve("vocab.json")) && Files.isRegularFile(path.resolve("merges.txt")))
        }
      val classpathEntries=explicitCpFile match
        case Some(cpFile) =>
          val classpath=Files.readString(cpFile).trim
          require(classpath.nonEmpty && !classpath.contains('\n') && !classpath.contains('\r'),"native classpath must be one nonempty line")
          classpath.split(java.util.regex.Pattern.quote(java.io.File.pathSeparator),-1).map { entry =>
            require(entry.nonEmpty,s"native classpath contains an empty entry: $entry")
            val path=Path.of(entry)
            val resolved=(if path.isAbsolute then path else cpFile.getParent.resolve(path)).toAbsolutePath.normalize()
            require(Files.exists(resolved),s"native classpath entry missing: $resolved")
            resolved.toString
          }.mkString(java.io.File.pathSeparator)
        case None =>
          managedClasspathEntries().getOrElse(
            throw IllegalArgumentException("native engine requires SDFX_SCALATORCH_CLASSPATH_FILE or scala-torch-models on the classpath"))
      val javaExe=env.get("SDFX_SCALATORCH_JAVA").map { value =>
        require(value.trim.nonEmpty,"native engine requires SDFX_SCALATORCH_JAVA")
        resolveJavaExe(Path.of(value.trim).toAbsolutePath.normalize())
      }.getOrElse(discoverJava25(env,bundled))
      require(Files.isRegularFile(javaExe),"native Java executable is missing (Java 25 required)")
      val workdir=modelOpt.map(m => if singleFile then m.getParent else m)
        .orElse(appHome.map(_.toAbsolutePath.normalize()).filter(Files.isDirectory(_)))
        .getOrElse(Path.of(".").toAbsolutePath.normalize())
      Some(EngineLaunch(Seq(javaExe.toString,"-XX:MaxRAMPercentage=50.0","--enable-native-access=ALL-UNNAMED","-cp",classpathEntries,
        "scalatorch.models.worker.NativeWorkerMain",modelOpt.fold("-")(_.toString),config.fold("-")(_.toString)),
        workdir,
        env=env.filter((key,_) => key.startsWith("SDFX_SCALATORCH_") || key.startsWith("SCALATORCH_")) ++
          tokenizer.map(path => "SDFX_SCALATORCH_TOKENIZER" -> path.toString),helloTimeout=30.seconds))

private final class ScalaTorchLaunchMarker

final class ScalaTorchBackendProvider extends EngineBackendProvider:
  val name="scala-torch"
  def requested(environment: Map[String,String]): Boolean=ScalaTorchLaunch.requested(environment)
  def usesNativeBackend(settings: sdfx.core.config.Settings,environment: Map[String,String]): Boolean =
    environment.get("SDFX_ENGINE_BACKEND") match
      case Some(value) => value.trim.equalsIgnoreCase("scala-torch")
      case None => settings.engineBackend match
        case "scala-torch" => true
        case "python" => false
        case _ => ScalaTorchLaunch.requested(environment)
  def launch(home: sdfx.core.config.AppHome,settings: sdfx.core.config.Settings,dryRun: Boolean,
    environment: Map[String,String]): EngineLaunch =
    if dryRun then return ScalaTorchLaunch.dryRunLaunch(home)
    def device(value: String): String =
      if value != "auto" then value else if ScalaTorchLaunch.cudaAvailable then "cuda" else "cpu"
    val model=settings.scalaTorchModel.orElse(ScalaTorchLaunch.defaultModel(settings,Some(home.root))).orElse(Some("-"))
    val saved=Seq(
      "SDFX_SCALATORCH_CLASSPATH_FILE"->settings.scalaTorchClasspath,
      "SDFX_SCALATORCH_MODEL"->model,
      "SDFX_SCALATORCH_SCHEDULER"->settings.scalaTorchScheduler,
      "SDFX_SCALATORCH_JAVA"->settings.scalaTorchJava,
      // read by the worker's CUDA layer itself (no SDFX_ prefix)
      "SCALATORCH_CUDNN_PATH"->settings.scalaTorchCudnnDir.map(_.trim).filter(_.nonEmpty),
      "SCALATORCH_CUDA_VRAM_LIMIT_MB"->settings.scalaTorchVramLimitMb.filter(_ > 0).map(_.toString),
      "SDFX_SCALATORCH_UNET_DEVICE"->Some(device(settings.scalaTorchUnetDevice)),
      "SDFX_SCALATORCH_VAE_DEVICE"->Some(device(settings.scalaTorchVaeDevice)),
      "SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS"->Some(settings.scalaTorchUnetOffloadWeights.toString),
      "SDFX_SCALATORCH_VAE_OFFLOAD_WEIGHTS"->Some(settings.scalaTorchVaeOffloadWeights.toString),
      // CUDA memory budget of a session: "auto" = 92 % of what the GPU has free when the session opens. A fixed number failed with
      // "CUDA session allocation budget exceeded" for jobs that need more than it (SDXL + LoRA, fp32 SDXL, big images, ControlNet).
      "SDFX_SCALATORCH_UNET_MAX_BYTES"->Some("auto"),
      "SDFX_SCALATORCH_VAE_MAX_BYTES"->Some("auto"),
      "SDFX_SCALATORCH_MAX_ATTENTION_SCORE_ELEMENTS"->Some((1L<<30).toString))
    val configured=saved.foldLeft(environment) { case (env,(key,value)) =>
      if env.contains(key) then env else value.map(v => env.updated(key,v)).getOrElse(env)
    }
    System.err.println(s"[sdfx] scala-torch: U-Net ${configured("SDFX_SCALATORCH_UNET_DEVICE")}, VAE ${configured("SDFX_SCALATORCH_VAE_DEVICE")}, precision ${settings.dtype}" +
      (if configured("SDFX_SCALATORCH_UNET_DEVICE")=="cpu" then " (CPU execution is very slow; choose cuda or auto in Settings)" else ""))
    ScalaTorchLaunch.fromEnvironment(configured,Some(home.root)).getOrElse(
      throw new IllegalArgumentException("scala-torch backend requires its classpath, model and scheduler configuration"))
