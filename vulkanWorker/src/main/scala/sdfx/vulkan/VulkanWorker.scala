package sdfx.vulkan

import java.awt.image.BufferedImage
import java.io.{BufferedReader, InputStreamReader, OutputStream, OutputStreamWriter}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.nio.file.attribute.BasicFileAttributes
import java.util.UUID
import java.util.concurrent.{CancellationException, Executors, ThreadLocalRandom}
import javax.imageio.ImageIO
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import sdfx.core.imagebrowser.BrowserFiles
import sdfx.core.infotext.{HardwareInfo, Img2ImgHiresMetadata}
import sdfx.core.model.Img2ImgParams
import sdfx.core.protocol.*
import sdfx.vulkan.tools.VulkanTools
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
  private var activeResourceFiles = Map.empty[String,VulkanHashes.File]
  private val toolSession = new VulkanTools.Session
  private val wdBatch = new sdfx.vulkan.tools.VulkanWdBatch
  private val wdCommands = Set("caption","batch_interrogate_prepare","batch_interrogate_execute")
  private val controlModels = new sdfx.vulkan.tools.VulkanControlModels
  private val adapters = backend.devices().asScala.toVector
  private val device = select(adapters, selection)
  private val writer = new OutputStreamWriter(out, UTF_8)
  private val lock = new Object
  private val executor = Executors.newSingleThreadExecutor()
  private val jobs = mutable.LinkedHashMap.empty[String, Job]
  private var protection = InterruptionProtection()
  @volatile private var loaded = Option.empty[LoadedModel]
  private val benchmarks = new sdfx.vulkan.tools.VulkanBenchmarks
  private val benchmarkCommands = Set("system_benchmark", "vram_estimator")
  @volatile private var loadRequest = Option.empty[LoadModelRequest]
  private var activeOptions = Option.empty[VulkanInference.ModelOptions]

  private final class Job(val id: String, val kind: String, val owner: Option[String], val imagePassword: Option[String]):
    @volatile var sensorFailure = Option.empty[Throwable]
    @volatile var cancelled = false
    @volatile var interrupts = 0
    var frames = Option.empty[sdfx.vulkan.tools.VulkanFrames]
    var commit: () => Unit = () => ()
    var publishing = false
    var running = false
    val files = mutable.ArrayBuffer.empty[Path]
    def check(): Unit =
      if cancelled then throw CancellationException()
      sensorFailure.foreach(e => throw e)

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
    "features" -> Arr.from(Vector("txt2img", "aesthetic_embedding", "aesthetic_score", "lora", "embeddings", "external_vae", "controlnet", "img2img", "animatediff", "animatediff_encode", "photomaker", "inpaint", "hires_fix", "img2img_hires_fix", "temporal_lab", "vae_tools", "list_devices", "system_info", "vram_estimator", "prevent_interrupt", VulkanClipIo.id, VulkanFusionPrompt.id, VulkanWeightGradient.id) ++ VulkanTools.features ++
      Vector(VulkanIncantations.id) ++
      (if backend.supportsUnetHooks() then VulkanUnetHooks.attentionHooks.toVector.sorted else Vector.empty) ++
      (if backend.supportsUnetResize() then VulkanUnetHooks.resizeHooks.toVector.sorted else Vector.empty) ++
      Option.when(backend.supportsTiledDiffusion())(VulkanUnetHooks.tiledDiffusion).toVector ++
      Option.when(backend.supportsLatentModifier())(VulkanLatentModifier.id).toVector ++
      Option.when(backend.supportsNegPip())(VulkanNegPip.id).toVector ++
      Option.when(backend.supportsFooocusInpaint())(VulkanFooocusInpaint.id).toVector ++
      Option.when(backend.supportsCdTuner())(VulkanCdTuner.id).toVector ++
      Option.when(backend.supportsTGate())(VulkanTGate.id).toVector ++
      Option.when(backend.supportsDeepCache())(VulkanDeepCache.id).toVector ++
      Option.when(backend.supportsReferenceOnly())(VulkanReference.id).toVector ++
      Option.when(backend.supportsLLLite())(VulkanLLLite.id).toVector ++
      Option.when(backend.supportsLlul())(VulkanLlul.id).toVector ++
      Option.when(backend.supportsCharacteristic())(VulkanCharacteristic.id).toVector ++
      Option.when(backend.supportsComposableLora())(VulkanComposableLora.id).toVector ++
      Option.when(backend.supportsAestheticTraining())(VulkanGraph.aestheticHook).toVector ++
      Option.when(backend.supportsAgentAttention())(VulkanAgentAttention.id).toVector ++
      Option.when(backend.supportsDynamicThreshold())("dynamic_thresholding") ++
      (if backend.supportsSvd() then Vector("img2vid","novel_view") else Vector.empty) ++ Option.when(sdfx.vulkan.tools.VulkanCharaImport.available)("chara_searcher") ++ Option.when(sdfx.vulkan.tools.VulkanBlip.available && !sdfx.vulkan.tools.VulkanWdTagger.available)("caption") ++ Option.when(sdfx.vulkan.tools.VulkanCustomCode.enabled)("custom_code") ++ sdfx.vulkan.tools.VulkanWdTagger.features ++ Option.when(sdfx.vulkan.tools.VulkanFaceRestore.available)("face_restore") ++ Vector("vulkan_nsfw") ++ Option.when(sdfx.vulkan.tools.VulkanSafety.available)("nsfw_checker")))

  private def status =
    val data = VulkanDiagnostics.status(device, backend.memory(device.id()), loaded)
    loadRequest.foreach { r =>
      data("runtime")("dtype") = r.dtype
      data("runtime")("vram_mode") = r.vramMode
      data("runtime")("vae_tiling") = r.vaeTiling
    }
    data

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
            val password = decode(msg.get("image_password").map(_.str))
            if password.exists(_.isEmpty) || password.isDefined && !isGeneration(kind) && !Set("vae_tools","temporal_lab","controlnet_fastload_data","batch_interrogate_execute")(kind) then
              invalid("Invalid image password transport")
            if password.isDefined && kind == "controlnet_fastload_data" && params.obj.get("operation").flatMap(_.strOpt).exists(Set("save","encode")) then
              invalid("Protected sessions cannot publish readable Fastload data")
            if msg.get("allow_plaintext_video").exists(v => v.boolOpt.isEmpty || !isGeneration(kind) || password.isEmpty) then
              invalid("Invalid plaintext video permission")
            val owner = decode(msg.get("owner_id").map(_.str))
            kind match
              case "shutdown" =>
                stopJobs()
                result(key, Obj())
                running = false
              case "ping" => result(key, hello)
              case "list_samplers" => result(key, Json.toJs(samplerList))
              case "list_models" => result(key, models(params))
              case "list_devices" => result(key, Obj("devices" -> Arr.from(adapters.map(d => Obj("id" -> d.id(), "name" -> d.name())))))
              case "system_status" => result(key, status)
              case "memory" => result(key, Json.toJs(VulkanDiagnostics.memory(backend.memory(device.id()))))
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
                val tooLate = canCancel && job.exists(j => j.running && (j.publishing || (!isGeneration(j.kind) && !benchmarkCommands(j.kind) && j.kind != "system_info" && j.kind != "run_code" && j.kind != "novel_view" && j.kind != "img2vid" && j.kind != "chara_searcher" && j.kind != "aesthetic_embedding" && j.kind != "clip_interrogator" && j.kind != "restore_faces" && j.kind != "vae_tools" && j.kind != "temporal_lab" && !wdCommands(j.kind) && !VulkanTools.commands(j.kind))))
                val affected = canCancel && !tooLate && job.isDefined
                if affected then job.foreach(cancel)
                val decision = InterruptionDecision(!protection.enabled && !tooLate, protection.enabled, protection,
                  affected, Option.when(protection.enabled)("Interruptions are blocked"), tooLate)
                val response = Json.toJs(decision)
                response("active_id") = job.filter(j => affected && j.running).fold[Value](ujson.Null)(j => Str(j.id))
                result(key, response)
              }
              case k if wdCommands(k) || VulkanTools.commands(k) || benchmarkCommands(k) || Set("run_code", "novel_view", "img2vid", "chara_searcher", "restore_faces", "txt2img", "img2img", "aesthetic_embedding", "clip_interrogator", "temporal_lab", "vae_tools", "system_info", "load_model", "unload", "set_vae")(k) =>
                val job = new Job(key, kind, owner, password)
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
      try backend.unload() finally controlModels.close()

  private def isGeneration(kind: String): Boolean = kind == "txt2img" || kind == "img2img"

  private def cancel(job: Job): Unit =
    if job.publishing then return
    job.interrupts += 1
    job.cancelled = true
    if job.running && (isGeneration(job.kind) || job.kind == "run_code" || job.kind == "novel_view" || job.kind == "img2vid" || job.kind == "vae_tools" || job.kind == "clip_interrogator" || job.kind == "aesthetic_embedding" || job.kind == "batch_interrogate_execute" || benchmarkCommands(job.kind)) then backend.cancel()

  private def stopJobs(): Unit = lock.synchronized { jobs.values.foreach(cancel) }

  private def modelUpscale(path: Path, image: BufferedImage, width: Int, height: Int): BufferedImage =
    val scaled = backend.upscale(path, device.id(), image, width, height)
    sdfx.vulkan.tools.DiffusionImageResize.lanczos(scaled, width, height)

  private def execute(job: Job, raw: Value): Unit =
    try
      lock.synchronized { job.check(); job.running = true }
      val response = job.kind match
        case "run_code" => sdfx.vulkan.tools.VulkanCustomCode.run(raw,device.name(),loaded.map(_.path),() => job.check(),
          value => generate(job,generation(value)))
        case "novel_view" =>
          val image = sdfx.vulkan.tools.VulkanNovelView.render(raw,backend,device.id(),() => job.check(),
            (done,total) => emit(Obj("id" -> job.id,"type" -> "event","event" -> "progress","step" -> done,"total" -> total,"pass" -> "base")))
          val dir = Files.createDirectories(Path.of(raw("output_dir").str).toAbsolutePath)
          val output = Files.createTempFile(dir,"novel-view-",".png")
          job.files += output
          sdfx.core.png.ScrambledPng.write(image,output,Seq.empty,job.imagePassword)
          lock.synchronized { job.check(); job.publishing = true }
          Obj("path" -> output.toString)
        case "img2vid" =>
          if !backend.supportsSvd() then unsupported("This Vulkan runtime does not include the SVD temporal model")
          val frames = VulkanSvd.render(raw, backend, device.id(), () => job.check(), (step,total) =>
            emit(Obj("id" -> job.id, "type" -> "event", "event" -> "progress", "step" -> step, "total" -> total, "pass" -> "base")))
          val dir = Files.createDirectories(Path.of(raw("output_dir").str).toAbsolutePath)
          val gif = Files.createTempFile(dir, "svd-", ".gif")
          job.files += gif
          val fps = raw.obj.get("fps").fold(6)(_.num.toInt)
          sdfx.core.image.Gif.writeSequence(frames.iterator.map { frame => job.check(); frame }, gif,
            math.round(1000.0 / fps).toInt, Some(0))
          val preview = Files.createTempFile(dir, "svd-frame0-", ".png")
          job.files += preview
          sdfx.core.png.ScrambledPng.write(frames.head, preview, Seq.empty, job.imagePassword)
          lock.synchronized { job.check(); job.publishing = true }
          Obj("path" -> gif.toString, "frames" -> frames.size, "preview" -> preview.toString)
        case "temporal_lab" =>
          sdfx.vulkan.tools.VulkanTemporal.execute(raw, () => job.check(), () => lock.synchronized {
            job.check()
            job.publishing = true
          }, (step,total,phase) => emit(Obj("id" -> job.id,"type" -> "event","event" -> "progress",
            "step" -> step,"total" -> total,"pass" -> phase)),job.imagePassword)
        case "restore_faces" =>
          val source = sdfx.vulkan.tools.DiffusionImageInput.readRgb(Path.of(raw("image_path").str))
          val restored = sdfx.vulkan.tools.VulkanFaceRestore(source,raw,device.name(),() => job.check())
          val dir = Files.createDirectories(Path.of(raw("output_dir").str).toAbsolutePath)
          val output = Files.createTempFile(dir,"faces-",".png")
          job.files += output
          sdfx.core.png.ScrambledPng.write(restored,output,Seq.empty,job.imagePassword)
          lock.synchronized { job.check(); job.publishing = true }
          Obj("path" -> output.toString)
        case "image_tool" if raw.obj.get("tool").contains(ujson.Str("nsfw_checker")) =>
          val dir=Path.of(raw.obj.get("output_dir").fold(".")(_.str)).toAbsolutePath
          val output=new sdfx.vulkan.tools.VulkanSafety(backend,device.id(),device.name(),()=>job.check()).tool(raw,dir,job.imagePassword)
          lock.synchronized { job.check(); job.publishing=true }
          output
        case "image_tool" if raw.obj.get("tool").contains(ujson.Str("aesthetic_score")) =>
          val image = sdfx.vulkan.tools.DiffusionImageInput.readRgb(Path.of(raw("image_path").str))
          val result = scoreImage(image,raw,job)
          val dir = Files.createDirectories(Path.of(raw.obj.get("output_dir").fold(".")(_.str)).toAbsolutePath)
          val output = Files.createTempFile(dir,"aesthetic-",".png")
          job.files += output
          sdfx.core.png.ScrambledPng.write(image,output,result.chunks,job.imagePassword)
          lock.synchronized { job.check(); job.publishing = true }
          Obj("path"->output.toString,"outputs"->Obj(),"info"->Obj.from(result.fields.map((k,v)=>k->ujson.Str(v))))
        case "caption" if raw.obj.get("method").fold("blip")(_.str)=="blip" =>
          val catalog=sdfx.vulkan.tools.VulkanBlip
          val name=sys.env.get("SDFX_BLIP_MODEL").map {
            case "Salesforce/blip-image-captioning-base" => "blip-base"
            case "Salesforce/blip-image-captioning-large" => "blip-large"
            case other => other
          }.getOrElse(catalog.known.headOption.getOrElse("blip-base"))
          Obj("caption"->catalog.caption(Path.of(raw("image_path").str),device.name(),()=>job.check(),60,name))
        case "caption" => sdfx.vulkan.tools.VulkanWdTagger.caption(raw,device.name(),() => job.check())
        case k @ ("batch_interrogate_prepare" | "batch_interrogate_execute") =>
          val owner = job.owner.getOrElse(throw IllegalArgumentException("WD batch owner required"))
          def publish():Unit = lock.synchronized { job.check(); job.publishing = true }
          if k == "batch_interrogate_prepare" then wdBatch.prepare(raw,owner,currentModel,()=>job.check(),()=>publish())
          else wdBatch.execute(raw,owner,currentModel,device.name(),job.imagePassword,()=>job.check(),()=>publish(), r=>generate(job,generation(r)))
        case "chara_searcher" if raw.obj.get("action").contains(ujson.Str("import")) =>
          sdfx.vulkan.tools.VulkanCharaImport.execute(raw,backend,device.id(),device.name(),()=>job.check(),()=>lock.synchronized {
            job.check();job.publishing=true
          },(step,total)=>emit(Obj("id"->job.id,"type"->"event","event"->"progress","pass"->"chara_searcher","step"->step,"total"->total)))
        case "chara_searcher" => sdfx.vulkan.tools.VulkanBaseClip.search(raw,backend,device.id(),()=>job.check())
        case "aesthetic_embedding" =>
          sdfx.vulkan.tools.VulkanAesthetic.execute(raw,backend,device.id(),() => job.check(),() => lock.synchronized {
            job.check()
            job.publishing = true
          },(stage,step,total) => emit(Obj("type" -> "event","id" -> job.id,"event" -> "progress","pass" -> stage,"step" -> step,"total" -> total)))
        case "clip_interrogator" =>
          sdfx.vulkan.tools.VulkanClip.execute(raw,backend,device.id(),() => job.check(),(stage,step,total) =>
            emit(Obj("type" -> "event","id" -> job.id,"event" -> "progress","pass" -> stage,"step" -> step,"total" -> total)), () => lock.synchronized { job.check(); job.publishing = true })
        case "vae_tools" =>
          sdfx.vulkan.tools.VulkanVae.execute(raw, backend, device.id(), currentModel, () => job.cancelled, () => lock.synchronized {
            job.check()
            job.publishing = true
          }, (name, fields) => {
            fields("event") = name
            fields("type") = "event"
            fields("id") = job.id
            emit(fields)
          }, job.imagePassword)
        case k if VulkanTools.commands(k) =>
          VulkanTools.execute(k, raw, () => job.cancelled, () => lock.synchronized {
            job.check()
            job.publishing = true
          }, (name, fields) => {
            fields("event") = name
            fields("type") = "event"
            fields("id") = job.id
            emit(fields)
          }, display(device), loaded.map(_.kind), modelUpscale, toolSession, loaded)
        case k if benchmarkCommands(k) =>
          benchmarks.run(k, raw, status, loadRequest, () => backend.memory(device.id()), () => job.cancelled,
            (name, fields) =>
              fields("event") = name
              fields("type") = "event"
              fields("id") = job.id
              emit(fields),
            (p, dir) =>
              val request = GenerationRequest(p, dir.toString, previewEvery = 0)
              Json.fromJs[GenerationResult](generate(job, generation(Json.toJs(request)))))
        case "system_info" => VulkanDiagnostics.inventory(raw, status, hello("features"), () => job.check())
        case "load_model" => load(raw)
        case "unload" =>
          backend.unload()
          loaded = None
          loadRequest = None
          activeOptions = None
          Obj()
        case "set_vae" =>
          val model = currentModel
          val path = decode(raw.obj.get("path").filterNot(_ == ujson.Null).map(_.str)).map(VulkanConditioning.file(_, "VAE"))
          if path == model.vaePath then Json.toJs(model)
          else loadModel(loadRequest.get.copy(vaePath = path))
        case "txt2img" | "img2img" =>
          sdfx.vulkan.tools.VulkanEmbeddings.withGeneration(raw, currentModel, () => job.cancelled) { prepared =>
            runGeneration(job,prepared)
          }
      lock.synchronized {
        job.check()
        job.commit()
        result(job.id, response)
        jobs.remove(job.id)
      }
    catch case e if NonFatal(e) || e.isInstanceOf[OutOfMemoryError] =>
      job.frames.foreach(_.incomplete(() => job.interrupts > 1))
      job.files.foreach(p => try Files.deleteIfExists(p) catch case NonFatal(ex) => System.err.println(ex.getMessage))
      lock.synchronized {
        error(Str(job.id), if job.cancelled then CancellationException() else job.sensorFailure.getOrElse(e))
        jobs.remove(job.id)
      }

  private def runGeneration(job: Job, raw: Value): Value =
    val gen = generation(raw)
    val req = gen.request
    val mode = req.previewMethod match
      case "none" => "none"
      case "approx" => "proj"
      // Decode with the active VAE when separate TAESD weights are unavailable.
      case "taesd" => if sdfx.vulkan.tools.VulkanVae.tinyPath(currentModel,None).isDefined then "tae" else "vae"
      case "vae" => "vae"
      case method => invalid(s"Unknown preview method: $method")
    job.frames = sdfx.vulkan.tools.VulkanFrames.start(req, () => job.cancelled, job.imagePassword)
    val recording = job.frames.map(_.config)
    val noisy = recording.exists(_.imageType == "Noisy")
    val preview = req.previewEvery > 0 && mode != "none"
    val decoder = recording.fold(mode)(c => if c.imageType == "Preview" && mode == "proj" then "proj" else "vae")
    Using.resource(backend.generationSettings(if recording.isDefined then 1 else req.previewEvery, decoder,
      req.params.eta.map(Double.box).orNull, preview || !noisy, noisy,
      (step, pass, passSteps, isNoisy, frame) =>
        if isNoisy == noisy then job.frames.foreach(_.frame(step,pass,passSteps,frame))
        if !job.cancelled && preview && !isNoisy && step % req.previewEvery == 0 then
          val bytes = new java.io.ByteArrayOutputStream()
          val visible = req.params.hooks.get("nsfw_checker").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).fold(frame){ cfg =>
            new sdfx.vulkan.tools.VulkanSafety(backend,device.id(),device.name(),()=>job.check()).filter(frame,cfg,true).image
          }
          if ImageIO.write(visible, "jpeg", bytes) then
            emit(Obj("id" -> job.id, "type" -> "event", "event" -> "preview", "step" -> step,
              "jpeg_b64" -> java.util.Base64.getEncoder.encodeToString(bytes.toByteArray))))) { _ =>
      generate(job, gen)
    }

  private def currentModel: LoadedModel =
    loaded.getOrElse(throw EngineError("no_model", "Load a checkpoint before generation"))

  private def load(raw: Value): Value =
    val req = decode(Json.fromJs[LoadModelRequest](raw))
    checkFields(raw, Json.toJs(req))
    loadModel(req)

  private def loadModel(req: LoadModelRequest): Value =
    val opts = VulkanConditioning.options(req, device.id())
    val path = Path.of(req.path).toAbsolutePath.normalize()
    if !Files.isRegularFile(path) || !checkpoint(path) then invalid(s"A supported checkpoint file is required: $path")
    val start = System.nanoTime()
    try
      val hash = BrowserFiles.sha256(path).take(10)
      val vae = Option(opts.vae()).filter(_.nonEmpty)
      val vaeHash = vae.map(p => BrowserFiles.sha256(Path.of(p)).take(10))
      backend.load(path, device.id(), opts)
      val kind = backend.modelVersion() match
        case "SD 1.x" => "sd15"
        case "SDXL" => "sdxl"
        case other => other
      val model = LoadedModel(checkpointName(path.getFileName.toString), path.toString,
        kind, hash, (System.nanoTime() - start) / 1000000, vae.map(VulkanConditioning.stem), vaeHash,
        modelToken = Some(UUID.randomUUID().toString), vaePath = vae)
      loaded = Some(model)
      loadRequest = Some(req.copy(path = path.toString, vaePath = vae))
      activeOptions = Some(opts)
      activeResourceFiles = Map.empty
      Json.toJs(model)
    catch
      case e: VulkanInference.NativeFailure => throw e
      case NonFatal(e) => throw EngineError("model_load_failed", s"${display(device)}: ${e.getMessage}")

  private def resolveControls(req: GenerationRequest, job: Job): Map[String,String] =
    val selected = req.params.controlnet.iterator.filter(_.enabled).map(_.model).toSet
    req.controlnetPaths.collect { case (name,path) if selected(name) =>
      name -> controlModels.resolve(path,currentModel.kind,() => job.cancelled)
    }

  private def prepareConditioning(req: GenerationRequest, job: Job, resources: Option[VulkanHashes.Run] = None,
      incantationsBase: Option[(Int, Double)] = None): VulkanInference.Conditioning =
    val previous = loadRequest.get
    req.params.vae.filter(_.nonEmpty).foreach { name =>
      val matches = if Set("Automatic", "None")(name) then previous.vaePath.isEmpty
        else previous.vaePath.exists(path => path == name || VulkanConditioning.stem(path) == name.replace('\\', '/').split('/').last)
      if !matches then invalid(s"Select VAE '$name' with set_vae before generation")
    }
    val controls = resolveControls(req,job)
    val (control, conditioning) = VulkanConditioning.prepare(req.copy(controlnetPaths = controls), previous, () => job.cancelled,
      incantationsBase)
    val names = req.embeddingPaths.map(VulkanConditioning.stem).toSet
    val extra = toolSession.embeddingPaths(currentModel).filterNot(p => names(VulkanConditioning.stem(p)))
    val embeddings = VulkanConditioning.activeEmbeddings(req.copy(embeddingPaths = req.embeddingPaths ++ extra))
    val baseOptions = VulkanConditioning.options(previous, device.id(), control, embeddings, req.params.hooks)
    val tiny = Option.when(req.previewMethod == "taesd")(sdfx.vulkan.tools.VulkanVae.tinyPath(currentModel,None)).flatten
    val opts = tiny.fold(baseOptions)(path => baseOptions.withTinyPreview(path.toString))
    if !activeOptions.contains(opts) then
      val controlFile = controls.find(_._2 == control).map { (name,normalized) =>
        val selectedPath = Path.of(req.controlnetPaths(name))
        val file = if Files.isDirectory(selectedPath) then selectedPath.resolve(Path.of(normalized).getFileName) else selectedPath
        s"controlnet:${file.getFileName}" -> file.toString
      }
      val paths = opts.embeddings().asScala.map(e => s"embed:${e.name()}" -> e.path()).toVector ++ controlFile
      val files = paths.map((k,v) => k -> new VulkanHashes.File(Path.of(v))).toMap
      backend.load(Path.of(previous.path), device.id(), opts)
      activeOptions = Some(opts)
      activeResourceFiles = files
    resources.foreach { run =>
      activeResourceFiles.foreach(run.record)
      req.params.loras.foreach { lora =>
        req.loraPaths.get(lora.name).foreach(path => run.record(s"lora:${lora.name}", new VulkanHashes.File(Path.of(path))))
      }
    }
    conditioning

  private def scoreImage(image:java.awt.image.BufferedImage,raw:Value,job:Job):sdfx.vulkan.tools.VulkanAestheticScore.Result =
    def score()=sdfx.vulkan.tools.VulkanAestheticScore(image,raw,backend,device.id(),()=>job.check())
    try score()
    catch case failure:VulkanInference.NativeFailure if failure.status==6 && loaded.nonEmpty && activeOptions.nonEmpty =>
      val model=currentModel
      val options=activeOptions.get
      backend.unload()
      activeOptions=None
      try score()
      finally
        backend.load(Path.of(model.path),device.id(),options)
        activeOptions=Some(options)

  private def generate(job: Job, gen: Generation): Value =
    val fooocus = gen.request.params.hooks.get(VulkanFooocusInpaint.id)
      .exists(!_.obj.get("enabled").contains(ujson.Bool(false)))
    if fooocus then VulkanClipIo.withScratch("sdv-fooocus-") { dir =>
      Using.resource(backend.temporaryDglModelFiles())(_ => generate(job, gen, Some(dir), None))
    }
    else gen.request.params.hooks.get(VulkanComposableLora.id).filter(VulkanComposableLora.plotRequested) match
      case None => generate(job, gen, None, None)
      case Some(config) =>
        if !backend.supportsComposableLoraTrace() then unsupported("This Vulkan runtime lacks Composable LoRA tracing")
        val trace = new VulkanLoraTrace.Run(config)
        Using.resource(VulkanInference.traceLoraPredictions(trace.record))(_ => generate(job, gen, None, Some(trace)))

  private def generate(job: Job, gen: Generation, fooocusDir: Option[Path], loraTrace: Option[VulkanLoraTrace.Run]): Value =
    val model = currentModel
    val requested = gen.request.params
    val sag = requested.hooks.get("sag").filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
    val pag = requested.hooks.get("pag").exists(v => !v.obj.get("enabled").contains(ujson.Bool(false)))
    val skipSag = sag.nonEmpty && (model.kind=="sdxl" || requested.isImg2Img || requested.controlnet.exists(_.enabled) || pag)
    val sagWarnings =
      if skipSag then Seq("SAG applies to SD1.x txt2img without ControlNet or PAG; ignored")
      else if sag.exists(_.obj.get("blur_sigma").exists(_.num != 2)) then Seq("The SAG reference uses a fixed Gaussian blur; blur sigma ignored")
      else Nil
    val skipAgent = requested.hooks.get(VulkanAgentAttention.id)
      .filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
      .exists(config => !VulkanAgentAttention.parse(config).applicable(model.kind, requested.hires.enabled))
    val skipLLLite = requested.hooks.get(VulkanLLLite.id)
      .filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
      .exists(config => !VulkanLLLite.parse(config).active)
    val skipLlul = requested.hooks.get(VulkanLlul.id)
      .filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
      .exists { config =>
        val llul = VulkanLlul.parse(config)
        !llul.canApply(requested.steps) &&
          !(requested.hires.enabled && llul.canApply(VulkanHiresGraph.fullSteps(requested)))
      }
    val skipCharacteristic = requested.hooks.get(VulkanCharacteristic.id)
      .filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
      .exists { raw =>
        val config = VulkanCharacteristic.parse(raw)
        val base = config.applicable(requested.steps, requested.cfgScale)
        val hires = requested.hires.enabled && config.applicable(requested.steps, requested.hires.cfgScale.getOrElse(requested.cfgScale))
        !base && !hires
      }
    val skipComposable = requested.hooks.get(VulkanComposableLora.id)
      .exists(config => !config.obj.get("enabled").contains(ujson.Bool(false)) && !VulkanComposableLora.needsGraph(gen.request))
    val effectiveHooks = requested.hooks -- Option.when(skipSag)("sag") -- Option.when(skipAgent)(VulkanAgentAttention.id) --
      Option.when(skipLLLite)(VulkanLLLite.id) -- Option.when(skipLlul)(VulkanLlul.id) --
      Option.when(skipCharacteristic)(VulkanCharacteristic.id) -- Option.when(skipComposable)(VulkanComposableLora.id)
    val req = if effectiveHooks.size == requested.hooks.size then gen.request
      else gen.request.copy(params = requested.copy(hooks = effectiveHooks))
    val p = req.params
    if isGeneration(job.kind) && ((job.kind == "img2img") != p.isImg2Img) then invalid("img2img requests require an init image and the img2img operation")
    if VulkanIncantations.seg(p.hooks).nonEmpty && !backend.supportsSeg() then
      unsupported("This Vulkan runtime lacks Incantations SEG")
    if VulkanIncantations.pag(p.hooks).nonEmpty && !backend.supportsIncantationsPag() then
      unsupported("This Vulkan runtime lacks Incantations PAG")
    if VulkanIncantations.scfg(p.hooks).nonEmpty && !backend.supportsScfg() then
      unsupported("This Vulkan runtime lacks Incantations S-CFG")
    val hires = VulkanImageInputs.hires(req)
    val separateHires = VulkanImageInputs.separateHires(req)
    val refinement = Img2ImgHiresMetadata.active(p.hooks)
    val dir = Files.createDirectories(Path.of(req.outputDir).toAbsolutePath.normalize())
    val start = System.nanoTime()
    val firstSeed = if p.seed >= 0 then p.seed else ThreadLocalRandom.current().nextLong(1L << 32)
    val tweak = p.hooks.get("hires_fix_tweaks").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).filter(_ => p.hires.enabled && !p.isImg2Img)
    val baseSubseed = if p.subseed >= 0 then p.subseed else ThreadLocalRandom.current().nextLong(1L << 32)
    val order: Seq[(Int,Option[VulkanHiresTweaks.Pass])] = tweak match
      case Some(config) => VulkanHiresTweaks.plan(config,p,firstSeed,baseSubseed).map((i,pass) => i -> Some(pass))
      case None => (0 until p.imagesPerJob).map(_ -> None)
    val keyword = sdfx.vulkan.tools.VulkanKeyword.plan(req, model, toolSession, firstSeed, () => job.cancelled)
    keyword.foreach(plan => job.commit = plan.commit)
    val resources = Option.when(p.hooks.get("pnginfo_injection").exists(!_.obj.get("enabled").contains(ujson.Bool(false))))(new VulkanHashes.Run)
    val composable = p.hooks.get(VulkanComposableLora.id)
      .exists(config => !config.obj.get("enabled").contains(ujson.Bool(false)))
    val conditioningRequest = if composable then req.copy(params = p.copy(loras = Nil)) else req
    val conditioning = Option.when(keyword.isEmpty)(prepareConditioning(conditioningRequest, job, resources))
    val difference = p.hooks.get("inpaint_difference").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).map { config =>
      val path = p.img2img.getOrElse(invalid("Inpaint Difference requires an img2img input")).initImage
      val image = sdfx.vulkan.tools.DiffusionImageInput.readRgb(Path.of(path), maxBytes = 200000000L)
      (image, VulkanTools.differenceMask(config, image, () => job.cancelled))
    }
    val cdTuner = VulkanCdTuner.config(p.hooks, Set("sdxl", "SDXL Inpaint")(model.kind))
    val hasCacheGraph = p.hooks.exists((key, config) =>
      VulkanGraph.cacheHooks(key) && !config.obj.get("enabled").contains(ujson.Bool(false)))
    val graph = Option.when(VulkanNegPip.applies(p,p.prompt,p.negativePrompt))(VulkanNegPip.id)
      .orElse(p.hooks.iterator.collectFirst { case (key, config)
        if key != VulkanLatentModifier.id && key != VulkanNegPip.id && VulkanGraph.hooks(key) &&
          !(hasCacheGraph && VulkanGraph.cacheGuidanceHooks(key)) &&
          !config.obj.get("enabled").contains(ujson.Bool(false)) && VulkanFusionPrompt.applies(key,p.prompt,p.negativePrompt) &&
          (key != VulkanGraph.aestheticHook || sdfx.vulkan.tools.VulkanAestheticTraining.settings(config).active) &&
          (key != VulkanCdTuner.id || cdTuner.nonEmpty) &&
          (key != VulkanUnetHooks.tiledDiffusion || VulkanUnetHooks.needsTiles(config,p.width,p.height)) => key })
      .orElse(VulkanLatentModifier.config(p).map(_ => VulkanLatentModifier.id))
      .orElse(Option.when(VulkanIncantations.attentionPasses(p))(VulkanIncantations.id))
      .orElse(Option.when(tweak.nonEmpty || separateHires && p.hires.upscaler.startsWith("Latent"))(VulkanGraph.latentHires))
    def cdFollowup(width: Int, height: Int): Option[VulkanCdTuner.Options] =
      graph.filter(_ == VulkanCdTuner.id).flatMap { _ =>
        val baseSteps = p.img2img.fold(p.steps)(i => ((p.steps * i.denoisingStrength).toInt + 1).min(p.steps))
        cdTuner.flatMap(_.followup(width.toDouble * height / (p.width.toDouble * p.height), baseSteps))
      }
    val graphRequest = if graph.nonEmpty || refinement.nonEmpty then
      val attachmentPaths = if graph.contains(VulkanFooocusInpaint.id) then req.controlnetPaths else
        p.hooks.get(VulkanLLLite.id).flatMap(_.obj.get("model")).flatMap(_.strOpt)
          .flatMap(name => req.controlnetPaths.get(name).map(name -> _)).toMap
      req.copy(controlnetPaths = attachmentPaths ++ resolveControls(req,job))
    else req
    val fooocusAssets = Option.when(graph.contains(VulkanFooocusInpaint.id)) {
      VulkanFooocusInpaint.prepareAssets(backend, model, graphRequest, difference.nonEmpty, fooocusDir.get, () => job.cancelled)
    }
    val temperature = p.hooks.get("gpu_temperature_guard").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).map { h =>
      if graph.nonEmpty then unsupported("GPU temperature guard in custom DGL graphs")
      val tools = sdfx.vulkan.tools.NativeTemperatureGuard
      val c = tools.parse(sdfx.vulkan.tools.NativeJson.parse(h.render()))
      new tools.Guard(c, () => tools.read(c), () => job.check(), (event,reading) =>
        if event != "reading" || c.printReadings then
          emit(Obj("id" -> job.id, "type" -> "event", "event" -> "log",
            "level" -> (if event == "timeout" then "warn" else "info"),
            "message" -> s"GPU temperature $event: $reading C (provider=${c.provider}, device=${c.device})")))
    }
    val auxiliary = scala.collection.mutable.ArrayBuffer.empty[GeneratedImage]
    var outputIndex = 0
    val imgs = order.zipWithIndex.flatMap { case ((i,pass),index) =>
      job.check()
      loraTrace.foreach(_.imageIndex = index)
      val seed = Math.addExact(firstSeed, i.toLong)
      job.frames.foreach(_.begin(index,seed))
      val inputs = p.img2img.filter(_ => graph.isEmpty).map(VulkanImageInputs.prepare(_, p.width, p.height, seed, difference, () => job.cancelled))
      def maskConditioning(strength: Double): Double =
        VulkanImageInputs.maskConditioning(p.hooks.get("conditioning_highres_fix"), strength)
      val imageOptions = new VulkanInference.ImageOptions(inputs.map(_.init).orNull,
        inputs.flatMap(_.mask).orNull, p.img2img.fold(0.75)(_.denoisingStrength), hires.filterNot(_ => separateHires).orNull, maskConditioning(p.img2img.fold(0.75)(_.denoisingStrength)))
      def progress(pass: Option[String]): VulkanInference.Progress = (stage, step, total) =>
        if step > 0 && total > 0 && job.sensorFailure.isEmpty && !job.cancelled then
          try temperature.foreach(_.step())
          catch case NonFatal(e) => job.sensorFailure = Some(e)
        if job.cancelled || job.sensorFailure.nonEmpty then backend.cancel()
        else emit(Obj("id" -> job.id, "type" -> "event", "event" -> "progress", "step" -> step, "total" -> total,
          "pass" -> pass.getOrElse(stage), "image" -> index, "images" -> order.size, "batch_size" -> 1))
      val planned = keyword.map(_.images(i))
      val activePrompt = planned.fold(p.prompt)(image => if composable then image.source else image.prompt)
      val activeConditioning = conditioning.getOrElse {
        val adapted = req.copy(params = p.copy(prompt = activePrompt, loras = if composable then Nil else planned.get.loras))
        prepareConditioning(adapted, job, resources)
      }
      var aestheticHash = Option.empty[String]
      def adapterStamp(path: Path) =
        val attributes = Files.readAttributes(path, classOf[BasicFileAttributes])
        (attributes.fileKey(), attributes.size(), attributes.lastModifiedTime())
      val aestheticAdapters = if graph.contains(VulkanGraph.aestheticHook) then
        val used = planned.fold(p.loras)(_.loras) ++
          req.hiresLoras.filter(_ => p.hires.enabled).toSeq.flatten ++
          req.img2imgHiresLoras.filter(_ => refinement.nonEmpty).toSeq.flatten
        used.filter(_.weight.toFloat != 0).flatMap(l => req.loraPaths.get(l.name)).distinct.map { raw =>
          val path = Path.of(raw)
          path -> adapterStamp(path)
        }
      else Nil
      def recordAesthetic(hash: String): Unit =
        if aestheticHash.exists(_ != hash) then invalid("Aesthetic Gradients embedding changed during the job")
        aestheticHash = Some(hash)
      def aestheticProgress(step: Int, loss: Float): Unit =
        emit(Obj("id" -> job.id, "type" -> "event", "event" -> "progress", "step" -> step,
          "total" -> sdfx.vulkan.tools.VulkanAestheticTraining.settings(p.hooks(VulkanGraph.aestheticHook)).steps,
          "pass" -> "aesthetic_gradients", "loss" -> loss.toDouble, "image" -> index,
          "images" -> order.size, "batch_size" -> 1))
      val rendered = graph match
        case Some(hook) => VulkanGraph.render(backend, device.id(), model, loadRequest.get, graphRequest, hook,
          activePrompt, planned.fold(p.loras)(_.loras), seed, gen.scheduler, () => job.cancelled, pass, difference,
          fooocus = fooocusAssets, aestheticDigest = recordAesthetic, aestheticProgress = aestheticProgress)
        case None =>
          val generated = backend.generate(activePrompt, p.negativePrompt, p.width, p.height, p.steps, p.cfgScale, seed,
            samplers(p.samplerName), schedulers(gen.scheduler), p.clipSkip, imageOptions, activeConditioning, progress(None))
          val highres = if !separateHires then generated else
            job.check()
            val h = p.hires
            val native = hires.get
            val w = native.width
            val ht = native.height
            val source = VulkanTools.upscaleImage(h.upscaler,req.upscalerPaths,generated,w,ht,modelUpscale)
            val sampler = h.samplerName.getOrElse(p.samplerName)
            val method = samplers.getOrElse(sampler,unsupported(s"hires sampler $sampler"))
            val scheduler = h.scheduler.getOrElse(p.scheduler)
            val schedule = if scheduler == "Automatic" then defaultSchedule(sampler) else scheduler
            val steps = if native.steps == 0 then p.steps else native.steps
            // sd.cpp img2img retains floor(scheduleSteps*strength)+1 steps.
            val scheduleSteps = math.max(steps,math.floor(steps/native.strength).toInt)
            if scheduleSteps > 10000 then invalid("Hires schedule exceeds 10000 steps")
            val strength = (steps - 0.5) / scheduleSteps
            val pass = p.copy(width = w, height = ht, steps = steps, samplerName = sampler, scheduler = schedule,
              cfgScale = h.cfgScale.getOrElse(p.cfgScale), prompt = Option(h.prompt).filter(_.nonEmpty).getOrElse(activePrompt),
              negativePrompt = Option(h.negativePrompt).filter(_.nonEmpty).getOrElse(p.negativePrompt),
              hires = h.copy(enabled = false), loras = req.hiresLoras.getOrElse(planned.fold(p.loras)(_.loras)))
            val mask = inputs.flatMap(_.mask).map(VulkanImageInputs.resize(_,w,ht,0)).orNull
            val condition = prepareConditioning(req.copy(params = pass),job,resources,Some(p.steps -> p.cfgScale))
            backend.generate(pass.prompt,pass.negativePrompt,w,ht,scheduleSteps,pass.cfgScale,seed,method,
              schedulers.getOrElse(schedule,unsupported(s"hires scheduler $schedule")),p.clipSkip,
              new VulkanInference.ImageOptions(source,mask,strength,null,maskConditioning(strength)),condition,progress(Some("hires")))
          Seq(inputs.fold(highres)(_.finish(highres)))
      rendered.map { first =>
        job.check()
        val initial = refinement.fold(first) { h =>
          val (w, ht) = Img2ImgHiresMetadata.targetSize(h, first.getWidth, first.getHeight)
          val resized = VulkanTools.upscaleImage(h("upscaler").str, req.upscalerPaths, first, w, ht, modelUpscale)
          val strength = h("denoise").num
          if strength == 0 then resized
          else
            job.check()
            def inherit(key: String, fallback: String) =
              Option(VulkanCdTuner.cleanPrompt(h(key).str)).filter(_.nonEmpty).getOrElse(fallback)
            val prompt = inherit("prompt", activePrompt)
            val negative = inherit("negative_prompt", p.negativePrompt)
            val sampler = if h("sampler").str == "Use same sampler" then p.samplerName else h("sampler").str
            val schedule = if h("scheduler").str == "Use same scheduler" then p.scheduler else h("scheduler").str
            val method = samplers.getOrElse(sampler, unsupported(s"refinement sampler $sampler"))
            val actualSchedule = if schedule == "Automatic" then defaultSchedule(sampler) else schedule
            val nativeSchedule = schedulers.getOrElse(actualSchedule, unsupported(s"refinement scheduler $actualSchedule"))
            val steps = h("steps").num.toInt
            val cfg = h("cfg").num
            val refinementParams = p.copy(width = w, height = ht,
              steps = if steps == 0 then p.steps else steps, cfgScale = if cfg == 0 then p.cfgScale else cfg,
              hires = p.hires.copy(enabled = false), loras = req.img2imgHiresLoras.getOrElse(planned.fold(p.loras)(_.loras)))
            val clip = graph.filter(_ == VulkanClipIo.id).flatMap { _ =>
              VulkanClipIo.refinementConfig(p.hooks(VulkanClipIo.id), prompt == activePrompt, negative == p.negativePrompt)
            }
            val scheduleHooks = p.hooks.filter { (key, value) =>
              (VulkanFusionPrompt.textHooks(key) || key == VulkanNegPip.id) && !value.obj.get("enabled").contains(ujson.Bool(false))
            }
            val refinementClock = refinementParams.copy(img2img = p.img2img.map(_.copy(denoisingStrength = strength)))
            val textGraph = Option.when(VulkanNegPip.applies(refinementClock,prompt,negative))(VulkanNegPip.id)
              .orElse(scheduleHooks.keysIterator.find(key => VulkanFusionPrompt.textHooks(key) && VulkanFusionPrompt.applies(key,prompt,negative)))
            if clip.nonEmpty && textGraph.nonEmpty then unsupported("Clip I/O and prompt scheduling in the same refinement")
            val unetHooks = p.hooks.filter { (key, value) =>
              VulkanUnetHooks.hooks(key) && !value.obj.get("enabled").contains(ujson.Bool(false)) &&
                (key != VulkanUnetHooks.tiledDiffusion || VulkanUnetHooks.needsTiles(value,w,ht))
            }
            if unetHooks.nonEmpty && (clip.nonEmpty || textGraph.nonEmpty) then
              unsupported("U-Net hooks and custom text conditioning in the same refinement")
            val modifier = VulkanLatentModifier.config(refinementParams).map(VulkanLatentModifier.id -> _)
            val cdRefinement = cdFollowup(w, ht)
            val attentionRefinement = VulkanIncantations.attentionAt(p.hooks,
              VulkanImageInputs.img2imgSteps(refinementParams.steps, strength))
            val otherRefinement = clip.map(value => VulkanClipIo.id -> Map(VulkanClipIo.id -> value))
              .orElse(textGraph.map(_ -> scheduleHooks))
              .orElse(unetHooks.keysIterator.nextOption().map(_ -> unetHooks))
              .orElse(modifier.map(entry => VulkanLatentModifier.id -> Map(entry)))
              .orElse(cdRefinement.map(_ => VulkanCdTuner.id -> Map(VulkanCdTuner.id -> p.hooks(VulkanCdTuner.id))))
              .orElse(graph.filter(VulkanGraph.imageAttentionHooks).map(key =>
                key -> p.hooks.filter((k, _) => VulkanGraph.imageAttentionHooks(k))))
              .orElse(graph.filter(VulkanGraph.cacheHooks).map(key =>
                key -> p.hooks.filter((k, _) => VulkanGraph.cacheHooks(k) || VulkanGraph.cacheGuidanceHooks(k))))
              .orElse(graph.filter(_ == VulkanGraph.aestheticHook).map { key =>
                val config = ujson.Obj.from(p.hooks(key).obj)
                aestheticHash.foreach(hash => config("embedding_sha256") = hash)
                key -> Map(key -> config)
              })
            if attentionRefinement && otherRefinement.nonEmpty then
              unsupported("Incantations with other sampling hooks in img2img Hires Fix")
            val refinementGraph = otherRefinement.orElse(Option.when(attentionRefinement) {
              val key = VulkanIncantations.id
              key -> Map(key -> p.hooks(key))
            })
              .map((key,hooks) => key -> (hooks ++ modifier))
            refinementGraph match
              case Some((hook, hooks)) =>
                val params = refinementParams.copy(prompt = prompt, negativePrompt = negative,
                  samplerName = sampler, scheduler = actualSchedule,
                  img2img = Some(p.img2img.getOrElse(Img2ImgParams("")).copy(mask = None, denoisingStrength = strength)),
                  hooks = hooks)
                VulkanGraph.render(backend, device.id(), model, loadRequest.get, graphRequest.copy(params = params),
                  hook, prompt, params.loras, seed, actualSchedule, () => job.cancelled,
                  refinementImage = Some(resized), cdRefinement = cdRefinement, aestheticDigest = recordAesthetic,
                  aestheticProgress = aestheticProgress).head
              case None =>
                val conditioning2 = prepareConditioning(req.copy(params = refinementParams.copy(prompt = prompt,
                  samplerName = sampler)), job, resources)
                backend.generate(prompt, negative, w, ht, refinementParams.steps, refinementParams.cfgScale,
                  seed, method, nativeSchedule, p.clipSkip,
                  new VulkanInference.ImageOptions(resized, null, strength, null), conditioning2, progress(Some("img2img_hires")))
        }
        aestheticAdapters.foreach { (path, stamp) =>
          if adapterStamp(path) != stamp then invalid(s"Aesthetic Gradients LoRA changed during the job: ${path.getFileName}")
        }
        val scaled = p.hooks.get("sonar").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).fold(initial){ h =>
          new VulkanSonar(h,p,seed).upscale(initial,req.upscalerPaths,modelUpscale)
        }
        val facePasses = Seq("adetailer","batch_face_swap").flatMap { key =>
          p.hooks.get(key).filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).map { cfg =>
            if key == "adetailer" then cfg else sdfx.vulkan.tools.VulkanADetailer.facePass(cfg,model.path,model.hash)
          }
        }
        lazy val faceConditioning =
          if VulkanIncantations.cfg(p.hooks).nonEmpty then
            activeConditioning.withSampleArgs(VulkanConditioning.sampleArgs(p.copy(hooks = p.hooks - VulkanIncantations.id)))
          else activeConditioning
        val detailed = facePasses.foldLeft(scaled) { (canvas,config) =>
          val models = Path.of(model.path).toAbsolutePath.getParent
          sdfx.vulkan.tools.VulkanADetailer(backend,device.id(),Option(models.getParent).getOrElse(models),canvas,config,p,activePrompt,seed,
            () => job.cancelled, (unit,count) => emit(Obj("id" -> job.id,"type" -> "event","event" -> "log",
              "level" -> "info","message" -> s"ADetailer unit ${unit+1} detected $count regions")),
            (pos,neg,w,h,steps,cfg,sampler,regionSeed,opts) => cdFollowup(w, h) match
              case None =>
                backend.generate(pos,neg,w,h,steps,cfg,regionSeed,samplers.getOrElse(sampler,unsupported(s"ADetailer sampler $sampler")),
                  schedulers(gen.scheduler),p.clipSkip,opts,faceConditioning,progress(Some("adetailer")))
              case Some(tuner) =>
                val params = p.copy(prompt = pos, negativePrompt = neg, width = w, height = h, steps = steps,
                  cfgScale = cfg, samplerName = sampler, hires = p.hires.copy(enabled = false),
                  img2img = Some(sdfx.core.model.Img2ImgParams("", denoisingStrength = opts.strength)),
                  hooks = Map(VulkanCdTuner.id -> p.hooks(VulkanCdTuner.id)))
                val prepared = VulkanImageInputs.Inputs(opts.init(), Option(opts.mask()), opts.init(), None, None)
                VulkanGraph.render(backend, device.id(), model, loadRequest.get, graphRequest.copy(params = params),
                  VulkanCdTuner.id, pos, planned.fold(p.loras)(_.loras), regionSeed, gen.scheduler, () => job.cancelled,
                  cdRefinement = Some(tuner), preparedInput = Some(prepared)).head)
        }

        val restored = p.hooks.get("face_restore").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).fold(detailed) { options =>
          sdfx.vulkan.tools.VulkanFaceRestore(detailed,options,device.name(),() => job.check())
        }
        val image = p.hooks.get("nsfw_checker").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).fold(restored){ cfg =>
          val censor=new sdfx.vulkan.tools.VulkanSafety(backend,device.id(),device.name(),()=>job.check())
          val screened=censor.filter(restored,cfg,false)
          for (item,suffix)<-Seq(screened.beforeCensor->"-before-censor",screened.mask->"-censor-mask"); im<-item do
            val f=Files.createTempFile(dir,"vulkan-nsfw-",".png")
            job.files+=f
            sdfx.core.png.ScrambledPng.write(im,f,Seq.empty,job.imagePassword)
            auxiliary+=GeneratedImage(f.toString,seed,width=im.getWidth,height=im.getHeight,outputSuffix=Some(suffix),auxiliaryOf=Some(outputIndex))
          screened.image
        }
        outputIndex+=1
        val score = p.hooks.get("aesthetic_score").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).map { config =>
          val result=scoreImage(image,config,job)
          sdfx.vulkan.tools.VulkanAestheticScore.metadata(result,config,p,activePrompt,seed,model.hash)
        }
        job.frames.foreach(_.finalImage(image))
        val file = Files.createTempFile(dir, "vulkan-", ".png")
        job.files += file
        sdfx.core.png.ScrambledPng.write(image, file, score.toSeq.flatMap(_.chunks), job.imagePassword)
        val hardware = Option.when(req.hardwareInfo.enabled)(HardwareInfo.Record(display(device),
          f"${(System.nanoTime() - start) / 1e9}%.2fs", Option(req.hardwareInfo.remark).filter(_.nonEmpty)))
        GeneratedImage(file.toString, seed, subseed = pass.fold(-1L)(_.baseSubseed), width = image.getWidth, height = image.getHeight, hardwareInfo = hardware,
          effectivePrompt = planned.map(image => if composable then image.source else image.replay),
          modelKeywordSourcePrompt = planned.map(_.source),
          hiresSeed = pass.map(_.seed), hiresSubseed = pass.map(_.subseed), hiresRepeat = pass.map(_.repeat),
          hiresSubseedStrength = pass.map(_.strength), hiresSeedResizeFromW = pass.map(_.width), hiresSeedResizeFromH = pass.map(_.height),
          infotextFields = Seq("Vulkan device" -> display(device)) ++ aestheticHash.toSeq.map("Aesthetic embedding SHA256" -> _) ++
            score.toSeq.flatMap(_.fields), pngChunks = score.toSeq.flatMap(_.chunks))
      }
    }
    val diagnostics = for
      trace <- loraTrace
      first <- imgs.headOption
    yield
      job.check()
      trace.finish(dir, first, job.imagePassword, path => { job.files += path; () })
    Json.toJs(GenerationResult(imgs ++ diagnostics.map(_._1), (System.nanoTime() - start) / 1e9, Some(model.name), Some(model.hash),
      auxiliaryImages = auxiliary.toSeq, intermediateArtifacts = job.frames.toSeq.flatMap(_.finish()),
      composableLoraTrace = diagnostics.map(_._2),
      systemInfo = Option.when(req.systemInfoFields.nonEmpty)(VulkanDiagnostics.metadata(status, req.systemInfoFields)),
      scheduler = Some(gen.scheduler), warnings = gen.warnings ++ sagWarnings ++ resources.toSeq.flatMap(_.warnings),
      resourceHashes = resources.map(r => r.hashes ++ Map("model" -> model.hash) ++ model.vaeHash.map("vae" -> _)).getOrElse(Map.empty)))
