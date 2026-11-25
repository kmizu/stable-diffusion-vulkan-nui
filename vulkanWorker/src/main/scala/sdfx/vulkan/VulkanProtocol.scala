package sdfx.vulkan

import java.nio.file.{Files, Path}
import java.util.Locale
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import sdfx.core.protocol.*
import ujson.{Arr, Obj, Value}

private[vulkan] object VulkanProtocol:
  private val samplerDefinitions = Vector(
    ("DPM++ 2M", "dpm++2m", "Karras"), ("Euler", "euler", "Uniform"), ("Euler a", "euler_a", "Uniform"),
    ("DPM++ 2M SDE", "dpm++2m_sde", "Exponential"), ("DPM++ 2S a", "dpm++2s_a", "Karras"),
    ("LMS", "lms", "Uniform"), ("Heun", "heun", "Uniform"), ("DPM2", "dpm2", "Karras"), ("LCM", "lcm", "Uniform"))
  val samplers = samplerDefinitions.map((name, method, _) => name -> method).toMap
  private val schedulerDefinitions = Vector("Karras" -> "karras", "Uniform" -> "discrete",
    "Exponential" -> "exponential", "SGM Uniform" -> "sgm_uniform", "Align Your Steps" -> "ays",
    "Simple" -> "simple", "Normal" -> "discrete", "Beta" -> "beta")
  val schedulers = schedulerDefinitions.toMap
  val samplerList = SamplerList(samplerDefinitions.map((name, _, schedule) => SamplerInfo(name, true, schedule)),
    "Automatic" +: schedulerDefinitions.map(_._1), upscalers = (VulkanImageInputs.upscalers.keys.toSeq ++ Seq("Nearest", "Lanczos")).distinct.sorted,
    preprocessors = sdfx.vulkan.tools.VulkanTools.preprocessors)
  def defaultSchedule(sampler: String): String = samplerList.samplers.find(_.name == sampler).get.defaultScheduler
  def unsupported(what: String): Nothing = throw EngineError("unsupported", s"Vulkan worker does not support $what")
  def invalid(what: String): Nothing = throw EngineError("invalid_params", what)

  def decode[A](body: => A): A =
    try body
    catch case NonFatal(e) => invalid(Option(e.getMessage).getOrElse("Invalid request parameters"))

  def checkFields(raw: Value, known: Value): Unit =
    (raw.obj.keySet -- known.obj.keySet).headOption.foreach(k => unsupported(k))

  case class Generation(request: GenerationRequest, scheduler: String, warnings: Seq[String] = Nil)
  def generation(raw: Value): Generation =
    val req = decode(Json.fromJs[GenerationRequest](raw))
    val p = req.params
    val wire = Json.toJs(req)
    checkFields(raw, wire)
    checkFields(raw("params"), wire("params"))
    p.validate.headOption.foreach(invalid)
    if req.outputDir.isBlank then invalid("output_dir is required")
    p.hooks.foreach { (key, config) =>
      if !VulkanGraph.hooks(key) && !Set("nsfw_checker","aesthetic_score", "embedding_merge", "save_intermediate", "steps_animation", "gpu_temperature_guard", "conditioning_highres_fix", "face_restore", "adetailer", "batch_face_swap", "apg", "cfg_schedule", "tiling", "neveroom", "model_sampling", "model_keyword", "pnginfo_injection", "inpaint_difference", "img2img_hires_fix","hires_fix_tweaks")(key) then unsupported(s"engine hook $key")
      if config.objOpt.isEmpty then invalid(s"Invalid configuration for $key")
      if key == "pnginfo_injection" && config.obj.keys.exists(_ != "enabled") then invalid("Invalid PNG resource hash configuration")
      config.obj.get("enabled").foreach(value => decode(value.bool))
    }
    if p.subseedStrength != 0 && !(p.hires.enabled && !p.isImg2Img && p.hooks.get("hires_fix_tweaks").exists(!_.obj.get("enabled").contains(ujson.Bool(false)))) then unsupported("variation strength")
    if p.eta.exists(v => !v.isFinite || v < 0) then invalid("eta must be finite and nonnegative")
    if !samplers.contains(p.samplerName) then unsupported(s"sampler ${p.samplerName}")
    val schedule = if p.scheduler == "Automatic" then defaultSchedule(p.samplerName) else p.scheduler
    if !schedulers.contains(schedule) then unsupported(s"scheduler ${p.scheduler}")
    val sized = p.withEngineSize
    if sized.width.toLong * sized.height > 64L * 1024 * 1024 || sized.imagesPerJob > 64 then invalid("Generation exceeds pixel or image count limit")
    val skip = Vector(
      ("soft_inpainting",!p.isInpaint,"Soft Inpainting requires an inpaint mask. The regular generation pass was used."),
      ("img2imgalt",!p.isImg2Img || p.isInpaint,"img2img Alternative requires plain img2img. The regular generation pass was used."),
      ("hires_fix_tweaks",!p.hires.enabled || p.isImg2Img,"Hires Fix Tweaks applies to txt2img with Hires enabled.")
    ).filter((key,unused,_) => unused && p.hooks.get(key).exists(!_.obj.get("enabled").contains(ujson.Bool(false))))
    Generation(req.copy(params = sized.copy(hooks = sized.hooks -- skip.map(_._1))), schedule, skip.map(_._3))

  def display(d: VulkanInference.Device): String = s"${d.id()}: ${d.name()}"
  def select(devices: Seq[VulkanInference.Device], selected: Option[String]): VulkanInference.Device =
    val available = devices.map(display).mkString(", ")
    selected.filter(_.trim.nonEmpty) match
      case None => devices.headOption.getOrElse(throw IllegalStateException("No compatible Vulkan adapter found"))
      case Some(s) =>
        val matches = devices.filter(d => Seq(d.id(), d.name(), display(d)).exists(_.equalsIgnoreCase(s.trim)))
        if matches.size != 1 then invalid(s"Unknown or ambiguous Vulkan device '$s'. Available adapters: $available")
        matches.head

  def models(params: Value): Obj =
    val dirs = decode(params.obj.get("dirs").toSeq.flatMap(_.arr.map(_.str)).filter(_.nonEmpty))
    val entries = dirs.flatMap { s =>
      val base = Path.of(s).toAbsolutePath.normalize()
      if !Files.isDirectory(base) then Nil
      else Using.resource(Files.walk(base)) { walk =>
        walk.iterator().asScala.filter(p => Files.isRegularFile(p) && checkpoint(p)).map { p =>
          val name = base.relativize(p).toString.replace('\\', '/').dropRight(".safetensors".length)
          ModelEntry(name, p.toString, "unknown", Files.size(p).toDouble / (1024 * 1024))
        }.toVector
      }
    }
    Obj("models" -> Json.toJs(entries.distinctBy(_.path).sortBy(_.name.toLowerCase(Locale.ROOT))))
  def checkpoint(path: Path): Boolean = path.getFileName.toString.toLowerCase(Locale.ROOT).endsWith(".safetensors")
