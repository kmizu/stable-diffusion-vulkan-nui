package sdfx.vulkan

import java.nio.file.{Files, Path}
import java.util.Locale
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import sdfx.core.protocol.*
import ujson.{Arr, Obj, Value}

private[vulkan] object VulkanProtocol:
  val samplers = Map("DPM++ 2M" -> "dpm++2m", "Euler" -> "euler", "Euler a" -> "euler_a")
  val schedulers = Map("Karras" -> "karras", "Uniform" -> "discrete")
  def defaultSchedule(sampler: String): String = if sampler == "DPM++ 2M" then "Karras" else "Uniform"
  val samplerList = SamplerList(Seq("DPM++ 2M", "Euler", "Euler a").map(s => SamplerInfo(s, true, defaultSchedule(s))),
    Seq("Automatic", "Karras", "Uniform"))
  def unsupported(what: String): Nothing = throw EngineError("unsupported", s"Vulkan worker does not support $what")
  def invalid(what: String): Nothing = throw EngineError("invalid_params", what)

  def decode[A](body: => A): A =
    try body
    catch case NonFatal(e) => invalid(Option(e.getMessage).getOrElse("Invalid request parameters"))

  def checkFields(raw: Value, known: Value): Unit =
    (raw.obj.keySet -- known.obj.keySet).headOption.foreach(k => unsupported(k))

  case class Generation(request: GenerationRequest, scheduler: String)
  def generation(raw: Value): Generation =
    val req = decode(Json.fromJs[GenerationRequest](raw))
    val p = req.params
    val wire = Json.toJs(req)
    checkFields(raw, wire)
    checkFields(raw("params"), wire("params"))
    p.validate.headOption.foreach(invalid)
    if req.outputDir.isBlank then invalid("output_dir is required")
    if p.isImg2Img then unsupported("img2img")
    if p.hires.enabled then unsupported("hires fix")
    if p.loras.nonEmpty || req.loraPaths.nonEmpty || req.manualLoras.exists(_.nonEmpty) ||
        req.hiresLoras.exists(_.nonEmpty) || req.img2imgHiresLoras.exists(_.nonEmpty) then unsupported("LoRA")
    if p.controlnet.exists(_.enabled) || req.controlnetPaths.nonEmpty then unsupported("ControlNet")
    if req.embeddingPaths.nonEmpty then unsupported("textual inversion embeddings")
    if p.hooks.nonEmpty then unsupported(s"engine hooks: ${p.hooks.keys.toSeq.sorted.mkString(", ")}")
    if p.vae.exists(s => !Set("", "None", "Automatic")(s)) then unsupported("external VAE")
    if p.subseedStrength != 0 then unsupported("variation strength")
    if p.eta.nonEmpty then unsupported("eta")
    if req.systemInfoFields.nonEmpty then unsupported("system info fields")
    if !samplers.contains(p.samplerName) then unsupported(s"sampler ${p.samplerName}")
    val schedule = if p.scheduler == "Automatic" then defaultSchedule(p.samplerName) else p.scheduler
    if !schedulers.contains(schedule) then unsupported(s"scheduler ${p.scheduler}")
    val sized = p.withEngineSize
    if sized.width.toLong * sized.height > 64L * 1024 * 1024 || sized.imagesPerJob > 64 then invalid("Generation exceeds pixel or image count limit")
    Generation(req.copy(params = sized), schedule)

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
