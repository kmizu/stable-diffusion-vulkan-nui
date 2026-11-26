package sdfx.vulkan

import java.util.Locale
import sdfx.core.infotext.Img2ImgHiresMetadata
import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] object VulkanIncantations:
  val id = "incantations"
  private val sigmaMin = 0.002
  private val sigmaMax = 80.0
  private val powers = Seq("0.01", "0.1", "1.0", "2.0", "4.0")
  private val clamps = Seq("4.0", "2.0", "1.0")
  private val names = Set("Constant", "Linear", "Inverse-Linear", "Cosine", "Sine", "Interval") ++
    powers.map(p => s"PCS (s=$p)") ++ clamps.flatMap(c => Seq(s"Clamp-Linear (c=$c)", s"Clamp-Cosine (c=$c)"))

  private def flag(config: Value, field: String): Boolean =
    config.obj.get(field).exists {
      case ujson.Bool(value) => value
      case ujson.Num(value) => value != 0
      case ujson.Str(value) => Set("enable", "enabled", "true", "on", "yes", "1")(value.trim.toLowerCase(Locale.ROOT))
      case _ => false
    }

  private def number(config: Value, field: String, default: Double): Double =
    val result = config.obj.get(field).filterNot(_ == ujson.Null).fold(default)(value =>
      value.numOpt.orElse(value.strOpt.flatMap(_.toDoubleOption))
        .getOrElse(VulkanProtocol.invalid(s"Incantations: invalid $field")))
    if !result.isFinite then VulkanProtocol.invalid(s"Incantations: $field must be finite")
    result

  def configured(hooks: Map[String, Value]): Option[Value] =
    hooks.get(id).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))

  def cfg(hooks: Map[String, Value]): Option[Value] =
    configured(hooks).filter(flag(_, "cfg_interval_enable"))

  final case class Seg(blur: Double, start: Int, end: Int):
    def canApply(count: Int): Boolean =
      count > 0 && end >= 0 && start <= end && start <= math.max(0, count - 2)

  def seg(hooks: Map[String, Value]): Option[Seg] =
    configured(hooks).filter(flag(_, "seg_active")).flatMap { config =>
      def step(field: String, default: Int): Int =
        val value = number(config, field, default)
        if !value.isValidInt then VulkanProtocol.invalid(s"Incantations: $field must be an integer")
        value.toInt
      val blur = number(config, "seg_blur_sigma", 11)
      Option.when(blur != 0)(Seg(blur.max(-64).min(11), step("seg_start_step", 0), step("seg_end_step", 150)))
    }

  def segPasses(p: GenerationParams): Boolean = seg(p.hooks).exists { config =>
    val base = p.img2img.fold(p.steps)(i => VulkanImageInputs.img2imgSteps(p.steps, i.denoisingStrength))
    config.canApply(base) || (p.hires.enabled && config.canApply(VulkanHiresGraph.refinementSteps(p)))
  }

  def segArguments(hooks: Map[String, Value], width: Int, height: Int, count: Int): String =
    seg(hooks).filter(_.canApply(count)).fold("") { config =>
      s", seg_blur = ${config.blur}, seg_start = ${config.start}, seg_end = ${config.end}, seg_width = $width, seg_height = $height"
    }

  def validate(p: GenerationParams): Unit = configured(p.hooks).foreach { config =>
    val unsupported = Seq("scfg_active" -> "S-CFG",
      "pag_active" -> "PAG", "t2i0_active" -> "Multi-Concept T2I-Zero").collect {
      case (field, name) if flag(config, field) => name
    }
    if unsupported.nonEmpty then VulkanProtocol.unsupported(s"Incantations ${unsupported.mkString(", ")}")
    if seg(p.hooks).nonEmpty then
      val refinement = Img2ImgHiresMetadata.active(p.hooks).filter(_("denoise").num > 0)
      if p.cfgScale <= 1 || p.hires.enabled && p.hires.cfgScale.exists(_ <= 1) ||
          refinement.exists(pass => pass("cfg").num > 0 && pass("cfg").num <= 1) then
        VulkanProtocol.unsupported("Incantations SEG requires CFG scale greater than 1")
      validateSampler(p.samplerName, "SEG")
      if p.hires.enabled then p.hires.samplerName.foreach(validateSampler(_, "SEG"))
      refinement.map(_("sampler").str).filterNot(_ == "Use same sampler").foreach(validateSampler(_, "SEG"))
    if flag(config, "cfg_interval_enable") then
      val conflicts = Set("cfg_schedule", "apg", "mahiro", "latent_modifier", "dynamic_thresholding",
        "rescale_cfg", "cd_tuner", "detail_daemon", "cads", "semantic_guidance", "multidiffusion",
        "pag", "sag", "reference", "neutral_prompt", "composable_lora")
      val present = conflicts.toSeq.sorted.filter(key => p.hooks.get(key)
        .exists(!_.obj.get("enabled").contains(ujson.Bool(false))))
      if present.nonEmpty then
        VulkanProtocol.unsupported(s"Incantations CFG Scheduler with ${present.mkString(", ")}")
      if p.cfgScale <= 1 || p.hires.enabled && p.hires.cfgScale.exists(_ <= 1) then
        VulkanProtocol.unsupported("Incantations CFG Scheduler requires CFG scale greater than 1")
      validateSampler(p.samplerName)
      if p.hires.enabled then p.hires.samplerName.foreach(validateSampler(_))
  }

  def validateSampler(sampler: String, feature: String = "CFG Scheduler"): Unit =
    if sampler != "Euler" && sampler != "Euler a" then
      VulkanProtocol.unsupported(s"Incantations $feature with sampler $sampler")

  def graphArguments(p: GenerationParams, count: Int): String =
    cfg(p.hooks).fold("")(config =>
      ", schedule = " + schedule(config, p.steps, p.cfgScale, count).mkString("[", ",", "]"))

  private def level(index: Int, steps: Int): Double =
    if index == 0 then sigmaMax
    else if index >= steps then 0.0
    else
      val hi = math.pow(sigmaMax, 1.0 / 3)
      val lo = math.pow(sigmaMin, 1.0 / 3)
      math.pow(hi + index.toDouble / (steps - 1) * (lo - hi), 3)

  private def nearest(sigma: Double, steps: Int): Double =
    if sigma <= sigmaMin then 0.0
    else if sigma >= sigmaMax then sigmaMax
    else
      var low = 0
      var high = steps - 1
      while low <= high do
        val mid = (low + high) / 2
        val value = level(mid, steps)
        if math.abs(value - sigma) < 1e-6 then return value
        if value < sigma then high = mid - 1
        else low = mid + 1
      val a = level(low, steps)
      val b = level(high, steps)
      if math.abs(a - sigma) < math.abs(b - sigma) then a else b

  def schedule(config: Value, steps: Int, baseCfg: Double, count: Int): Vector[Float] =
    val name = config.obj.get("cfg_interval_schedule").filterNot(_ == ujson.Null)
      .fold("Constant")(_.str)
    if !names(name) then VulkanProtocol.invalid(s"Incantations: unknown CFG schedule '$name'")
    val a = nearest(number(config, "cfg_interval_low", 0), steps)
    val b = nearest(number(config, "cfg_interval_high", 100), steps)
    val min = math.min(a, b)
    val max = math.max(a, b)
    Vector.tabulate(count) { index =>
      val step = math.max(0, index - 1)
      val progress = step.toDouble / steps
      val sigma = level(step, steps)
      val value = name match
        case "Constant" => baseCfg
        case _ if sigma < min || sigma > max => 1.0
        case "Linear" => baseCfg * 2 * (1 - progress)
        case "Inverse-Linear" => baseCfg * 2 * progress
        case "Cosine" => baseCfg * (1 + math.cos(math.Pi * progress))
        case "Sine" => baseCfg * (math.sin(math.Pi * progress - math.Pi / 2) + 1)
        case "Interval" => if step >= 1 && step <= 5 then baseCfg else 1.0
        case n if n.startsWith("PCS (s=") =>
          val power = n.stripPrefix("PCS (s=").stripSuffix(")").toDouble
          baseCfg * ((1 - math.cos(math.Pi * math.pow(1 - progress, power))) / 2)
        case n if n.startsWith("Clamp-Linear (c=") =>
          val floor = n.stripPrefix("Clamp-Linear (c=").stripSuffix(")").toDouble
          math.max(floor, baseCfg * 2 * (1 - progress))
        case n =>
          val floor = n.stripPrefix("Clamp-Cosine (c=").stripSuffix(")").toDouble
          math.max(floor, baseCfg * (1 + math.cos(math.Pi * progress)))
      if !value.toFloat.isFinite then VulkanProtocol.invalid(s"Incantations: CFG schedule $name is non-finite at step $step")
      value.toFloat
    }
