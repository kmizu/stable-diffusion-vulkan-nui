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

  private def step(config: Value, field: String, default: Int): Int =
    val value = number(config, field, default)
    if !value.isValidInt then VulkanProtocol.invalid(s"Incantations: $field must be an integer")
    value.toInt

  sealed trait StepWindow:
    def start: Int
    def end: Int
    final def canApply(count: Int): Boolean =
      count > 0 && end >= 0 && start <= end && start <= math.max(0, count - 2)

  final case class Seg(blur: Double, start: Int, end: Int) extends StepWindow
  final case class Pag(scale: Float, start: Int, end: Int, sanf: Boolean) extends StepWindow
  final case class Scfg(scale: Float, start: Int, end: Int, r: Int, min: Float, max: Float, clamp: Float) extends StepWindow

  final case class T2iZero(start: Int, end: Int, tokens: Vector[Int], radius: Int,
      threshold: Double, strength: Double, alpha: Double, ema: Double) extends StepWindow:
    def suppresses(index: Int): Boolean = alpha > 0 && index >= start && (end <= 0 || index <= end)

  def t2i0(hooks: Map[String, Value]): Option[T2iZero] =
    configured(hooks).filter(flag(_, "t2i0_active")).map { config =>
      val raw = config.obj.get("t2i0_tokens").filterNot(_ == ujson.Null).fold("")(_.str)
      val tokens = if raw.isEmpty then Vector.empty else raw.split(",", -1).toVector.map(token =>
        token.trim.toIntOption.getOrElse(VulkanProtocol.invalid("T2I-Zero Tokens must be comma separated integers")))
      val radius = step(config, "t2i0_window_size", 2)
      if radius < 0 then VulkanProtocol.invalid("T2I-Zero window size must be nonnegative")
      val settings = Seq(number(config, "t2i0_correction_threshold", .5), number(config, "t2i0_correction_strength", 0),
        number(config, "t2i0_ctnms_alpha", .1), number(config, "t2i0_ema_factor", 0))
      if settings.exists(value => !value.toFloat.isFinite) then VulkanProtocol.invalid("T2I-Zero settings must fit in float32")
      T2iZero(step(config, "t2i0_step_start", 1), step(config, "t2i0_step_end", 25), tokens, radius,
        settings(0), settings(1), settings(2), settings(3))
    }

  private def pagAt(settings: Option[Pag], index: Int): Boolean =
    settings.exists { config =>
      val step = math.max(0, index - 1)
      step >= config.start && step <= config.end
    }

  def t2iClock(hooks: Map[String, Value], count: Int, offset: Int, perturbed: Boolean): Vector[Int] =
    val settings = pag(hooks)
    var next = offset
    Vector.tabulate(count) { index =>
      val result = next + (if perturbed then 1 else 0)
      next += (if pagAt(settings, index) then 2 else 1)
      result
    }

  def t2iOffset(hooks: Map[String, Value], count: Int): Int =
    val settings = pag(hooks)
    count + (0 until count).count(pagAt(settings, _))

  private def t2iAt(hooks: Map[String, Value], count: Int, offset: Int): Boolean = t2i0(hooks).exists { config =>
    config.strength != 0 && config.canApply(count) || t2iClock(hooks, count, offset, perturbed = false)
      .zipWithIndex.exists { (index, evaluation) =>
        config.suppresses(index) || pagAt(pag(hooks), evaluation) && config.suppresses(index + 1)
      }
  }

  def t2iArguments(hooks: Map[String, Value], count: Int, width: Int, height: Int,
      clip: String, branch: String, previous: Int = 0, retain: Boolean = false): String = t2i0(hooks).fold("") { config =>
    val clock = t2iClock(hooks, count, t2iOffset(hooks, previous), perturbed = branch == "perturbed")
    val strength = if branch == "pos" && config.canApply(count) then config.strength else 0.0
    val first = clock.find(config.suppresses)
    if strength == 0 && first.isEmpty then ""
    else
      val previousActive = t2iClock(hooks, previous, 0, perturbed = false).exists(config.suppresses)
      val reset = if previousActive then -1 else first.getOrElse(-1)
      val tokens = if config.tokens.isEmpty then "" else s", t2i0_tokens = ${config.tokens.mkString("[", ",", "]")}"
      val history = if config.ema <= 0 then ""
        else if retain then clock.findLast(config.suppresses).fold("")(last => s", t2i0_save = $last")
        else if previousActive && branch != "perturbed" then s", t2i0_previous = $branch" else ""
      s""", t2i0_prompt = $clip, t2i0_start = ${config.start}, t2i0_end = ${config.end}, t2i0_radius = ${config.radius}, t2i0_threshold = ${config.threshold}, t2i0_strength = $strength, t2i0_alpha = ${config.alpha}, t2i0_ema = ${config.ema}, t2i0_width = $width, t2i0_height = $height, t2i0_clock = ${clock.mkString("[", ",", "]")}, t2i0_reset = $reset, t2i0_branch = "$branch"$tokens$history"""
  }

  def seg(hooks: Map[String, Value]): Option[Seg] =
    configured(hooks).filter(flag(_, "seg_active")).flatMap { config =>
      val blur = number(config, "seg_blur_sigma", 11)
      Option.when(blur != 0)(Seg(blur.max(-64).min(11), step(config, "seg_start_step", 0), step(config, "seg_end_step", 150)))
    }

  def pag(hooks: Map[String, Value]): Option[Pag] =
    configured(hooks).filter(flag(_, "pag_active")).flatMap { config =>
      val scale = number(config, "pag_scale", 0).toFloat
      if !scale.isFinite then VulkanProtocol.invalid("Incantations PAG scale must fit in float32")
      Option.when(scale > 0)(Pag(scale, step(config, "pag_start_step", 0), step(config, "pag_end_step", 150),
        flag(config, "pag_sanf")))
    }

  def scfg(hooks: Map[String, Value]): Option[Scfg] =
    configured(hooks).filter(flag(_, "scfg_active")).flatMap { config =>
      def value(field: String, default: Double): Float =
        val result = number(config, field, default).toFloat
        if !result.isFinite then VulkanProtocol.invalid(s"Incantations S-CFG: $field must fit in float32")
        result
      val r = step(config, "scfg_r", 4)
      if r < 1 then VulkanProtocol.invalid("Incantations S-CFG R must be at least 1")
      val scale = value("scfg_scale", 1)
      Option.when(scale > 0)(Scfg(scale, step(config, "scfg_start_step", 0), step(config, "scfg_end_step", 150),
        r, value("scfg_rate_min", 0.8), value("scfg_rate_max", 3), value("scfg_rate_clamp", 0)))
    }

  def attentionAt(hooks: Map[String, Value], count: Int, offset: Int = 0): Boolean =
    seg(hooks).exists(_.canApply(count)) || pag(hooks).exists(_.canApply(count)) || scfg(hooks).exists(_.canApply(count)) ||
      t2iAt(hooks, count, offset)

  def attentionPasses(p: GenerationParams): Boolean =
    val base = p.img2img.fold(p.steps)(i => VulkanImageInputs.img2imgSteps(p.steps, i.denoisingStrength))
    val high = if p.hires.enabled then VulkanHiresGraph.refinementSteps(p) else 0
    attentionAt(p.hooks, base) || p.hires.enabled && attentionAt(p.hooks, high, t2iOffset(p.hooks, base))

  def segArguments(hooks: Map[String, Value], width: Int, height: Int, count: Int): String =
    seg(hooks).filter(_.canApply(count)).fold("") { config =>
      s", seg_blur = ${config.blur}, seg_start = ${config.start}, seg_end = ${config.end}, seg_width = $width, seg_height = $height"
    }

  def scfgArguments(hooks: Map[String, Value], width: Int, height: Int, count: Int): String =
    scfg(hooks).filter(_.canApply(count)).fold("") { c =>
      s", scfg_scale = ${c.scale}, scfg_r = ${c.r}, scfg_start = ${c.start}, scfg_end = ${c.end}, scfg_min = ${c.min}, scfg_max = ${c.max}, scfg_clamp = ${c.clamp}, scfg_width = $width, scfg_height = $height"
    }

  def pagGraph(hooks: Map[String, Value], count: Int, unet: String, condition: String, args: String,
      positive: String, base: String, output: String, t2iArgs: String = ""): Vector[String] =
    pag(hooks).filter(_.canApply(count)).fold(Vector(s"$output = $base")) { config =>
      val branch = s"${output}_perturbed"
      Vector(s"$branch = predictor($unet, $condition$args, pag_regions = 512$t2iArgs)",
        s"$output = incantations_pag($base, $positive, $branch, scale = ${config.scale}, start_step = ${config.start}, end_step = ${config.end}, sanf = ${config.sanf})")
    }

  def validate(p: GenerationParams): Unit = configured(p.hooks).foreach { config =>
    t2i0(p.hooks).filter(t => t.strength != 0 || t.alpha > 0).foreach { t =>
      validateSampler(p.samplerName, "T2I-Zero")
      if p.hires.enabled then p.hires.samplerName.foreach(validateSampler(_, "T2I-Zero"))
      Img2ImgHiresMetadata.active(p.hooks).filter(_("denoise").num > 0).map(_("sampler").str)
        .filterNot(_ == "Use same sampler").foreach(validateSampler(_, "T2I-Zero"))
      if t.alpha > 0 && t.ema > 0 && pag(p.hooks).nonEmpty then
        VulkanProtocol.unsupported("T2I-Zero CTNMS EMA with Incantations PAG")
    }
    val attention = Option.when(seg(p.hooks).nonEmpty)("SEG").orElse(Option.when(pag(p.hooks).nonEmpty)("PAG"))
      .orElse(Option.when(scfg(p.hooks).nonEmpty)("S-CFG"))
    attention.foreach { feature =>
      val refinement = Img2ImgHiresMetadata.active(p.hooks).filter(_("denoise").num > 0)
      if p.cfgScale <= 1 || p.hires.enabled && p.hires.cfgScale.exists(_ <= 1) ||
          refinement.exists(pass => pass("cfg").num > 0 && pass("cfg").num <= 1) then
        VulkanProtocol.unsupported(s"Incantations $feature requires CFG scale greater than 1")
      validateSampler(p.samplerName, feature)
      if p.hires.enabled then p.hires.samplerName.foreach(validateSampler(_, feature))
      refinement.map(_("sampler").str).filterNot(_ == "Use same sampler").foreach(validateSampler(_, feature))
    }
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
