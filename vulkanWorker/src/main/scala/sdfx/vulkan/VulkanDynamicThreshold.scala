package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

/** Computes Dynamic Thresholding on denoised predictions before the Euler update. */
private[vulkan] final class VulkanDynamicThreshold(config: Value, p: GenerationParams):
  private def number(name: String, default: Double): Double =
    val value = config.obj.get(name).fold(default)(_.num)
    if !value.isFinite then VulkanProtocol.invalid(s"Invalid Dynamic Thresholding $name")
    value

  private def choice(name: String, default: String, choices: Set[String]): String =
    val value = config.obj.get(name).fold(default)(_.str)
    if !choices(value) then VulkanProtocol.invalid(s"Invalid Dynamic Thresholding $name")
    value

  private val modes = Set("Constant", "Linear Down", "Half Cosine Down", "Cosine Down",
    "Linear Up", "Half Cosine Up", "Cosine Up", "Power Up", "Power Down",
    "Linear Repeating", "Cosine Repeating", "Sawtooth")
  private val mimic = number("mimic_scale", 7)
  private val mimicMin = number("mimic_scale_min", 0)
  private val cfgMin = number("cfg_scale_min", 0)
  private val mimicMode = choice("mimic_mode", "Constant", modes)
  private val cfgMode = choice("cfg_mode", "Constant", modes)
  private val percentile = number("threshold_percentile", 1)
  private val schedule = number("sched_val", 1)
  private val phi = number("interpolate_phi", 1)
  private val separate = config.obj.get("separate_feature_channels").forall(_.bool)
  private val fromZero = choice("scaling_startpoint", "MEAN", Set("MEAN", "ZERO")) == "ZERO"
  private val useStd = choice("variability_measure", "AD", Set("AD", "STD")) == "STD"
  if percentile < 0 || percentile > 1 || phi < 0 || phi > 1 || schedule < 0 then
    VulkanProtocol.invalid("Invalid Dynamic Thresholding percentile, interpolation or schedule")

  def step(lines: scala.collection.mutable.Builder[String, Vector[String]], n: Int, input: String,
      output: String, pos: String, neg: String): Unit =
    def put(label: String, expression: String): String =
      val name = s"dynthresh${n}_$label"
      lines += s"$name = $expression"
      name
    def sub(a: String, b: String) = s"tensor::add($a,tensor::mul($b,-1))"
    def crop(value: String, channel: Int) = s"tensor::crop($value, axes = [2], start = [$channel], size = [1])"
    def concat(parts: Seq[String]) = s"tensor::concat(${parts.mkString("[", ",", "]")}, axis = 2)"
    def stat(value: String, method: String, rank: String = "") =
      s"""tensor::stat_f32($value, mode = "$method"$rank)"""
    def kth(value: String, rank: Int) = stat(value, "kth", s", rank = $rank")
    def abs(value: String) = s"tensor::maximum($value,tensor::mul($value,-1))"
    val sigma = s"sigmas[${n-1}]"
    def scale(target: Double, minimum: Double, mode: String) =
      if mode == "Constant" then target.toString
      else s"""dynthresh_scale($pos,$sigma,$target,$minimum,"$mode",$schedule)"""
    val range = put("range", s"""tensor::vector([$sigma,0], dtype = "f32")""")
    def denoise(label: String, branch: String) =
      put(label, s"""sample($input, $range, predict = $branch, method = "euler", rng = rng)""")
    val cond = denoise("cond", pos)
    val uncond = denoise("uncond", neg)
    val delta = put("delta", sub(cond, uncond))
    val mimicTarget = put("mimic", s"tensor::add($uncond,tensor::mul($delta,${scale(mimic, mimicMin, mimicMode)}))")
    val cfgTarget = put("cfg", s"tensor::add($uncond,tensor::mul($delta,${scale(p.cfgScale, cfgMin, cfgMode)}))")
    val channels = 0 until 4
    def center(label: String, value: String): (Seq[String], Seq[String]) =
      val pairs = channels.map { channel =>
        val plane = put(s"${label}_$channel", crop(value, channel))
        val mean = put(s"${label}_mean_$channel", stat(plane, "mean"))
        put(s"${label}_center_$channel", s"tensor::add($plane,0 - $mean)") -> mean
      }
      pairs.unzip
    val (mimicCentered, _) = center("m", mimicTarget)
    val (cfgCentered, cfgMeans) = center("c", cfgTarget)
    val planeSize = (p.width / 8) * (p.height / 8)
    val groupSize = planeSize * (if separate then 1 else 4)
    def references(label: String, group: Int, m: String, c: String): (String, String) =
      if useStd then
        put(s"${label}_m_$group", stat(m, "std")) -> put(s"${label}_c_$group", stat(c, "std"))
      else
        val threshold = percentile * (groupSize - 1)
        val low = threshold.toInt + 1
        val high = math.min(groupSize, low + 1)
        val weight = threshold - threshold.toInt
        val magnitude = put(s"${label}_mag_$group", abs(c))
        val lower = put(s"${label}_low_$group", kth(magnitude, low))
        val quantile = if weight == 0 then lower else
          put(s"${label}_q_$group", s"$lower * ${1 - weight} + ${kth(magnitude, high)} * $weight")
        put(s"${label}_m_$group", kth(abs(m), groupSize)) -> quantile
    val refs = if separate then channels.map(i => references("ref", i, mimicCentered(i), cfgCentered(i)))
      else Seq(references("ref", 0, concat(mimicCentered), concat(cfgCentered)))
    val corrected = channels.map { channel =>
      val (m, c) = refs(if separate then channel else 0)
      val centered = cfgCentered(channel)
      def largest(label: String, a: String, b: String) =
        put(label, kth(s"""tensor::vector([$a,$b], dtype = "f32")""", 2))
      val reference = if fromZero || useStd then c else largest(s"limit_$channel", m, c)
      val safe = largest(s"safe_$channel", reference, "0.000000000001")
      val tensor = if fromZero then crop(cfgTarget, channel)
        else if useStd then centered
        else s"tensor::clamp($centered, min = 0 - $reference, max = $reference)"
      val scaled = s"tensor::mul($tensor,$m / $safe)"
      if fromZero then scaled else s"tensor::add($scaled,${cfgMeans(channel)})"
    }
    val guided = put("result", s"tensor::lerp($cfgTarget,${concat(corrected)},$phi)")
    lines += s"$output = tensor::lerp($guided,$input,sigmas[$n] / $sigma)"
