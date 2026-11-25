package sdfx.vulkan

import ujson.Value

/** Adjust the model input scale and training timestep while retaining the solver's original sigma. */
private[vulkan] object VulkanDetailDaemon:
  val functions = """fn detail_bound(bounds, index, pred, target) {
    mid = (bounds.lo + bounds.hi) / 2.0
    sigma = t_to_sigma(pred, mid)
    side = tensor::to_list(tensor::threshold(tensor::vector([target - sigma], dtype = "f32"), threshold = 0), element = "number")[0]
    return {lo = bounds.lo + side * (mid - bounds.lo), hi = mid + side * (bounds.hi - mid)}
  }
  fn detail_transform(step, args) {
    return {timestep = args.time, input_scale = args.scale}
  }
  """

private[vulkan] final class VulkanDetailDaemon(config: Value, steps: Int):
  private def number(k: String, default: Double, lo: Double, hi: Double): Double =
    val v = config.obj.get(k).fold(default)(_.num)
    require(v.isFinite && v >= lo && v <= hi,s"Invalid Detail Daemon $k")
    v
  private val mode = config.obj.get("mode").fold("both")(_.str.toLowerCase(java.util.Locale.ROOT))
  require(Set("both","cond","uncond")(mode),"Invalid Detail Daemon mode")
  private val end = number("end",0.8,0,1)
  private val start = math.min(end,number("start",0.2,0,1))
  private val bias = number("bias",0.5,0,1)
  private val amount = number("amount",0.1,-1,1)
  private val exponent = number("exponent",1,0,10)
  private val startOffset = number("start_offset",0,-1,1)
  private val endOffset = number("end_offset",0,-1,1)
  private val fade = number("fade",0,0,1)
  private val smooth = config.obj.get("smooth").fold(true)(_.bool)
  private def index(at: Double): Int = math.rint(at*(steps-1)).toInt
  private val first = index(start)
  private val peak = index(start+bias*(end-start))
  private val last = index(end)
  private def ramp(x: Double): Double = math.pow(if smooth then 0.5*(1-math.cos(math.Pi*x)) else x,exponent)
  private def multiplier(i: Int): Double =
    val value = if i < first then startOffset
      else if i > last then endOffset
      else if i >= peak then endOffset+(amount-endOffset)*ramp(if last==peak then 1 else 1.0-(i-peak).toDouble/(last-peak))
      else startOffset+(amount-startOffset)*ramp(if peak==first then 0 else (i-first).toDouble/(peak-first))
    value*(1-fade)

  def branches(step: Int, lines: scala.collection.mutable.Builder[String,Vector[String]], conditional: String, unconditional: String): (String,String) =
    val mult = multiplier(step-1)
    if math.abs(mult) < 1e-6 then return (conditional,unconditional)
    val factor = math.max(0.001,1-mult)
    val i = step-1
    val arg = s"dd${step}_args"
    lines += s"dd${step}_sigma = sigmas[$i] * $factor"
    lines += s"dd${step}_bound = fold(tensor::to_list(tensor::arange(0, 18)), {lo = 0.0, hi = 1.0}, process = detail_bound, pred = pos, target = dd${step}_sigma)"
    lines += s"dd${step}_time = (dd${step}_bound.lo + dd${step}_bound.hi) / 2.0"
    lines += s"dd${step}_adjusted = t_to_sigma(pos, dd${step}_time)"
    lines += s"dd${step}_scales = denoiser_scalings(pos,dd${step}_adjusted)"
    lines += s"$arg = {time = dd${step}_time * 999.0, scale = dd${step}_scales.input_scale}"
    def transform(label: String, active: Boolean, fallback: String): String =
      if !active then fallback else
        val id = s"dd${step}_$label"
        lines += s"$id = predictor(unet, latent_$label, transform = detail_transform, transform_args = $arg)"
        id
    (transform("pos",mode!="uncond",conditional),transform("neg",mode!="cond",unconditional))
