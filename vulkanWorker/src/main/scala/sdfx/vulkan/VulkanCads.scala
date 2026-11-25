package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanCads:
  val functions = """fn cads_reduce(x, index) {
    size = tensor::numel(x)
    half = math::ceil_div(size,2)
    padded = tensor::pad(x, [0,half * 2 - size])
    return tensor::add(padded[:half], padded[half:])
  }
  fn cads_mean(x) {
    size = tensor::numel(x)
    flat = tensor::reshape(x, [size])
    total = fold(tensor::to_list(tensor::arange(0, 30)), flat, process = cads_reduce)
    return tensor::to_list(total, element = "number")[0] / size
  }
  fn cads_sqrt_step(estimate,index,value) {
    return (estimate + value / estimate) / 2.0
  }
  fn cads_sqrt(x) {
    safe = tensor::to_list(tensor::clamp(tensor::vector([x], dtype = "f32"), min = 0.00000000000001, max = 1.0e30), element = "number")[0]
    return fold(tensor::to_list(tensor::arange(0, 25)), safe + 1.0, process = cads_sqrt_step, value = safe)
  }
  fn cads_rescale(source, noisy, mixing) {
    a = cads_mean(source)
    b = cads_mean(noisy)
    ca = tensor::add(source, -a)
    cb = tensor::add(noisy, -b)
    va = cads_mean(tensor::mul(ca,ca))
    vb = cads_mean(tensor::mul(cb,cb))
    corrected = tensor::add(tensor::mul(cb,cads_sqrt(va) / cads_sqrt(vb)), a)
    return tensor::lerp(noisy, corrected, mixing)
  }
  """

private[vulkan] final class VulkanCads(config: Value, total: Int, sdxl: Boolean):
  private def number(k: String, default: Double): Double =
    val v = config.obj.get(k).fold(default)(_.num)
    require(v.isFinite && v >= 0 && v <= 1,s"Invalid CADS $k")
    v
  private val tau1 = number("tau1",0.6)
  private val tau2 = number("tau2",0.9)
  require(tau2 >= tau1,"CADS tau2 must be at least tau1")
  private val scale = number("noise_scale",0.25)
  private val mixing = number("mixing_factor",1)
  private val rescale = config.obj.get("rescale").fold(true)(_.bool)
  private val negative = config.obj.get("apply_to_uncond").exists(_.bool)

  def branches(step: Int, lines: scala.collection.mutable.Builder[String,Vector[String]]): (String,String) =
    val time = 1-(step-1).toDouble/math.max(1,total-1)
    val gamma = if time<=tau1 then 1.0 else if time>=tau2 then 0.0 else (tau2-time)/(tau2-tau1)
    if gamma >= 1 then return ("pos","neg")
    def perturb(label: String): String =
      val context = s"latent_$label"
      val source = s"$context.crossattn"
      val name = s"cads${step}_$label"
      val pooled = if sdxl then s", vector = $context.vector" else ""
      lines += s"${name}_noise = tensor::randn_like($source, rng = cads_rng)"
      lines += s"${name}_raw = tensor::add(tensor::mul($source,${math.sqrt(gamma)}),tensor::mul(${name}_noise,${scale*math.sqrt(1-gamma)}))"
      val embedding = if rescale then s"cads_rescale($source,${name}_raw,$mixing)" else s"${name}_raw"
      lines += s"$name = predictor(unet,{crossattn = $embedding$pooled})"
      name
    val neg = if negative then perturb("neg") else "neg"
    (perturb("pos"),neg)
