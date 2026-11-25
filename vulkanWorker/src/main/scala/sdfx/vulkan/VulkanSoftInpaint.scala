package sdfx.vulkan

import java.awt.image.BufferedImage
import ujson.Value

private[vulkan] final class VulkanSoftInpaint(h: Value):
  private def option(key: String, default: Double): Double =
    val value = h.obj.get(key).fold(default)(_.num)
    require(value.isFinite,s"Invalid Soft Inpainting $key")
    value
  private val power = option("mask_blend_power",1).max(.001)
  private val ramp = (option("mask_blend_scale",.5)/option("inpaint_detail_preservation",4).max(.001)).max(.001)
  private val influence = option("composite_mask_influence",0)
  private val threshold = option("composite_difference_threshold",.5).max(1e-6)
  private val contrast = option("composite_difference_contrast",2)

  def sample(steps: Int,w: Int,h: Int,method: String,eta: String,seed: Long,tiled: String): Vector[String] =
    if !Set("euler","euler_a")(method) then VulkanProtocol.unsupported("Soft Inpainting requires Euler or Euler a")
    val code = Vector.newBuilder[String]
    code += s"""soft_rng = make_rng("cpu", seed = ${seed & 0xffffffffL})"""
    code += s"""soft_orig = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(init), rng = soft_rng, posterior = "mode"$tiled))"""
    code += "soft_noise = tensor::randn_like(soft_orig, rng = soft_rng)"
    code += s"""soft_mask = tensor::interpolate(image::to_f32(tensor::grayscale(paint_mask, axis = 0)), size = [${w/8},${h/8}], mode = "bilinear")"""
    var previous = "start"
    for i <- 0 until steps do
      val limit = math.pow(1-(i+1.0)/steps,power)
      code += s"""soft_step$i = sample($previous, sigmas[$i:${i+2}], predict = guided, method = "$method", rng = rng$eta)"""
      code += s"soft_keep$i = tensor::clamp(tensor::affine(soft_mask, scale = ${1/ramp}, offset = ${.5-limit/ramp}), 0, 1)"
      val reference = if i+1 == steps then "soft_orig" else s"add_noise(guided, soft_orig, soft_noise, sigma = sigmas[${i+1}])"
      previous = s"soft_blend$i"
      code += s"$previous = tensor::lerp($reference, soft_step$i, soft_keep$i)"
    code += s"latent = $previous"
    code.result()

  def composite(source: VulkanImageInputs.Inputs, generated: BufferedImage, cancelled: () => Boolean): BufferedImage =
    val w = generated.getWidth
    val h = generated.getHeight
    val out = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    val alpha = source.mask.get
    for y <- 0 until h do
      if cancelled() then throw new java.util.concurrent.CancellationException()
      for x <- 0 until w do
        val a = source.init.getRGB(x,y)
        val b = generated.getRGB(x,y)
        def at(rgb: Int,k: Int) = rgb >> k & 255
        val diff = (math.abs(at(a,0)-at(b,0))+math.abs(at(a,8)-at(b,8))+math.abs(at(a,16)-at(b,16)))/765.0
        val mask = (alpha.getRGB(x,y)&255)/255.0
        val boost = math.pow(diff/threshold,contrast).max(0).min(1)
        val weight = (mask*(1-influence)+mask.max(boost)*influence).max(0).min(1)
        def mix(k: Int) = (at(a,k)*(1-weight)+at(b,k)*weight).toInt << k
        out.setRGB(x,y,mix(0)|mix(8)|mix(16))
    source.finish(out)
