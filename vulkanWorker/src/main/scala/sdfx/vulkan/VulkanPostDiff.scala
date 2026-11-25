package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] final class VulkanPostDiff(config: Value, p: GenerationParams, sdxl: Boolean):
  if p.isImg2Img || p.samplerName != "Euler" then VulkanProtocol.unsupported("PostDiff outside Euler txt2img")
  private val scale = config.obj.get("scale").fold(0.5)(_.num)
  private val portion = config.obj.get("portion").fold(0.4)(_.num)
  private val snap = config.obj.get("snap").exists(_.bool)
  require(scale.isFinite && scale > 0 && scale < 1 && portion.isFinite && portion > 0 && portion < 1,"Invalid PostDiff settings")
  private val align = if sdxl then 32 else 64
  private def lower(full: Int): Int =
    val raw = full*scale
    val pixels = if snap then math.max(align,math.min(full-align,math.rint(raw/align).toInt*align)) else raw.toInt
    require((snap || pixels==raw) && pixels>0 && pixels < full && pixels%align == 0,
      s"PostDiff first-stage resolution must align to $align pixels")
    pixels
  private val transition = math.floor(p.steps*portion).toInt
  require(transition >= 1 && transition < p.steps,"PostDiff needs sampling steps on both sides of the transition")
  val size = s"width = ${lower(p.width)}, height = ${lower(p.height)}"

  def sample: Seq[String] =
    val i = transition-1
    val low = if i==0 then "start" else s"""sample(start, sigmas[:$transition], predict = guided, method = "euler", rng = rng)"""
    Seq(
      s"low = $low",
      s"""clean = sample(low, tensor::vector([sigmas[$i],0], dtype = "f32"), predict = guided, method = "euler", rng = rng)""",
      s"""resized = tensor::interpolate(clean, size = [${p.width/8},${p.height/8}], axes = [0,1], mode = "bilinear")""",
      s"mixed = add_noise(guided, resized, tensor::randn_like(resized, rng = rng), sigma = sigmas[$transition])",
      s"""latent = sample(mixed, sigmas[$transition:], predict = guided, method = "euler", rng = rng)"""
    )
