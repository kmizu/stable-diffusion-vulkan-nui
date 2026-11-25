package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

/** DDIM inversion carries normalized z_t between ascending training timesteps. */
private[vulkan] object VulkanInverse:
  def start(h: Value,p: GenerationParams,prompt: String,encoder: String,enc: String): Vector[String] =
    val configured = h.obj.get("inversion_steps").fold(0.0)(_.num)
    require(configured.isValidInt && configured >= 0 && configured <= 150,"Invalid inversion step count")
    val steps = if configured == 0 then p.steps else configured.toInt
    val scale = h.obj.get("decode_cfg_scale").fold(1.0)(_.num)
    val mix = h.obj.get("randomness").fold(0.0)(_.num)
    require(scale.isFinite && scale>=0 && mix.isFinite && mix>=0 && mix<=1,"Invalid inversion CFG or randomness")
    def overridePrompt(key: String, default: String) = ujson.write(ujson.Str(h.obj.get(key).map(_.str).filter(_.nonEmpty).getOrElse(default)))
    val lines = Vector.newBuilder[String]
    lines += s"inv_pos = predictor(unet, encode_prompt($encoder, ${overridePrompt("override_prompt",prompt)}, $enc))"
    lines += s"inv_neg = predictor(unet, encode_prompt($encoder, ${overridePrompt("override_negative",p.negativePrompt)}, $enc))"
    lines += s"inv_cfg = cfg(inv_pos, inv_neg, scale = $scale)"
    // SD1.x/SDXL single-file scheduler configs use steps_offset=1 and set_alpha_to_one=false.
    lines += "iv_initial_sig = t_to_sigma(inv_pos, t = 0)"
    lines += "iv_initial_scales = denoiser_scalings(inv_pos, iv_initial_sig)"
    lines += "iv_initial_norm = iv_initial_scales.input_scale"
    var previous = "base"
    for i <- 0 until steps do
      // Diffusers falls back to leading timestep spacing for the linspace Euler configurations.
      val t = (i*(1000/steps)+1).min(999)
      lines += s"iv_sig$i = t_to_sigma(inv_pos, t = ${t/999.0})"
      lines += s"iv_scales$i = denoiser_scalings(inv_pos, iv_sig$i)"
      lines += s"iv_norm$i = iv_scales$i.input_scale"
      lines += s"iv_model$i = tensor::div($previous, iv_norm$i)"
      lines += s"""iv_denoised$i = sample(iv_model$i, tensor::vector([iv_sig$i,0]), predict = inv_cfg, method = "euler", rng = rng)"""
      lines += s"iv_eps$i = tensor::div(tensor::add(iv_model$i, tensor::affine(iv_denoised$i, scale = -1)), iv_sig$i)"
      val previousSigma = if i == 0 then "iv_initial_sig" else s"iv_sig${i-1}"
      val previousNorm = if i == 0 then "iv_initial_norm" else s"iv_norm${i-1}"
      lines += s"iv_x$i = tensor::mul(tensor::add(tensor::div($previous,$previousNorm), tensor::mul(iv_eps$i,iv_sig$i-$previousSigma)),iv_norm$i)"
      previous = s"iv_x$i"
    if mix>0 then
      val norm = math.hypot(1-mix,mix)
      lines += s"iv_mixed = tensor::add(tensor::affine($previous,scale = ${(1-mix)/norm}), tensor::affine(tensor::randn_like(base,rng = rng),scale = ${mix/norm}))"
      previous = "iv_mixed"
    lines += s"start = add_noise(guided, tensor::zeros_like(base), $previous, sigma = sigmas[0])"
    lines.result()
