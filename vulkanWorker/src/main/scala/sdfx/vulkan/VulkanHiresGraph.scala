package sdfx.vulkan

import sdfx.core.model.LoraRef
import sdfx.core.protocol.GenerationRequest

private[vulkan] object VulkanHiresGraph:
  private def q(value: String) = ujson.write(ujson.Str(value))

  /** The first pass leaves its diffusion latent in `latent`; sample again at the target resolution. */
  def refinement(req: GenerationRequest, firstPrompt: String, firstLoras: Seq[LoraRef], seed: Long,
      baseUnet: String, baseText: String, eta: String, device: String, customNoise: Boolean, tiled: String,
      cdTuner: Option[VulkanCdTuner.Options] = None): Vector[String] =
    import VulkanProtocol.{invalid,unsupported}
    val p = req.params
    val h = p.hires
    val (width,height) = h.targetSize(p.width,p.height)
    val method = h.upscaler match
      case "Latent" | "Latent (antialiased)" => "bilinear"
      case "Latent (bicubic)" | "Latent (bicubic antialiased)" => "bicubic"
      case "Latent (nearest)" => "nearest"
      case "Latent (nearest-exact)" => "nearest_exact"
      case "Lanczos" => "lanczos"
      case "Nearest" => "nearest"
      case name if req.upscalerPaths.contains(name) => "lanczos"
      case name => unsupported(s"hires upscaler $name")
    val lines = Vector.newBuilder[String]
    val adapters = req.hiresLoras.getOrElse(firstLoras)
    var unet = baseUnet
    var text = baseText
    if adapters != firstLoras then
      unet = "unet0"
      text = "text0"
      for (l,i) <- adapters.zipWithIndex do
        if !l.weight.toFloat.isFinite then invalid(s"Invalid hires LoRA weight: ${l.name}")
        if l.unet.exists(_ != l.weight) || l.lbw.nonEmpty || l.dyn.nonEmpty || l.start.nonEmpty || l.stop.nonEmpty then
          unsupported("scheduled or block hires LoRA")
        val path = VulkanConditioning.file(req.loraPaths.getOrElse(l.name,invalid(s"Unresolved hires LoRA: ${l.name}")),"LoRA")
        lines += s"hr_lora$i = load_lora(${q(path)})"
        lines += s"hr_unet$i = apply_lora($unet, hr_lora$i, weight = ${l.weight})"
        lines += s"hr_text$i = apply_lora($text, hr_lora$i, weight = ${l.weight})"
        unet = s"hr_unet$i"
        text = s"hr_text$i"
    val encoding = s"width = $width, height = $height, clip_skip = ${p.clipSkip}"
    val resolved = p.hooks.get("hires_fix_tweaks").exists(_.obj.get("prompts_resolved").exists(_.bool))
    def selected(value: String, fallback: String) = if resolved || value.nonEmpty then value else fallback
    val areaRatio = width.toDouble * height / (p.width.toDouble * p.height)
    val baseSteps = p.img2img.fold(p.steps)(i => ((p.steps * i.denoisingStrength).toInt + 1).min(p.steps))
    for (name,prompt) <- Seq("pos" -> selected(h.prompt,firstPrompt), "neg" -> selected(h.negativePrompt,p.negativePrompt)) do
      val grown = areaRatio > 1
      val cdt = cdTuner.fold("")(t => t.args(t.weights(areaRatio, baseSteps), "cdt_map", name == "neg",
        colorShift = grown, high = true, colorMapped = grown))
      lines += s"hr_$name = predictor($unet, encode_prompt($text, ${q(prompt)}, $encoding)$cdt)"
    lines += s"hr_guided = cfg(hr_pos, hr_neg, scale = ${h.cfgScale.getOrElse(p.cfgScale)})"
    val aa = h.upscaler.contains("antialiased")
    lines += s"""hr_rng = make_rng("cpu", seed = $seed)"""
    if h.upscaler.startsWith("Latent") then
      lines += s"hr_base = tensor::interpolate(latent, size = [${width/8},${height/8}], mode = ${q(method)}, antialias = $aa)"
    else
      val saturation = cdTuner.fold("")(_.decoderArgs)
      lines += s"hr_pixels = vae_decode(vae, unet_to_vae_latents(vae, latent)$tiled$saturation)"
      val input = req.upscalerPaths.get(h.upscaler).fold("hr_pixels") { file =>
        lines += s"hr_upscaler = load_upscaler(${q(VulkanConditioning.file(file,"upscaler"))}, backend = ${q(device)})"
        lines += "hr_large = upscale(hr_upscaler, hr_pixels)"
        "hr_large"
      }
      lines += s"hr_resized = tensor::interpolate($input, size = [$width,$height], axes = [1,2], mode = ${q(method)})"
      lines += s"hr_base = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(hr_resized), rng = hr_rng$tiled))"
    val sampler = h.samplerName.getOrElse(p.samplerName)
    val nativeMethod = VulkanProtocol.samplers.getOrElse(sampler,unsupported(s"hires sampler $sampler"))
    val schedule = h.scheduler.getOrElse(p.scheduler)
    val name = if schedule == "Automatic" then VulkanProtocol.defaultSchedule(sampler) else schedule
    val scheduler = VulkanProtocol.schedulers.getOrElse(name,unsupported(s"hires scheduler $name"))
    val requested = if h.secondPassSteps == 0 then p.steps else h.secondPassSteps
    val full = if h.denoisingStrength == 0 then requested else math.ceil(requested/h.denoisingStrength.min(.999) - 1e-9).toInt
    require(full <= 10000,"Hires schedule exceeds 10000 steps")
    val count = (full*h.denoisingStrength).toInt.min(full).max(1)
    lines += s"hr_full = make_sigmas(hr_guided, latent = hr_base, scheduler = ${q(scheduler)}, steps = $full)"
    lines += s"hr_sigmas = hr_full[${full-count}:]"
    val noise = if customNoise then "hr_noise" else "tensor::randn_like(hr_base, rng = hr_rng)"
    lines += s"hr_start = add_noise(hr_guided, hr_base, $noise, sigma = hr_sigmas[0])"
    val mask = if p.img2img.exists(_.mask.nonEmpty) then
      lines += s"""hr_mask = tensor::interpolate(latent_mask, size = [${width/8},${height/8}], mode = "nearest_max")"""
      ", base = hr_base, denoise_mask = hr_mask"
    else ""
    lines += s"hr_latent = sample(hr_start, hr_sigmas, predict = hr_guided, method = ${q(nativeMethod)}, rng = hr_rng$eta$mask)"
    lines.result()
