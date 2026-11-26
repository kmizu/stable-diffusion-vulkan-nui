package sdfx.vulkan

import sdfx.core.model.LoraRef
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.GenerationRequest
import sdfx.vulkan.tools.VulkanAestheticTraining

private[vulkan] object VulkanHiresGraph:
  private def q(value: String) = ujson.write(ujson.Str(value))

  def fullSteps(p: GenerationParams): Int =
    val h = p.hires
    val requested = if h.secondPassSteps == 0 then p.steps else h.secondPassSteps
    if h.denoisingStrength == 0 then requested
    else math.ceil(requested/h.denoisingStrength.min(.999) - 1e-9).toInt

  def refinementSteps(p: GenerationParams): Int =
    val full = fullSteps(p)
    (full * p.hires.denoisingStrength).toInt.min(full).max(1)

  /** The first pass leaves its diffusion latent in `latent`; sample again at the target resolution. */
  def refinement(req: GenerationRequest, firstPrompt: String, firstLoras: Seq[LoraRef], seed: Long,
      baseUnet: String, baseText: String, eta: String, device: String, customNoise: Boolean, tiled: String,
      cdTuner: Option[VulkanCdTuner.Options] = None, cacheArgs: String = "",
      reference: Option[VulkanReference.Config] = None,
      agent: Option[VulkanAgentAttention.Config] = None, controlWeight: Option[Double] = None,
      inpaint: Boolean = false, lllite: Option[(VulkanLLLite.Config, String)] = None,
      llul: Option[VulkanLlul.Config] = None,
      characteristic: Option[VulkanCharacteristic.Config] = None,
      composable: Option[ujson.Value] = None, runtimeLoraWeights: Boolean = false,
      aesthetic: Option[VulkanAestheticTraining.Conditioning] = None, masked: Boolean = false): Vector[String] =
    import VulkanProtocol.{invalid,unsupported}
    val p = req.params
    val h = p.hires
    val (targetWidth, targetHeight) = h.targetSize(p.width, p.height)
    val (width, height) = VulkanGraph.nativeSize(targetWidth, targetHeight)
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
    if composable.isEmpty && adapters != firstLoras then
      unet = "unet0"
      text = "text0"
      for (l,i) <- adapters.zipWithIndex do
        if !l.weight.toFloat.isFinite then invalid(s"Invalid hires LoRA weight: ${l.name}")
        if l.unet.exists(_ != l.weight) || l.lbw.nonEmpty || l.dyn.nonEmpty || l.start.nonEmpty || l.stop.nonEmpty then
          unsupported("scheduled or block hires LoRA")
        val path = VulkanConditioning.file(req.loraPaths.getOrElse(l.name,invalid(s"Unresolved hires LoRA: ${l.name}")),"LoRA")
        val (withText, withUnet) = aesthetic.fold((true, true))(a =>
          if l.weight.toFloat == 0 then (false, false) else a.components(path))
        lines += s"hr_lora$i = load_lora(${q(path)})"
        val mode = if aesthetic.nonEmpty then ", mode = \"at_runtime\"" else ""
        val nextUnet = if withUnet then s"apply_lora($unet, hr_lora$i, weight = ${l.weight}$mode)" else unet
        val nextText = if withText then s"apply_lora($text, hr_lora$i, weight = ${l.weight}$mode)" else text
        lines += s"hr_unet$i = $nextUnet"
        lines += s"hr_text$i = $nextText"
        unet = s"hr_unet$i"
        text = s"hr_text$i"
    val encoding = s"width = $width, height = $height, clip_skip = ${p.clipSkip}"
    val resolved = p.hooks.get("hires_fix_tweaks").exists(_.obj.get("prompts_resolved").exists(_.bool))
    def selected(value: String, fallback: String) = if resolved || value.nonEmpty then value else fallback
    val positivePrompt = selected(h.prompt, firstPrompt)
    val negativePrompt = selected(h.negativePrompt, p.negativePrompt)
    val full = fullSteps(p)
    require(full <= 10000,"Hires schedule exceeds 10000 steps")
    val count = refinementSteps(p)
    val loraPlan = composable.map { config =>
      val (offset, total) = VulkanComposableLora.clock(p, config, count, hires = true)
      VulkanComposableLora.plan(positivePrompt, negativePrompt, req.manualLoras, adapters, config, firstPrompt,
        count, offset, total)
    }
    val areaRatio = targetWidth.toDouble * targetHeight / (p.width.toDouble * p.height)
    val baseSteps = p.img2img.fold(p.steps)(i => VulkanImageInputs.img2imgSteps(p.steps, i.denoisingStrength))
    val incantations = VulkanIncantations.attentionPasses(p)
    val t2i = incantations && VulkanIncantations.t2i0(p.hooks).exists(t => t.strength != 0 || t.alpha > 0)
    def t2iArgs(branch: String): String =
      if t2i then
        val peer = if branch == "neg" then "hr_latent_pos" else "hr_latent_neg"
        s", t2i0_peer = $peer, t2i0_empty = hr_latent_empty" +
          VulkanIncantations.t2iArguments(p.hooks, count, p.width, p.height, "latent_pos", branch, baseSteps)
      else ""
    reference.foreach(config => lines ++= config.prepare("hr_ref", tiled))
    val agentArgs = agent.fold("")(_.args(seed, hires = true))
    val llulArgs = llul.filter(_.canApply(full)).fold("")(_.args(targetWidth, targetHeight, "llul_mask", full))
    controlWeight.foreach(_ => lines += "hr_control_hint = image::to_f32(hr_control_image)")
    val concat = if inpaint then
      lines += s"hr_inpaint_condition = tensor::interpolate(inpaint_condition, size = [${width/8},${height/8}])"
      ", concat = hr_inpaint_condition"
    else ""
    val aa = h.upscaler.contains("antialiased")
    val seg = VulkanIncantations.segArguments(p.hooks, p.width, p.height, count)
    val scfg = VulkanIncantations.scfgArguments(p.hooks, p.width, p.height, count)
    val pag = VulkanIncantations.pag(p.hooks).exists(_.canApply(count))
    val randomAlgorithm = if llul.nonEmpty || characteristic.nonEmpty || composable.nonEmpty || aesthetic.nonEmpty || incantations then "cuda" else "cpu"
    lines += s"""hr_rng = make_rng("$randomAlgorithm", seed = $seed)"""
    if h.upscaler.startsWith("Latent") then
      lines += s"hr_base = tensor::interpolate(latent, size = [${width/8},${height/8}], mode = ${q(method)}, antialias = $aa)"
    else
      val saturation = cdTuner.fold("")(_.decoderArgs)
      val floatPixels = characteristic.nonEmpty || aesthetic.nonEmpty || incantations
      val dtype = if floatPixels then ", dtype = \"f32\"" else ""
      lines += s"hr_pixels = vae_decode(vae, unet_to_vae_latents(vae, latent)$tiled$saturation$dtype)"
      val pixelModel = req.upscalerPaths.get(h.upscaler)
      val input = pixelModel.fold("hr_pixels") { file =>
        lines += s"hr_upscaler = load_upscaler(${q(VulkanConditioning.file(file,"upscaler"))}, backend = ${q(device)})"
        val pixels = if floatPixels then "image::to_u8(hr_pixels)" else "hr_pixels"
        lines += s"hr_large = upscale(hr_upscaler, $pixels)"
        "hr_large"
      }
      val floatResize = floatPixels && pixelModel.isEmpty
      val axes = if floatResize then "" else ", axes = [1,2]"
      lines += s"hr_resized = tensor::interpolate($input, size = [$width,$height]$axes, mode = ${q(method)})"
      val pixels = if floatResize then "hr_resized" else "image::to_f32(hr_resized)"
      lines += s"hr_base = vae_to_unet_latents(vae, vae_encode(vae, $pixels, rng = hr_rng$tiled))"
    val llliteArgs = lllite.fold("") { (config, path) =>
      val hint = if config.image.nonEmpty then "hr_lllite_image"
        else if h.upscaler.startsWith("Latent") then
          lines += s"hr_lllite_decoded = vae_decode(vae, unet_to_vae_latents(vae, hr_base)$tiled)"
          "hr_lllite_decoded"
        else "hr_resized"
      lines += s"hr_lllite_hint = image::to_f32($hint, channels = 3)"
      config.args(path, "hr_lllite_hint")
    }
    loraPlan.foreach(plan => lines ++= VulkanComposableLora.prepare(plan, req, "hr_clora", unet, text, encoding,
      concat, supportsBlank = true, tracePass = composable.filter(VulkanComposableLora.plotRequested).map(_ => "hires"),
      runtimeWeights = runtimeLoraWeights))
    if t2i then
      lines += s"""hr_latent_empty = encode_prompt($text, "", $encoding)"""
      lines += s"hr_latent_pos = encode_prompt($text, ${q(positivePrompt)}, $encoding, clip_a1111_chunks = true)"
      lines += s"hr_latent_neg = encode_prompt($text, ${q(negativePrompt)}, $encoding, clip_a1111_chunks = true)"
    var pagGraph = Vector.empty[String]
    for (name,prompt) <- Seq("pos" -> positivePrompt, "neg" -> negativePrompt) if loraPlan.isEmpty do
      val grown = areaRatio > 1
      val cdt = cdTuner.fold("")(t => t.args(t.weights(areaRatio, baseSteps), "cdt_map", name == "neg",
        colorShift = grown, high = true, colorMapped = grown))
      val ref = reference.fold("")(_.args("hr_ref", positive = name == "pos"))
      val encoded = if t2i then s"hr_latent_$name" else aesthetic.fold(s"encode_prompt($text, ${q(prompt)}, $encoding)")(
        _.encode(s"hr_${name}_condition", text, prompt, encoding, name == "neg", adapters, lines))
      val branchArguments = if name == "neg" then seg else scfg
      val blank = if name != "neg" || aesthetic.fold(prompt.nonEmpty)(_ => prompt.trim.nonEmpty) then ""
        else if lllite.nonEmpty then ", lllite_blank_negative = true"
        else if llul.nonEmpty then ", llul_blank_negative = true"
        else if characteristic.nonEmpty || aesthetic.nonEmpty || incantations then ", sdxl_blank_negative = true"
        else ""
      val condition = controlWeight.fold(encoded)(weight =>
        s"controlnet::apply($encoded, control_model, hr_control_hint, strength = $weight)")
      lines += s"hr_$name = predictor($unet, $condition$concat$cdt$cacheArgs$ref$agentArgs$llliteArgs$llulArgs$blank$branchArguments${t2iArgs(name)})"
      if name == "pos" && pag then
        pagGraph = VulkanIncantations.pagGraph(p.hooks, count, unet, condition, concat, "hr_pos", "hr_incantations_base", "hr_guided", t2iArgs("perturbed"))
    val sampler = h.samplerName.getOrElse(p.samplerName)
    val nativeMethod = VulkanProtocol.samplers.getOrElse(sampler,unsupported(s"hires sampler $sampler"))
    val schedule = h.scheduler.getOrElse(p.scheduler)
    val name = if schedule == "Automatic" then VulkanProtocol.defaultSchedule(sampler) else schedule
    val scheduler = VulkanProtocol.schedulers.getOrElse(name,unsupported(s"hires scheduler $name"))
    val cfg = h.cfgScale.getOrElse(p.cfgScale)
    val cfgTiming = if cacheArgs.nonEmpty then VulkanGraph.cacheCfgArguments(p.hooks, cfg, count)
      else VulkanIncantations.graphArguments(p, count)
    val chgArgs = characteristic.fold("")(_.args(p.steps, cfg))
    val branchNames = if composable.nonEmpty then "hr_clora_branch0, hr_clora_neg" else "hr_pos, hr_neg"
    val cloraArgs = loraPlan.fold("")(_.guidance("hr_clora"))
    val target = if pag then "hr_incantations_base" else "hr_guided"
    lines += s"$target = cfg($branchNames, scale = $cfg$cfgTiming$chgArgs$cloraArgs)"
    lines ++= pagGraph
    val schedulePredictor = if composable.nonEmpty then "hr_clora_schedule" else "hr_guided"
    lines += s"hr_full = make_sigmas($schedulePredictor, latent = hr_base, scheduler = ${q(scheduler)}, steps = $full)"
    lines += s"hr_sigmas = hr_full[${full-count}:]"
    val noise = if customNoise then "hr_noise" else "tensor::randn_like(hr_base, rng = hr_rng)"
    lines += s"hr_start = add_noise($schedulePredictor, hr_base, $noise, sigma = hr_sigmas[0])"
    val mask = if masked then
      val sourceMask = if inpaint then "denoise_mask" else "latent_mask"
      lines += s"""hr_mask = tensor::interpolate($sourceMask, size = [${width/8},${height/8}], mode = "nearest_max")"""
      ", base = hr_base, denoise_mask = hr_mask"
    else ""
    lines += s"hr_latent = sample(hr_start, hr_sigmas, predict = hr_guided, method = ${q(nativeMethod)}, rng = hr_rng$eta$mask)"
    lines.result()
