package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.protocol.{GenerationRequest, LoadedModel, LoadModelRequest}
import ujson.Value

/** Builds the motion/identity model graphs using the pinned library's DGL operations. */
private[vulkan] object VulkanGraph:
  val hooks = Set("animatediff", "photomaker", "ip_adapter", "revision","sonar","semantic_guidance","neutral_prompt","latent_upscale","soft_inpainting","img2imgalt","postdiff",VulkanClipIo.id) ++ VulkanFusionPrompt.textHooks ++ VulkanLatentGraph.hooks ++ VulkanUnetHooks.hooks
  val latentHires = "latent hires refinement"
  private def quoted(s: String): String = ujson.write(ujson.Str(s))
  private def path(s: String): String = quoted(VulkanConditioning.file(s, "DGL model"))

  def render(engine: VulkanInference, device: String, model: LoadedModel, settings: LoadModelRequest,
      req: GenerationRequest, hook: String, prompt: String, loras: Seq[LoraRef], seed: Long, schedule: String,
      cancelled: () => Boolean, tweak: Option[VulkanHiresTweaks.Pass] = None,
      difference: Option[(BufferedImage, BufferedImage)] = None,
      refinementImage: Option[BufferedImage] = None): Seq[BufferedImage] =
    if hook == VulkanClipIo.id then
      VulkanClipIo.withSession(req.params.hooks(hook), if Set("sdxl", "SDXL Inpaint")(model.kind) then 2048 else 768) { session =>
        val images = renderGraph(engine, device, model, settings, req, hook, prompt, loras, seed, schedule,
          cancelled, tweak, Some(session), None, difference, refinementImage)
        if cancelled() then throw new java.util.concurrent.CancellationException()
        session.publish()
        images
      }
    else if VulkanFusionPrompt.textHooks(hook) then
      VulkanClipIo.withScratch("sdv-prompt-fusion-") { scratch =>
        renderGraph(engine, device, model, settings, req, hook, prompt, loras, seed, schedule,
          cancelled, tweak, None, Some(scratch), difference, refinementImage)
      }
    else renderGraph(engine, device, model, settings, req, hook, prompt, loras, seed, schedule, cancelled, tweak, None, None, None, None)

  private def renderGraph(engine: VulkanInference, device: String, model: LoadedModel, settings: LoadModelRequest,
      req: GenerationRequest, hook: String, prompt: String, loras: Seq[LoraRef], seed: Long, schedule: String,
      cancelled: () => Boolean, tweak: Option[VulkanHiresTweaks.Pass], clipIo: Option[VulkanClipIo.Session],
      fusionScratch: Option[Path], difference: Option[(BufferedImage, BufferedImage)],
      refinementImage: Option[BufferedImage]): Seq[BufferedImage] =
    import VulkanProtocol.{invalid, unsupported}
    def check(): Unit = if cancelled() then throw new java.util.concurrent.CancellationException()
    val p = req.params
    val customText = clipIo.nonEmpty || fusionScratch.nonEmpty
    val hiresPass = hook == latentHires
    val h = if hiresPass then ujson.Obj() else p.hooks(hook)
    val latentHook = VulkanLatentGraph.hooks(hook)
    if p.hooks.get("dynamic_thresholding").exists(!_.obj.get("enabled").contains(ujson.Bool(false))) &&
        !engine.supportsDynamicThreshold() then unsupported("This Vulkan runtime lacks Dynamic Thresholding")
    val up = Option.when(hook == "latent_upscale")(VulkanLatentUpscale.parse(p,h,schedule))
    val soft = Option.when(hook == "soft_inpainting")(new VulkanSoftInpaint(h))
    val postDiff = Option.when(hook == "postdiff")(new VulkanPostDiff(h,p,model.kind == "sdxl"))
    val unetHooks = Option.when(VulkanUnetHooks.hooks(hook)) {
      if !engine.supportsUnetHooks() then unsupported("This Vulkan runtime lacks U-Net attention and skip hooks")
      if !engine.supportsUnetResize() && p.hooks.exists((k,v) => VulkanUnetHooks.resizeHooks(k) && !v.obj.get("enabled").contains(ujson.Bool(false))) then
        unsupported("This Vulkan runtime lacks U-Net resizing hooks")
      VulkanUnetHooks.parse(p,model.kind=="sdxl",seed)
    }
    val unetArgs = unetHooks.fold("")(_.args)
    val sonar = Option.when(hook=="sonar")(new VulkanSonar(h,p,seed))
    val inverse = hook == "img2imgalt"
    if soft.nonEmpty && !p.img2img.exists(_.mask.nonEmpty) then invalid("Soft Inpainting needs an img2img mask")
    if inverse && p.img2img.isEmpty then invalid("img2img Alternative needs an input image")
    val imageHooks = if customText then Set("conditioning_highres_fix", "inpaint_difference", "img2img_hires_fix") else Set.empty[String]
    val textHooks = if fusionScratch.nonEmpty then VulkanFusionPrompt.textHooks else Set.empty[String]
    val permitted = Set(hook,"nsfw_checker","aesthetic_score","model_keyword","pnginfo_injection","adetailer","batch_face_swap") ++ imageHooks ++ textHooks ++ Option.when(hiresPass)("hires_fix_tweaks") ++ (if latentHook then VulkanLatentGraph.hooks else Set.empty[String]) ++ (if unetHooks.nonEmpty then VulkanUnetHooks.hooks else Set.empty[String])
    val other = p.hooks.filterNot { (key, value) =>
      value.obj.get("enabled").contains(ujson.Bool(false)) || !VulkanFusionPrompt.applies(key,p.prompt,p.negativePrompt)
    }.keySet -- permitted
    if other.nonEmpty || p.hires.enabled && !hiresPass || p.controlnet.exists(_.enabled) || VulkanConditioning.activeEmbeddings(req).nonEmpty then
      unsupported(s"$hook with other sampling hooks, hires, embeddings or ControlNet")
    if !customText && up.isEmpty && soft.isEmpty && !hiresPass && p.img2img.exists(_.mask.nonEmpty) then unsupported(s"masked $hook generation")
    val isVideo = hook == "animatediff"
    if isVideo && model.kind != "sd15" then unsupported("AnimateDiff motion models outside SD1.x")
    if hook == "photomaker" && (model.kind != "sdxl" || p.isImg2Img) then unsupported("PhotoMaker outside SDXL txt2img")
    if fusionScratch.nonEmpty && !Set("sd15", "sdxl", "SD 2.x", "SD 1.x Inpaint", "SD 2.x Inpaint", "SDXL Inpaint")(model.kind) then
      unsupported("Prompt Fusion with this checkpoint family")
    val inpaint = customText && (Set("SD 1.x Inpaint", "SDXL Inpaint")(model.kind) ||
      fusionScratch.nonEmpty && model.kind == "SD 2.x Inpaint")
    val family = model.kind match
      case "SDXL Inpaint" if inpaint => "sdxl_inpaint"
      case "SD 1.x Inpaint" if inpaint => "sd1_inpaint"
      case "SD 2.x Inpaint" if inpaint && fusionScratch.nonEmpty => "sd2_inpaint"
      case "SD 2.x" if fusionScratch.nonEmpty => "sd2"
      case "sdxl" => "sdxl"
      case _ => "sd1"
    val options = VulkanConditioning.options(settings, device)
    val placement = if options.placement().isEmpty then "" else s", params_backend = ${quoted(options.placement())}"
    val common = s"backend = ${quoted(device)}$placement, wtype = ${quoted(options.weights())}, family = ${quoted(family)}"
    val lines = Vector.newBuilder[String]
    val textSetup = Vector.newBuilder[String]
    lines += s"unet0 = load_unet(${path(model.path)}, $common)"
    val loadText = s"text0 = load_text_encoder(${path(model.path)}, $common)"
    lines += loadText
    textSetup += loadText
    val scale = if family.startsWith("sdxl") && model.vaePath.isEmpty then ", conv_scale = 0.03125" else ""
    lines += s"vae = load_vae(${path(model.vaePath.getOrElse(model.path))}, $common$scale)"
    for (l,i) <- loras.zipWithIndex do
      if l.unet.exists(_ != l.weight) || l.lbw.nonEmpty || l.dyn.nonEmpty || l.start.nonEmpty || l.stop.nonEmpty then unsupported("scheduled LoRA in a motion/identity graph")
      val loadLora = s"lora$i = load_lora(${path(req.loraPaths.getOrElse(l.name, invalid(s"Unresolved LoRA ${l.name}")))})"
      lines += loadLora
      textSetup += loadLora
      lines += s"unet${i+1} = apply_lora(unet$i, lora$i, weight = ${l.weight})"
      val applyText = s"text${i+1} = apply_lora(text$i, lora$i, weight = ${l.weight})"
      lines += applyText
      textSetup += applyText
    val unet = s"unet${loras.size}"
    val text = s"text${loras.size}"
    val textProgram = textSetup.result()
    lazy val adapter = path(h("model").str)
    val inputs = scala.collection.mutable.LinkedHashMap.empty[String,BufferedImage]
    sonar.flatMap(_.image).foreach(im=>inputs("sonar_ref")=im)
    val size = s"width = ${p.width}, height = ${p.height}"
    val enc = s"$size, clip_skip = ${p.clipSkip}"
    val fusion = fusionScratch.map { scratch =>
      val width = if family.startsWith("sdxl") then 2048 else if family.startsWith("sd2") then 1024 else 768
      val config = p.hooks.get(VulkanFusionPrompt.id).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
        .getOrElse(ujson.Obj())
      VulkanPromptFusion.prepare(engine,scratch,config,p,prompt,family.startsWith("sdxl"),width,textProgram,text,enc,() => check(),lines)
    }
    val frames = if isVideo then
      val context = h.obj.get("batch_size").fold(16)(_.num.toInt)
      val requested = h.obj.get("video_length").fold(16)(_.num.toInt)
      val n = if requested == 0 then context else requested
      if n < 1 || n > context || n > 32 then unsupported("AnimateDiff clips exceeding one motion context")
      if h.obj.get("free_init").exists(_.bool) || h.obj.get("last_frame").exists(_.str.nonEmpty) ||
        h.obj.get("interp").exists(v => !Set("off","")(v.str.toLowerCase)) || prompt.linesIterator.exists(_.matches("\\s*\\d+\\s*:.*")) then
        unsupported("AnimateDiff FreeInit, last-frame conditioning, interpolation or prompt travel")
      if p.batchSize != 1 then invalid("AnimateDiff batch size must be one")
      lines += s"motion = animatediff::load($adapter, backend = ${quoted(device)}$placement)"
      lines += s"unet = animatediff::apply($unet, motion)"
      lines += s"pos = predictor(unet, encode_prompt($text, ${quoted(prompt)}, $enc), prediction = \"eps\")"
      n
    else if clipIo.nonEmpty then
      lines += s"unet = $unet"
      clipIo.get.build("pos", prompt, text, enc, textProgram, engine, () => check(), lines)
      1
    else if fusion.nonEmpty then
      lines += s"unet = $unet"
      1
    else if unetHooks.nonEmpty || Set("sonar","semantic_guidance","neutral_prompt")(hook) || latentHook || up.nonEmpty || hiresPass || soft.nonEmpty || inverse || postDiff.nonEmpty then
      lines += s"unet = $unet"
      lines += s"latent_pos = encode_prompt($text, ${quoted(prompt)}, $enc)"
      lines += s"pos = predictor(unet, latent_pos$unetArgs)"
      1
    else if hook == "revision" then
      if family != "sdxl" then unsupported("Revision outside SDXL")
      val vision = sdfx.vulkan.tools.VulkanVision.revisionEncoder(() => check())
      val strength = h.obj.get("strength").fold(1.0)(_.num)
      require(strength.isFinite && strength >= 0 && strength <= 2,"Revision strength must be in 0..2")
      inputs("reference") = sdfx.vulkan.tools.DiffusionImageInput.readRgb(Path.of(h("image").str))
      lines += s"vision = load_vision_encoder(${path(vision.toString)}, backend = ${quoted(device)}$placement)"
      lines += s"pooled = tensor::mul(tensor::reshape(encode_vision(vision,reference,pooled = true), [1280]), $strength)"
      for (label,caption) <- Seq("pos"->prompt,"neg"->p.negativePrompt) do
        val cond = s"revision_$label"
        lines += s"$cond = encode_prompt($text, ${quoted(caption)}, $enc)"
        val replacement = if label == "pos" then "pooled" else "tensor::zeros_like(pooled)"
        val cross = if label=="pos" && h.obj.get("ignore_prompt").exists(_.bool) then s"tensor::zeros_like($cond.crossattn)" else s"$cond.crossattn"
        lines += s"${cond}_flat = tensor::reshape($cond.vector,[tensor::numel($cond.vector)])"
        lines += s"${cond}_replaced = tensor::replace_slice(${cond}_flat,$replacement,axis = 0,start = 0)"
        lines += s"${cond}_vector = tensor::reshape(${cond}_replaced,tensor::shape($cond.vector))"
        lines += s"$label = predictor($unet, {crossattn = $cross, vector = ${cond}_vector})"
      1
    else if hook == "ip_adapter" then
      val rawPath = Path.of(h("model").str).toAbsolutePath.normalize()
      val settings = sdfx.vulkan.tools.VulkanVision.inspect(rawPath,h.obj.get("image_encoder").fold("")(_.str),family,() => check())
      val weight = h.obj.get("scale").fold(0.7)(_.num)
      require(weight.isFinite && weight >= 0 && weight <= 2, "IP-Adapter scale must be in 0..2")
      val source = h.obj.get("image").flatMap(_.strOpt).filter(_.nonEmpty).getOrElse(invalid("IP-Adapter requires a reference image"))
      inputs("reference") = sdfx.vulkan.tools.DiffusionImageInput.readRgb(Path.of(source))
      lines += s"vision = load_vision_encoder(${path(settings.encoder.toString)}, backend = ${quoted(device)}$placement)"
      lines += s"adapter = ip_adapter::load($adapter, family = ${quoted(family)}, backend = ${quoted(device)}$placement)"
      val args = if settings.plus then "pooled = false, clip_skip = 2" else "pooled = true"
      lines += s"vision_pos = encode_vision(vision, reference, $args)"
      lines += (if settings.plus then s"vision_neg = encode_vision(vision, reference, $args, zero_input = true)"
        else "vision_neg = tensor::zeros_like(vision_pos)")
      for (name,value) <- Seq("pos" -> prompt, "neg" -> p.negativePrompt) do
        lines += s"tokens_$name = ip_adapter::project(adapter, vision_$name)"
        lines += s"cond_$name = ip_adapter::apply(encode_prompt($text, ${quoted(value)}, $enc), adapter, tokens_$name, strength = $weight)"
        lines += s"$name = predictor($unet, cond_$name)"
      1
    else
      val refs = h("images").arr.map(_.str).toVector
      require(refs.nonEmpty && refs.size <= 32, "PhotoMaker needs 1 to 32 identity photos")
      refs.zipWithIndex.foreach((file,i) => inputs(s"ref$i") = VulkanImageInputs.read(file))
      val trigger = h.obj.get("trigger_word").fold("img")(_.str)
      require(trigger.nonEmpty, "PhotoMaker needs a trigger word")
      val boundary = "(?<![\\p{L}\\p{N}_])" + java.util.regex.Pattern.quote(trigger) + "(?![\\p{L}\\p{N}_])"
      val occurrences = boundary.r
      require(occurrences.findFirstIn(prompt).isDefined, s"Prompt must include the PhotoMaker trigger $trigger")
      val actualPrompt = occurrences.replaceAllIn(prompt, "img")
      val merge = h.obj.get("start_merge_step").fold(10)(_.num.toInt)
      require(merge >= 0, "Negative PhotoMaker merge step")
      lines += s"pm = photomaker::load($adapter, family = \"sdxl\", backend = ${quoted(device)}$placement)"
      lines += s"unet = apply_lora($unet, load_lora($adapter))"
      lines += s"ids = tensor::reshape(photomaker::encode_images(pm, [${inputs.keys.mkString(",")}]), [2048, ${refs.size}])"
      lines += s"conditions = photomaker::encode_prompt($text, ${quoted(actualPrompt)}, ids, $enc)"
      lines += "identity = photomaker::apply(conditions.condition, pm, ids, conditions.mask)"
      lines += s"pos = switch_predictor(predictor(unet, conditions.plain), predictor(unet, identity), after_steps = $merge)"
      1
    if fusion.isEmpty && !Set("ip_adapter","revision")(hook) then
      clipIo match
        case Some(session) => session.build("neg", p.negativePrompt, text, enc, textProgram, engine, () => check(), lines)
        case None => lines += s"latent_neg = encode_prompt($text, ${quoted(p.negativePrompt)}, $enc)"
      if clipIo.isEmpty then lines += s"neg = predictor(unet, latent_neg$unetArgs)"
    val positive = if hook == "semantic_guidance" then VulkanSemanticGuidance.prepare(h,text,enc,family=="sdxl",lines) else "pos"
    val guidance = unetHooks.fold(s"guided = cfg($positive, neg, scale = ${p.cfgScale})")(_.guidance(p.cfgScale))
    if !customText then lines += guidance
    val maskGraph = customText && (inpaint || difference.nonEmpty || p.img2img.exists(_.mask.nonEmpty))
    // The native image pipeline defaults to the CPU implementation of the CUDA-compatible RNG.
    val randomAlgorithm = if maskGraph then "cuda" else "cpu"
    lines += s"rng = make_rng(\"$randomAlgorithm\", seed = $seed)"
    val encodeRng = if maskGraph then
      lines += s"encode_rng = make_rng(\"$randomAlgorithm\", seed = $seed)"
      "encode_rng"
    else "rng"
    val shape = s"latent_shape(vae, ${postDiff.fold(size)(_.size)}, batch = $frames)"
    var overlay: Option[VulkanImageInputs.Inputs] = None
    val tiled = if settings.vaeTiling then ", tiling = tiling(tile_size_w = 32, tile_size_h = 32)" else ""
    p.img2img match
      case None => lines += s"base = tensor::zeros($shape)"
      case Some(i) if up.nonEmpty =>
        val original = VulkanImageInputs.read(i.initImage)
        val source = if i.mask.isEmpty then original else
          val ip = i.copy(resizeMode = 0)
          val target = VulkanImageInputs.prepare(ip,p.width,p.height,seed,cancelled = cancelled)
          overlay = Some(target)
          inputs("paint_mask") = target.mask.get
          if i.inpaintFullRes then target.init
          else VulkanImageInputs.prepare(ip,original.getWidth,original.getHeight,seed,cancelled = cancelled).init
        val sw = source.getWidth/8
        val sh = source.getHeight/8
        require(sw >= 1 && sh >= 1,"Latent Upscale needs at least 8 pixels per side")
        inputs("init") = VulkanImageInputs.resize(source,sw*8,sh*8,0)
        lines += s"encoded = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(init), rng = rng$tiled))"
        lines ++= VulkanLatentUpscale.resize(up.get.method,sw,sh,p.width/8,p.height/8)
      case Some(i) =>
        val init = refinementImage.fold(VulkanImageInputs.prepare(i,p.width,p.height,seed,difference, cancelled)) { image =>
          VulkanImageInputs.Inputs(image, None, image, None, None)
        }
        inputs("init") = init.init
        init.mask.foreach { mask =>
          inputs("paint_mask") = mask
          overlay = Some(init)
        }
        val posterior = if inverse then ", posterior = \"mode\"" else ""
        lines += s"encoded = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(init), rng = $encodeRng$posterior$tiled))"
        lines += s"base = tensor::broadcast_to(encoded, $shape)"
    if overlay.nonEmpty then
      lines += "mask = tensor::threshold(image::to_f32(tensor::grayscale(paint_mask, axis = 0)), threshold = 0.5)"
    else if inpaint then lines += s"mask = tensor::full([${p.width},${p.height},1,1], 1)"
    if overlay.nonEmpty || inpaint then
      lines += "latent_mask = tensor::max_pool2d(mask, kernel = [8,8], stride = [8,8])"
    if inpaint then
      p.img2img match
        case Some(i) =>
          val weight = VulkanImageInputs.maskConditioning(p.hooks.get("conditioning_highres_fix"), i.denoisingStrength)
          lines += s"masked_pixels = tensor::lerp(image::to_f32(init), 0.5, tensor::mul(mask,$weight))"
          lines += s"masked_latent = vae_to_unet_latents(vae, vae_encode(vae, masked_pixels, rng = $encodeRng$tiled))"
        case None => lines += "masked_latent = tensor::zeros_like(base)"
      lines += "inpaint_condition = tensor::concat([latent_mask, masked_latent], axis = 2)"
      lines += "denoise_mask = tensor::max_pool2d(latent_mask, kernel = [3,3], padding_before = [1,1], padding_after = [1,1])"
    if customText then
      val extra = if inpaint then ", concat = inpaint_condition" else ""
      fusion match
        case Some(schedule) => schedule.predictors(extra,lines)
        case None =>
          lines += s"pos = predictor(unet, latent_pos$extra)"
          lines += s"neg = predictor(unet, latent_neg$extra)"
      lines += guidance
    val fullSteps = up.fold(p.steps)(_.fullSteps)
    val scheduleName = up.fold(schedule)(_.scheduler)
    lines += s"full = make_sigmas(guided, latent = base, scheduler = ${quoted(VulkanProtocol.schedulers(scheduleName))}, steps = $fullSteps)"
    val sigmas = up.fold(p.img2img.fold("full")(i => s"full[${p.steps - math.min(p.steps,(p.steps*i.denoisingStrength).toInt+1)}:]"))(_ => s"full[${fullSteps-p.steps}:]")
    val softSteps = soft.fold(0)(_ => (p.steps*p.img2img.get.denoisingStrength).toInt.max(1).min(p.steps))
    lines += s"sigmas = ${if inverse then "full" else if soft.nonEmpty then s"full[${p.steps-softSteps}:]" else sigmas}"
    val baseNoise = tweak.filter(_ => p.subseedStrength>0).flatMap { pass =>
      pass.copy(seed = seed,subseed = pass.baseSubseed,strength = p.subseedStrength,width = 0,height = 0).noise(engine,p.width/8,p.height/8)
    }
    if inverse then lines ++= VulkanInverse.start(h,p,prompt,text,enc)
    else if baseNoise.nonEmpty then lines += "start = add_noise(guided, base, base_noise, sigma = sigmas[0])"
    else lines += "start = add_noise(guided, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])"
    val eta = p.eta.fold("")(e => s", eta = $e")
    if soft.nonEmpty then lines ++= soft.get.sample(softSteps,p.width,p.height,VulkanProtocol.samplers(p.samplerName),eta,seed,tiled)
    else if postDiff.nonEmpty then lines ++= postDiff.get.sample
    else if sonar.nonEmpty then lines ++= sonar.get.sample(inputs.contains("sonar_ref"),tiled)
    else if hook == "neutral_prompt" then lines ++= VulkanNeutralPrompt.sample(p,prompt,h,text,enc)
    else if latentHook then lines ++= VulkanLatentGraph.sample(p,VulkanProtocol.samplers(p.samplerName),eta,family == "sdxl",seed)
    else
      val mask = if inpaint then ", base = base, denoise_mask = denoise_mask"
        else if overlay.nonEmpty then ", base = base, denoise_mask = latent_mask" else ""
      lines += s"latent = sample(start, sigmas, predict = guided, method = ${quoted(VulkanProtocol.samplers(p.samplerName))}, rng = rng$eta$mask)"
    val noise = tweak.flatMap { pass =>
      val (w,h) = p.hires.targetSize(p.width,p.height)
      pass.noise(engine,w/8,h/8)
    }
    if hiresPass then lines ++= VulkanHiresGraph.refinement(req,prompt,loras,tweak.fold(seed)(_.seed),unet,text,eta,device,noise.nonEmpty,tiled)
    val resultLatent = if hiresPass then "hr_latent" else "latent"
    lines += s"return {output = vae_decode(vae, unet_to_vae_latents(vae, $resultLatent)$tiled)}"
    val transforms = Seq("detail_daemon" -> VulkanDetailDaemon.functions, "cads" -> VulkanCads.functions)
      .collect { case (key,source) if latentHook && p.hooks.contains(key) => source }.mkString("\n") +
      (if Set("sonar","semantic_guidance","neutral_prompt")(hook) then VulkanCads.functions else "") +
      (if Set("sonar","neutral_prompt")(hook) then VulkanNeutralPrompt.functions else "")
    val noises = (noise.map("hr_noise" -> _) ++ baseNoise.map("base_noise" -> _)).toMap
    val names = inputs.keys.toVector ++ noises.keys
    val source = s"${transforms}fn main(${names.mkString(",")}) {\n${lines.result().mkString("\n")}\n}"
    if cancelled() then throw new java.util.concurrent.CancellationException()
    if isVideo then engine.dglFrames(source, inputs.asJava, "output").asScala.toVector
    else
      val image = engine.dglImage(source, inputs.asJava, "output", noises.asJava)
      Seq(soft.fold(overlay.fold(image)(_.finish(image)))(_.composite(overlay.get,image,cancelled)))
