package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.protocol.{GenerationRequest, LoadedModel, LoadModelRequest}
import sdfx.vulkan.tools.VulkanAestheticTraining
import ujson.Value

/** Builds the motion/identity model graphs using the pinned library's DGL operations. */
private[vulkan] object VulkanGraph:
  val imageAttentionHooks = Set(VulkanReference.id, VulkanLLLite.id)
  val cacheHooks = Set(VulkanTGate.id, VulkanDeepCache.id)
  val cacheGuidanceHooks = Set("cfg_cache", "cfg_schedule")

  def cacheCfgArguments(hooks: Map[String, Value], cfg: Double, steps: Int): String =
    val schedule = VulkanConditioning.cfgSchedule(hooks, cfg, steps)
      .fold("")(values => ", schedule = " + values.mkString("[", ",", "]"))
    schedule + VulkanGuidance.cacheSchedule(hooks).fold("") { (start, interval) =>
      s", cfg_cache_start = $start, cfg_cache_interval = $interval"
    }

  val aestheticHook = VulkanAestheticTraining.id
  val hooks = Set("animatediff", "photomaker", "ip_adapter", "revision","sonar","semantic_guidance","neutral_prompt","latent_upscale","soft_inpainting","img2imgalt","postdiff",VulkanClipIo.id,VulkanLatentModifier.id,VulkanNegPip.id,VulkanFooocusInpaint.id,VulkanCdTuner.id,VulkanAgentAttention.id,VulkanLlul.id,VulkanCharacteristic.id,VulkanComposableLora.id,aestheticHook) ++ imageAttentionHooks ++ cacheHooks ++ VulkanFusionPrompt.textHooks ++ VulkanLatentGraph.hooks ++ VulkanUnetHooks.hooks
  val latentHires = "latent hires refinement"
  private val unetFamilies = Set("sd15", "sdxl", "SD 2.x", "SD 1.x Inpaint", "SD 2.x Inpaint", "SDXL Inpaint")
  private def quoted(s: String): String = ujson.write(ujson.Str(s))
  private def path(s: String): String = quoted(VulkanConditioning.file(s, "DGL model"))

  /** Matches the pinned SD1.x, SD2.x and SDXL VAE's latent_shape alignment. */
  def nativeSize(width: Int, height: Int): (Int, Int) =
    def align(side: Int): Int = math.max(64, ((side.toLong + 32) / 64 * 64).toInt)
    (align(width), align(height))

  def render(engine: VulkanInference, device: String, model: LoadedModel, settings: LoadModelRequest,
      req: GenerationRequest, hook: String, prompt: String, loras: Seq[LoraRef], seed: Long, schedule: String,
      cancelled: () => Boolean, tweak: Option[VulkanHiresTweaks.Pass] = None,
      difference: Option[(BufferedImage, BufferedImage)] = None,
      refinementImage: Option[BufferedImage] = None,
      fooocus: Option[VulkanFooocusInpaint.Assets] = None,
      cdRefinement: Option[VulkanCdTuner.Options] = None,
      preparedInput: Option[VulkanImageInputs.Inputs] = None,
      aestheticDigest: String => Unit = _ => (), aestheticProgress: (Int, Float) => Unit = (_, _) => ()): Seq[BufferedImage] =
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
    else if hook == aestheticHook then
      VulkanClipIo.withScratch("sdv-aesthetic-") { scratch =>
        scala.util.Using.resource(engine.temporaryDglModelFiles()) { _ =>
          renderGraph(engine, device, model, settings, req, hook, prompt, loras, seed, schedule,
            cancelled, tweak, None, None, difference, refinementImage, preparedInput = preparedInput,
            aestheticScratch = Some(scratch), aestheticDigest = aestheticDigest, aestheticProgress = aestheticProgress)
        }
      }
    else renderGraph(engine, device, model, settings, req, hook, prompt, loras, seed, schedule, cancelled, tweak,
      None, None, difference, refinementImage, fooocus, cdRefinement, preparedInput)

  private def renderGraph(engine: VulkanInference, device: String, model: LoadedModel, settings: LoadModelRequest,
      req: GenerationRequest, hook: String, prompt: String, loras: Seq[LoraRef], seed: Long, schedule: String,
      cancelled: () => Boolean, tweak: Option[VulkanHiresTweaks.Pass], clipIo: Option[VulkanClipIo.Session],
      fusionScratch: Option[Path], difference: Option[(BufferedImage, BufferedImage)],
      refinementImage: Option[BufferedImage], fooocus: Option[VulkanFooocusInpaint.Assets] = None,
      cdRefinement: Option[VulkanCdTuner.Options] = None,
      preparedInput: Option[VulkanImageInputs.Inputs] = None, aestheticScratch: Option[Path] = None,
      aestheticDigest: String => Unit = _ => (), aestheticProgress: (Int, Float) => Unit = (_, _) => ()): Seq[BufferedImage] =
    import VulkanProtocol.{invalid, unsupported}
    def check(): Unit = if cancelled() then throw new java.util.concurrent.CancellationException()
    val p = req.params
    val aesthetic = aestheticScratch.map { scratch =>
      if !engine.supportsAestheticTraining() then unsupported("This Vulkan runtime lacks Aesthetic Gradients training")
      val sdxl = Set("sdxl", "SDXL Inpaint")(model.kind)
      if !sdxl && !Set("sd15", "SD 1.x Inpaint")(model.kind) then
        unsupported("Aesthetic Gradients requires an SD1.x or SDXL CLIP-L encoder")
      val config = VulkanAestheticTraining.settings(p.hooks(aestheticHook))
      (config, scratch, sdxl)
    }
    val cacheGraph = cacheHooks(hook)
    def cacheConfig(id: String): Option[Value] =
      Option.when(cacheGraph)(p.hooks.get(id).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))).flatten
    val tgate = cacheConfig(VulkanTGate.id).map { config =>
      if !engine.supportsTGate() then unsupported("This Vulkan runtime lacks T-GATE")
      if !unetFamilies(model.kind) then unsupported("T-GATE requires an SD1.x, SD2.x or SDXL U-Net")
      VulkanTGate.gate(config)
    }
    val deepcache = cacheConfig(VulkanDeepCache.id).map { config =>
      if !engine.supportsDeepCache() then unsupported("This Vulkan runtime lacks DeepCache")
      if !unetFamilies(model.kind) then unsupported("DeepCache requires an SD1.x, SD2.x or SDXL U-Net")
      if p.controlnet.exists(_.enabled) then unsupported("DeepCache with ControlNet")
      VulkanDeepCache.parse(config)
    }
    val cdTuner = Option.when(hook == VulkanCdTuner.id) {
      if !engine.supportsCdTuner() then unsupported("This Vulkan runtime lacks CD Tuner")
      if !unetFamilies(model.kind) then unsupported("CD Tuner requires an SD1.x, SD2.x or SDXL U-Net")
      cdRefinement.orElse(VulkanCdTuner.config(p.hooks, Set("sdxl", "SDXL Inpaint")(model.kind)))
    }.flatten
    def attentionConfig(id: String): Option[Value] =
      p.hooks.get(id).filter(config => imageAttentionHooks(hook) && !config.obj.get("enabled").contains(ujson.Bool(false)))
    val reference = attentionConfig(VulkanReference.id).map { config =>
      if !engine.supportsReferenceOnly() then unsupported("This Vulkan runtime lacks Reference only")
      if !unetFamilies(model.kind) then unsupported("Reference only requires an SD1.x, SD2.x or SDXL U-Net")
      VulkanReference.parse(config)
    }
    val lllite = attentionConfig(VulkanLLLite.id).map { raw =>
      if !engine.supportsLLLite() then unsupported("This Vulkan runtime lacks ControlNet-LLLite")
      if !Set("sdxl", "SDXL Inpaint")(model.kind) then unsupported("ControlNet-LLLite requires an SDXL U-Net")
      val config = VulkanLLLite.parse(raw)
      config -> VulkanConditioning.file(req.controlnetPaths.getOrElse(config.model, config.model), "ControlNet-LLLite model")
    }
    val agent = Option.when(hook == VulkanAgentAttention.id) {
      if !engine.supportsAgentAttention() then unsupported("This Vulkan runtime lacks Agent Attention")
      if !unetFamilies(model.kind) then unsupported("Agent Attention requires an SD1.x, SD2.x or SDXL U-Net")
      VulkanAgentAttention.parse(p.hooks(hook))
    }
    val llul = Option.when(hook == VulkanLlul.id) {
      if !engine.supportsLlul() then unsupported("This Vulkan runtime lacks LLuL")
      if !unetFamilies(model.kind) then unsupported("LLuL requires an SD1.x, SD2.x or SDXL U-Net")
      VulkanLlul.parse(p.hooks(hook))
    }
    val characteristic = Option.when(hook == VulkanCharacteristic.id) {
      if !engine.supportsCharacteristic() then unsupported("This Vulkan runtime lacks Characteristic Guidance")
      if !unetFamilies(model.kind) then unsupported("Characteristic Guidance requires an SD1.x, SD2.x or SDXL U-Net")
      if p.controlnet.exists(_.enabled) then unsupported("Characteristic Guidance with ControlNet")
      VulkanCharacteristic.parse(p.hooks(hook))
    }
    val composable = Option.when(hook == VulkanComposableLora.id) {
      if !engine.supportsComposableLora() then unsupported("This Vulkan runtime lacks Composable LoRA")
      if !unetFamilies(model.kind) then unsupported("Composable LoRA requires an SD1.x, SD2.x or SDXL U-Net")
      if Seq("adetailer", "batch_face_swap").exists(key =>
        p.hooks.get(key).exists(value => !value.obj.get("enabled").contains(ujson.Bool(false))))
      then unsupported("Composable LoRA in detector-driven refinement passes")
      p.hooks(hook)
    }
    val runtimeLoraWeights = composable.nonEmpty && engine.supportsRuntimeLoraWeights()
    val passSteps = p.img2img.fold(p.steps)(i => VulkanImageInputs.img2imgSteps(p.steps, i.denoisingStrength))
    val loraPlan = composable.map { config =>
      val (offset, total) = VulkanComposableLora.clock(p, config, passSteps, hires = false)
      VulkanComposableLora.plan(prompt, p.negativePrompt, req.manualLoras, loras, config, p.hires.prompt,
        passSteps, offset, total)
    }
    val signedText = hook == VulkanNegPip.id
    if signedText && !engine.supportsNegPip() then unsupported("This Vulkan runtime lacks NegPiP")
    val customText = clipIo.nonEmpty || fusionScratch.nonEmpty || signedText
    val unetGraph = VulkanUnetHooks.hooks(hook)
    val tileConfig = p.hooks.get(VulkanUnetHooks.tiledDiffusion).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
    val needsTiles = tileConfig.exists(VulkanUnetHooks.needsTiles(_,p.width,p.height))
    val tiledDiffusion = unetGraph && needsTiles
    val modifier = Option.when(customText || unetGraph || hook == VulkanLatentModifier.id)(
      VulkanLatentModifier.config(p)).flatten
    if modifier.nonEmpty && !engine.supportsLatentModifier() then unsupported("This Vulkan runtime lacks Latent Modifier")
    val patchedAttention = reference.nonEmpty || lllite.nonEmpty || agent.nonEmpty || llul.nonEmpty
    val incantations = hook == VulkanIncantations.id && VulkanIncantations.attentionPasses(p)
    val segArgs = if incantations then VulkanIncantations.segArguments(p.hooks, p.width, p.height, passSteps) else ""
    val scfgArgs = if incantations then VulkanIncantations.scfgArguments(p.hooks, p.width, p.height, passSteps) else ""
    val t2i = incantations && VulkanIncantations.t2i0(p.hooks).exists(t => t.strength != 0 || t.alpha > 0)
    def t2iArgs(branch: String): String =
      if t2i then
        val peer = if branch == "neg" then "latent_pos" else "latent_neg"
        s", t2i0_peer = $peer, t2i0_empty = latent_empty" +
          VulkanIncantations.t2iArguments(p.hooks, passSteps, p.width, p.height, "latent_pos", branch, retain = p.hires.enabled)
      else ""
    val incantationsPag = incantations && VulkanIncantations.pag(p.hooks).exists(_.canApply(passSteps))
    val patchedInputs = customText || unetGraph || modifier.nonEmpty || fooocus.nonEmpty || cdTuner.nonEmpty || cacheGraph || patchedAttention || characteristic.nonEmpty || composable.nonEmpty || aesthetic.nonEmpty || incantations
    val supportsControl = unetGraph || cdTuner.nonEmpty || tgate.nonEmpty || patchedAttention || aesthetic.nonEmpty || incantations || modifier.nonEmpty && !customText
    val hiresPass = hook == latentHires || (cdTuner.nonEmpty || cacheGraph || patchedAttention || characteristic.nonEmpty || composable.nonEmpty || aesthetic.nonEmpty || incantations) && p.hires.enabled
    val h = if hiresPass then ujson.Obj() else p.hooks(hook)
    val latentHook = VulkanLatentGraph.hooks(hook)
    if p.hooks.get("dynamic_thresholding").exists(!_.obj.get("enabled").contains(ujson.Bool(false))) &&
        !engine.supportsDynamicThreshold() then unsupported("This Vulkan runtime lacks Dynamic Thresholding")
    val up = Option.when(hook == "latent_upscale")(VulkanLatentUpscale.parse(p,h,schedule))
    val soft = Option.when(hook == "soft_inpainting")(new VulkanSoftInpaint(h))
    val postDiff = Option.when(hook == "postdiff")(new VulkanPostDiff(h,p,model.kind == "sdxl"))
    val unetHooks = Option.when(unetGraph) {
      if !engine.supportsUnetHooks() then unsupported("This Vulkan runtime lacks U-Net attention and skip hooks")
      if tiledDiffusion && !engine.supportsTiledDiffusion() then unsupported("This Vulkan runtime lacks MultiDiffusion")
      if !engine.supportsUnetResize() && p.hooks.exists((k,v) => VulkanUnetHooks.resizeHooks(k) && !v.obj.get("enabled").contains(ujson.Bool(false))) then
        unsupported("This Vulkan runtime lacks U-Net resizing hooks")
      val params = if tiledDiffusion then p else p.copy(hooks = p.hooks - VulkanUnetHooks.tiledDiffusion)
      VulkanUnetHooks.parse(params,Set("sdxl","SDXL Inpaint")(model.kind),seed)
    }
    val cacheArgs = tgate.fold("")(n => s", tgate_step = $n") + deepcache.fold("")(_.args)
    val unetArgs = unetHooks.fold("")(_.args) + cacheArgs + agent.fold("")(_.args(seed, hires = false))
    val sonar = Option.when(hook=="sonar")(new VulkanSonar(h,p,seed))
    val inverse = hook == "img2imgalt"
    if soft.nonEmpty && !p.img2img.exists(_.mask.nonEmpty) then invalid("Soft Inpainting needs an img2img mask")
    if inverse && p.img2img.isEmpty then invalid("img2img Alternative needs an input image")
    val imageHooks = if patchedInputs then Set("conditioning_highres_fix", "inpaint_difference") ++
      Option.when(characteristic.isEmpty && composable.isEmpty)("img2img_hires_fix") else Set.empty[String]
    val textHooks = if fusionScratch.nonEmpty then VulkanFusionPrompt.textHooks
      else if signedText then Set(VulkanWeightGradient.id)
      else Set.empty[String]
    val permitted = Set(hook,"nsfw_checker","aesthetic_score","model_keyword","pnginfo_injection","adetailer","batch_face_swap") ++ imageHooks ++ textHooks ++
      Option.when(hook == latentHires && !VulkanIncantations.attentionPasses(p))(VulkanIncantations.id) ++
      Option.when(reference.nonEmpty)(VulkanReference.id) ++ Option.when(lllite.nonEmpty)(VulkanLLLite.id) ++ Option.when(hiresPass)("hires_fix_tweaks") ++ (if cacheGraph then cacheHooks ++ cacheGuidanceHooks else Set.empty[String]) ++ Option.when(modifier.nonEmpty)(VulkanLatentModifier.id) ++ (if latentHook then VulkanLatentGraph.hooks else Set.empty[String]) ++ (if unetHooks.nonEmpty then VulkanUnetHooks.hooks else Set.empty[String])
    val other = p.hooks.filterNot { (key, value) =>
      value.obj.get("enabled").contains(ujson.Bool(false)) || !VulkanFusionPrompt.applies(key,p.prompt,p.negativePrompt) ||
        key == VulkanIncantations.id && VulkanIncantations.cfg(p.hooks).isEmpty && !VulkanIncantations.attentionPasses(p) ||
        key == VulkanUnetHooks.tiledDiffusion && !needsTiles ||
        key == VulkanLatentModifier.id && VulkanLatentModifier.config(p).isEmpty ||
        key == VulkanNegPip.id && !VulkanNegPip.applies(p,prompt,p.negativePrompt) ||
        key == VulkanCdTuner.id && VulkanCdTuner.config(p.hooks, model.kind == "sdxl").isEmpty
    }.keySet -- permitted
    if other.nonEmpty || p.hires.enabled && !hiresPass || !supportsControl && p.controlnet.exists(_.enabled) || VulkanConditioning.activeEmbeddings(req).nonEmpty then
      unsupported(s"$hook with other sampling hooks, hires, embeddings or ControlNet")
    if !patchedInputs && up.isEmpty && soft.isEmpty && !hiresPass && p.img2img.exists(_.mask.nonEmpty) then unsupported(s"masked $hook generation")
    val isVideo = hook == "animatediff"
    if isVideo && model.kind != "sd15" then unsupported("AnimateDiff motion models outside SD1.x")
    if hook == "photomaker" && (model.kind != "sdxl" || p.isImg2Img) then unsupported("PhotoMaker outside SDXL txt2img")
    if fusionScratch.nonEmpty && !unetFamilies(model.kind) then
      unsupported("Prompt Fusion with this checkpoint family")
    val sd2 = fusionScratch.nonEmpty || signedText || unetGraph || modifier.nonEmpty || cdTuner.nonEmpty || cacheGraph || patchedAttention || characteristic.nonEmpty || composable.nonEmpty || incantations
    val inpaint = patchedInputs && (Set("SD 1.x Inpaint", "SDXL Inpaint")(model.kind) ||
      sd2 && model.kind == "SD 2.x Inpaint")
    if (unetGraph || signedText || modifier.nonEmpty || incantations) && !unetFamilies(model.kind) then
      unsupported(s"$hook with this checkpoint family")
    val family = model.kind match
      case "SDXL Inpaint" if inpaint => "sdxl_inpaint"
      case "SD 1.x Inpaint" if inpaint => "sd1_inpaint"
      case "SD 2.x Inpaint" if inpaint && sd2 => "sd2_inpaint"
      case "SD 2.x" if sd2 => "sd2"
      case "sdxl" => "sdxl"
      case _ => "sd1"
    val options = VulkanConditioning.options(settings, device)
    val placement = if options.placement().isEmpty then "" else s", params_backend = ${quoted(options.placement())}"
    val common = s"backend = ${quoted(device)}$placement, wtype = ${quoted(options.weights())}, family = ${quoted(family)}"
    val lines = Vector.newBuilder[String]
    val textSetup = Vector.newBuilder[String]
    val unetPath = fooocus.fold(model.path)(_.unet.toString)
    lines += s"unet0 = load_unet(${path(unetPath)}, $common)"
    val loadText = s"text0 = load_text_encoder(${path(model.path)}, $common)"
    lines += loadText
    textSetup += loadText
    def loraFile(l: LoraRef): String =
      VulkanConditioning.file(req.loraPaths.getOrElse(l.name, invalid(s"Unresolved LoRA ${l.name}")), "LoRA")
    val aestheticConditioning = aesthetic.map { (config, scratch, sdxl) =>
      val trained = scala.collection.mutable.Map.empty[(String, Seq[(String, Float)]), String]
      val layouts = scala.collection.mutable.Map.empty[String, (Boolean, Boolean)]
      def components(path: String): (Boolean, Boolean) =
        layouts.getOrElseUpdate(path, VulkanAestheticTraining.loraComponents(path))
      var fingerprint = config.sha256
      VulkanAestheticTraining.Conditioning(config, sdxl, (caption, adapters) => {
        val textAdapters = adapters.filter(_.weight.toFloat != 0).map(l => loraFile(l) -> l.weight.toFloat)
          .filter((file, _) => components(file)._1)
        trained.getOrElseUpdate((caption, textAdapters), {
          val name = s"text_aesthetic_${trained.size}"
          val (tuned, digest) = VulkanAestheticTraining.train(engine, device, Path.of(model.path), sdxl,
            caption, config.copy(sha256 = fingerprint), scratch, textAdapters, cancelled, aestheticProgress)
          fingerprint = digest
          aestheticDigest(digest)
          lines += s"$name = load_text_encoder(${path(model.path)}, backend = ${quoted(device)}$placement, wtype = \"f32\", family = ${quoted(family)}, clip_l_path = ${path(tuned.toString)})"
          name
        })
      }, components)
    }
    val scale = if family.startsWith("sdxl") && model.vaePath.isEmpty then ", conv_scale = 0.03125" else ""
    lines += s"vae = load_vae(${path(model.vaePath.getOrElse(model.path))}, $common$scale)"
    val sharedLoras = if composable.nonEmpty then Nil else loras
    for (l,i) <- sharedLoras.zipWithIndex do
      if l.unet.exists(_ != l.weight) || l.lbw.nonEmpty || l.dyn.nonEmpty || l.start.nonEmpty || l.stop.nonEmpty then unsupported("scheduled LoRA in a motion/identity graph")
      val file = loraFile(l)
      val (withText, withUnet) = aestheticConditioning.fold((true, true))(a =>
        if l.weight.toFloat == 0 then (false, false) else a.components(file))
      val loadLora = s"lora$i = load_lora(${path(file)})"
      lines += loadLora
      textSetup += loadLora
      val mode = if aestheticConditioning.nonEmpty then ", mode = \"at_runtime\"" else ""
      val adaptedUnet = if withUnet then s"apply_lora(unet$i, lora$i, weight = ${l.weight}$mode)" else s"unet$i"
      lines += s"unet${i+1} = $adaptedUnet"
      val adaptedText = if withText then s"apply_lora(text$i, lora$i, weight = ${l.weight}$mode)" else s"text$i"
      val applyText = s"text${i+1} = $adaptedText"
      lines += applyText
      textSetup += applyText
    val unet = s"unet${sharedLoras.size}"
    val text = s"text${sharedLoras.size}"
    val textProgram = textSetup.result()
    lazy val adapter = path(h("model").str)
    val inputs = scala.collection.mutable.LinkedHashMap.empty[String,BufferedImage]
    sonar.flatMap(_.image).foreach(im=>inputs("sonar_ref")=im)
    val baseSize = (p.width, p.height)
    val (width, height) = if up.isEmpty && postDiff.isEmpty then nativeSize(p.width, p.height) else baseSize
    val hiresSize = p.hires.targetSize(p.width, p.height)
    val (hiresWidth, hiresHeight) = nativeSize(hiresSize._1, hiresSize._2)
    val size = s"width = $width, height = $height"
    val enc = s"$size, clip_skip = ${p.clipSkip}"
    if t2i then lines += s"""latent_empty = encode_prompt($text, "", $enc)"""
    def condition(label: String, caption: String, negative: Boolean): Unit =
      aestheticConditioning match
        case None =>
          val tokenIds = if t2i then ", clip_a1111_chunks = true" else ""
          lines += s"$label = encode_prompt($text, ${quoted(caption)}, $enc$tokenIds)"
        case Some(conditioning) => conditioning.encode(label, text, caption, enc, negative, sharedLoras, lines)
    def onCanvas(image: BufferedImage): BufferedImage =
      if (image.getWidth, image.getHeight) == (width, height) then image
      else VulkanImageInputs.resize(image, width, height, 0)
    val textSchedule = if signedText then Some(VulkanNegPip.prepare(p,prompt,text,enc,lines))
      else fusionScratch.map { scratch =>
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
    else if textSchedule.nonEmpty then
      lines += s"unet = $unet"
      1
    else if composable.nonEmpty then 1
    else if unetHooks.nonEmpty || modifier.nonEmpty || fooocus.nonEmpty || cdTuner.nonEmpty || cacheGraph || patchedAttention || characteristic.nonEmpty || aesthetic.nonEmpty || incantations || Set("sonar","semantic_guidance","neutral_prompt")(hook) || latentHook || up.nonEmpty || hiresPass || soft.nonEmpty || inverse || postDiff.nonEmpty then
      lines += s"unet = $unet"
      condition("latent_pos", prompt, negative = false)
      if !patchedInputs then lines += s"pos = predictor(unet, latent_pos$unetArgs)"
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
    else if hook == "photomaker" then
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
    else unsupported(s"Unknown graph hook: $hook")
    if textSchedule.isEmpty && composable.isEmpty && !Set("ip_adapter","revision")(hook) then
      clipIo match
        case Some(session) => session.build("neg", p.negativePrompt, text, enc, textProgram, engine, () => check(), lines)
        case None => condition("latent_neg", p.negativePrompt, negative = true)
      if !patchedInputs then lines += s"neg = predictor(unet, latent_neg$unetArgs)"
    val positive = if composable.nonEmpty then "clora_branch0"
      else if hook == "semantic_guidance" then VulkanSemanticGuidance.prepare(h,text,enc,family=="sdxl",lines) else "pos"
    val negative = if composable.nonEmpty then "clora_neg" else "neg"
    val concat = if inpaint then ", concat = inpaint_condition" else ""
    val control = Option.when(supportsControl)(VulkanConditioning.control(req,cancelled)).flatten
    val positiveInput = if control.nonEmpty then "controlled_pos" else "latent_pos"
    val negativeInput = if control.nonEmpty then "controlled_neg" else "latent_neg"
    val post = modifier.fold("")(VulkanLatentModifier.arguments(_,seed))
    val cfgTiming = if cacheGraph then VulkanGraph.cacheCfgArguments(p.hooks, p.cfgScale, passSteps)
      else VulkanIncantations.graphArguments(p, passSteps)
    val chgArgs = characteristic.fold("")(_.args(p.steps, p.cfgScale))
    val cloraArgs = loraPlan.fold("")(_.guidance("clora"))
    val target = if incantationsPag then "incantations_base" else "guided"
    val guidance = unetHooks.fold(s"$target = cfg($positive, $negative, scale = ${p.cfgScale}$post$cfgTiming$chgArgs$cloraArgs)")(_.guidance(p.cfgScale,concat,positiveInput,post))
    if !patchedInputs then lines += guidance
    val maskGraph = patchedInputs && (inpaint || difference.nonEmpty || p.img2img.exists(_.mask.nonEmpty) ||
      preparedInput.exists(_.mask.nonEmpty))
    // The native image pipeline defaults to the CPU implementation of the CUDA-compatible RNG.
    val randomAlgorithm = if maskGraph || agent.nonEmpty || lllite.nonEmpty || llul.nonEmpty || characteristic.nonEmpty || composable.nonEmpty || aesthetic.nonEmpty || incantations then "cuda" else "cpu"
    lines += s"rng = make_rng(\"$randomAlgorithm\", seed = $seed)"
    val encodeRng = if maskGraph || (characteristic.nonEmpty || composable.nonEmpty || aesthetic.nonEmpty || incantations) && p.isImg2Img then
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
        val init = preparedInput.getOrElse {
          refinementImage.fold(VulkanImageInputs.prepare(i,p.width,p.height,seed,difference, cancelled)) { image =>
            VulkanImageInputs.Inputs(image, None, image, None, None)
          }
        }
        inputs("init") = onCanvas(init.init)
        init.mask.foreach { mask =>
          inputs("paint_mask") = onCanvas(mask)
          overlay = Some(init)
        }
        val posterior = if inverse then ", posterior = \"mode\"" else ""
        lines += s"encoded = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(init), rng = $encodeRng$posterior$tiled))"
        lines += s"base = tensor::broadcast_to(encoded, $shape)"
    if overlay.nonEmpty then
      lines += "mask = tensor::threshold(image::to_f32(tensor::grayscale(paint_mask, axis = 0)), threshold = 0.5)"
    else if inpaint then lines += s"mask = tensor::full([$width,$height,1,1], 1)"
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
    if patchedInputs then
      llul.flatMap(_.mask).foreach { path =>
        inputs("llul_mask_image") = VulkanImageInputs.read(VulkanConditioning.file(path, "LLuL mask"))
        lines += "llul_mask = image::to_f32(tensor::grayscale(llul_mask_image, axis = 0))"
      }
      reference.foreach { config =>
        val file = VulkanConditioning.file(config.image, "reference image")
        if config.weight > 0 then
          val image = VulkanImageInputs.read(file)
          inputs("ref_image") = VulkanImageInputs.resize(image, width, height, 1)
          if hiresPass then
            inputs("hr_ref_image") = VulkanImageInputs.resize(image, hiresWidth, hiresHeight, 1)
          lines ++= config.prepare("ref", tiled)
      }
      lllite.foreach { (config, _) =>
        val source = config.image.map(VulkanImageInputs.read)
          .orElse(inputs.get("init"))
          .getOrElse(VulkanImageInputs.read(config.source(p.img2img.map(_.initImage))))
        inputs("lllite_image") = VulkanImageInputs.resize(source, width, height, 1)
        if hiresPass && config.image.nonEmpty then
          inputs("hr_lllite_image") = VulkanImageInputs.resize(source, hiresWidth, hiresHeight, 1)
        lines += "lllite_hint = image::to_f32(lllite_image, channels = 3)"
      }
      cdTuner.foreach(tuner => lines ++= tuner.prepareMap("cdt_map"))
      fooocus.foreach { assets =>
        if overlay.isEmpty then invalid("Fooocus Inpaint did not produce a prepared mask")
        lines ++= assets.prepareInput(width, height, encodeRng, tiled)
      }
      control.foreach { (modelPath, image, weight) =>
        inputs("control_image") = onCanvas(image)
        lines += s"control_model = controlnet::load(${quoted(modelPath)}, $common)"
        lines += "control_hint = image::to_f32(control_image)"
        lines += s"controlled_pos = controlnet::apply(latent_pos, control_model, control_hint, strength = $weight)"
        lines += s"controlled_neg = controlnet::apply(latent_neg, control_model, control_hint, strength = $weight)"
      }
      textSchedule match
        case Some(schedule) => schedule.predictors(concat,lines)
        case None if loraPlan.nonEmpty =>
          lines ++= VulkanComposableLora.prepare(loraPlan.get, req, "clora", unet, text, enc, concat, supportsBlank = true,
            tracePass = composable.filter(VulkanComposableLora.plotRequested).map(_ => "base"),
            runtimeWeights = runtimeLoraWeights)
        case None =>
          val fooocusArgs = fooocus.fold("")(_.predictorArgs)
          def cdArgs(negative: Boolean): String =
            cdTuner.fold("")(t => t.args(t.base, "cdt_map", negative, colorShift = !hiresPass, high = false))
          val posRef = reference.fold("")(_.args("ref", positive = true))
          val negRef = reference.fold("")(_.args("ref", positive = false))
          val llliteArgs = lllite.fold("")((config, path) => config.args(path, "lllite_hint"))
          val llulArgs = llul.filter(_.canApply(p.steps)).fold("")(_.args(p.width, p.height, "llul_mask", p.steps))
          lines += s"pos = predictor(unet, $positiveInput$concat$unetArgs$fooocusArgs${cdArgs(false)}$posRef$llliteArgs$llulArgs$scfgArgs${t2iArgs("pos")})"
          val blank = if aesthetic.fold(p.negativePrompt.nonEmpty)(_ => p.negativePrompt.trim.nonEmpty) then ""
            else if lllite.nonEmpty then ", lllite_blank_negative = true"
            else if llul.nonEmpty then ", llul_blank_negative = true"
            else if characteristic.nonEmpty then ", sdxl_blank_negative = true"
            else if aesthetic.nonEmpty || incantations then ", sdxl_blank_negative = true"
            else ""
          lines += s"neg = predictor(unet, $negativeInput$concat$unetArgs$fooocusArgs${cdArgs(true)}$negRef$llliteArgs$llulArgs$blank$segArgs${t2iArgs("neg")})"
      lines += guidance
      if incantationsPag then
        lines ++= VulkanIncantations.pagGraph(p.hooks, passSteps, unet, positiveInput, concat, positive, target, "guided", t2iArgs("perturbed"))
    val fullSteps = up.fold(p.steps)(_.fullSteps)
    val scheduleName = up.fold(schedule)(_.scheduler)
    val schedulePredictor = if composable.nonEmpty then "clora_schedule" else "guided"
    lines += s"full = make_sigmas($schedulePredictor, latent = base, scheduler = ${quoted(VulkanProtocol.schedulers(scheduleName))}, steps = $fullSteps)"
    val sigmas = up.fold(p.img2img.fold("full")(i => s"full[${p.steps - math.min(p.steps,(p.steps*i.denoisingStrength).toInt+1)}:]"))(_ => s"full[${fullSteps-p.steps}:]")
    val softSteps = soft.fold(0)(_ => (p.steps*p.img2img.get.denoisingStrength).toInt.max(1).min(p.steps))
    lines += s"sigmas = ${if inverse then "full" else if soft.nonEmpty then s"full[${p.steps-softSteps}:]" else sigmas}"
    val baseNoise = tweak.filter(_ => p.subseedStrength>0).flatMap { pass =>
      pass.copy(seed = seed,subseed = pass.baseSubseed,strength = p.subseedStrength,width = 0,height = 0).noise(engine,width/8,height/8)
    }
    if inverse then lines ++= VulkanInverse.start(h,p,prompt,text,enc)
    else if baseNoise.nonEmpty then lines += s"start = add_noise($schedulePredictor, base, base_noise, sigma = sigmas[0])"
    else lines += s"start = add_noise($schedulePredictor, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])"
    val eta = p.eta.fold("")(e => s", eta = $e")
    if soft.nonEmpty then lines ++= soft.get.sample(softSteps,width,height,VulkanProtocol.samplers(p.samplerName),eta,seed,tiled)
    else if postDiff.nonEmpty then lines ++= postDiff.get.sample
    else if sonar.nonEmpty then lines ++= sonar.get.sample(inputs.contains("sonar_ref"),tiled)
    else if hook == "neutral_prompt" then lines ++= VulkanNeutralPrompt.sample(p,prompt,h,text,enc)
    else if latentHook then lines ++= VulkanLatentGraph.sample(p,VulkanProtocol.samplers(p.samplerName),eta,family == "sdxl",seed)
    else
      val mask = if inpaint then ", base = base, denoise_mask = denoise_mask"
        else if overlay.nonEmpty then ", base = base, denoise_mask = latent_mask" else ""
      lines += s"latent = sample(start, sigmas, predict = guided, method = ${quoted(VulkanProtocol.samplers(p.samplerName))}, rng = rng$eta$mask)"
    val noise = tweak.flatMap(_.noise(engine, hiresWidth/8, hiresHeight/8))
    if hiresPass then
      val controlWeight = control.flatMap { _ =>
        VulkanConditioning.control(req.copy(params = p.copy(width = hiresWidth, height = hiresHeight)), cancelled).map {
          case (_, image, weight) =>
            inputs("hr_control_image") = image
            weight
        }
      }
      lines ++= VulkanHiresGraph.refinement(req,prompt,loras,tweak.fold(seed)(_.seed),unet,text,eta,device,
        noise.nonEmpty,tiled,cdTuner,cacheArgs,reference,agent,controlWeight,inpaint,lllite,llul,characteristic,composable,
        runtimeLoraWeights = runtimeLoraWeights, aesthetic = aestheticConditioning, masked = overlay.nonEmpty)
    val resultLatent = if hiresPass then "hr_latent" else "latent"
    val saturation = cdTuner.filter(_ => refinementImage.isEmpty).fold("")(_.decoderArgs)
    lines += s"return {output = vae_decode(vae, unet_to_vae_latents(vae, $resultLatent)$tiled$saturation)}"
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
      val decoded = engine.dglImage(source, inputs.asJava, "output", noises.asJava)
      val (outputWidth, outputHeight) = if hiresPass then hiresSize else baseSize
      val canvasChanged = if hiresPass then (hiresWidth, hiresHeight) != hiresSize else (width, height) != baseSize
      val image = if canvasChanged then VulkanImageInputs.resize(decoded, outputWidth, outputHeight, 0) else decoded
      Seq(soft.fold(overlay.fold(image)(_.finish(image)))(_.composite(overlay.get,image,cancelled)))
