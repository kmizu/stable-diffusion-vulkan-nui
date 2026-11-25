package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.protocol.{GenerationRequest, LoadedModel, LoadModelRequest}
import ujson.Value

/** Builds the motion/identity model graphs using the pinned library's DGL operations. */
private[vulkan] object VulkanGraph:
  val hooks = Set("animatediff", "photomaker", "ip_adapter") ++ VulkanLatentGraph.hooks
  private def quoted(s: String): String = ujson.write(ujson.Str(s))
  private def path(s: String): String = quoted(VulkanConditioning.file(s, "DGL model"))

  def render(engine: VulkanInference, device: String, model: LoadedModel, settings: LoadModelRequest,
      req: GenerationRequest, hook: String, prompt: String, loras: Seq[LoraRef], seed: Long, schedule: String): Seq[BufferedImage] =
    import VulkanProtocol.{invalid, unsupported}
    val p = req.params
    val h = p.hooks(hook)
    val latentHook = VulkanLatentGraph.hooks(hook)
    val permitted = Set(hook,"model_keyword","pnginfo_injection") ++ (if latentHook then VulkanLatentGraph.hooks else Set.empty[String])
    val other = p.hooks.filterNot((_, v) => v.obj.get("enabled").contains(ujson.Bool(false))).keySet -- permitted
    if other.nonEmpty || p.hires.enabled || p.controlnet.exists(_.enabled) || VulkanConditioning.activeEmbeddings(req).nonEmpty then
      unsupported(s"$hook with other sampling hooks, hires, embeddings or ControlNet")
    if p.img2img.exists(_.mask.nonEmpty) then unsupported(s"masked $hook generation")
    val isVideo = hook == "animatediff"
    if isVideo && model.kind != "sd15" then unsupported("AnimateDiff motion models outside SD1.x")
    if hook == "photomaker" && (model.kind != "sdxl" || p.isImg2Img) then unsupported("PhotoMaker outside SDXL txt2img")
    val family = if model.kind == "sdxl" then "sdxl" else "sd1"
    val options = VulkanConditioning.options(settings, device)
    val placement = if options.placement().isEmpty then "" else s", params_backend = ${quoted(options.placement())}"
    val common = s"backend = ${quoted(device)}$placement, wtype = ${quoted(options.weights())}, family = ${quoted(family)}"
    val lines = Vector.newBuilder[String]
    lines += s"unet0 = load_unet(${path(model.path)}, $common)"
    lines += s"text0 = load_text_encoder(${path(model.path)}, $common)"
    val scale = if family == "sdxl" && model.vaePath.isEmpty then ", conv_scale = 0.03125" else ""
    lines += s"vae = load_vae(${path(model.vaePath.getOrElse(model.path))}, $common$scale)"
    for (l,i) <- loras.zipWithIndex do
      if l.unet.exists(_ != l.weight) || l.lbw.nonEmpty || l.dyn.nonEmpty || l.start.nonEmpty || l.stop.nonEmpty then unsupported("scheduled LoRA in a motion/identity graph")
      lines += s"lora$i = load_lora(${path(req.loraPaths.getOrElse(l.name, invalid(s"Unresolved LoRA ${l.name}")))})"
      lines += s"unet${i+1} = apply_lora(unet$i, lora$i, weight = ${l.weight})"
      lines += s"text${i+1} = apply_lora(text$i, lora$i, weight = ${l.weight})"
    val unet = s"unet${loras.size}"
    val text = s"text${loras.size}"
    lazy val adapter = path(h("model").str)
    val inputs = scala.collection.mutable.LinkedHashMap.empty[String,BufferedImage]
    val size = s"width = ${p.width}, height = ${p.height}"
    val enc = s"$size, clip_skip = ${p.clipSkip}"
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
    else if latentHook then
      lines += s"unet = $unet"
      lines += s"pos = predictor(unet, encode_prompt($text, ${quoted(prompt)}, $enc))"
      1
    else if hook == "ip_adapter" then
      val rawPath = Path.of(h("model").str).toAbsolutePath.normalize()
      val settings = sdfx.vulkan.tools.VulkanVision.inspect(rawPath,h.obj.get("image_encoder").fold("")(_.str),family)
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
    if hook != "ip_adapter" then lines += s"neg = predictor(unet, encode_prompt($text, ${quoted(p.negativePrompt)}, $enc))"
    lines += s"guided = cfg(pos, neg, scale = ${p.cfgScale})"
    lines += s"rng = make_rng(\"cpu\", seed = $seed)"
    val shape = s"latent_shape(vae, $size, batch = $frames)"
    p.img2img match
      case None => lines += s"base = tensor::zeros($shape)"
      case Some(i) =>
        inputs("init") = VulkanImageInputs.resize(VulkanImageInputs.read(i.initImage),p.width,p.height,i.resizeMode)
        lines += "encoded = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(init), rng = rng))"
        lines += s"base = tensor::broadcast_to(encoded, $shape)"
    lines += s"full = make_sigmas(guided, latent = base, scheduler = ${quoted(VulkanProtocol.schedulers(schedule))}, steps = ${p.steps})"
    val sigmas = p.img2img.fold("full")(i => s"full[${p.steps - math.min(p.steps,(p.steps*i.denoisingStrength).toInt+1)}:]")
    lines += s"sigmas = $sigmas"
    lines += "start = add_noise(guided, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])"
    val eta = p.eta.fold("")(e => s", eta = $e")
    if latentHook then lines ++= VulkanLatentGraph.sample(p,VulkanProtocol.samplers(p.samplerName),eta)
    else lines += s"latent = sample(start, sigmas, predict = guided, method = ${quoted(VulkanProtocol.samplers(p.samplerName))}, rng = rng$eta)"
    val tiling = if settings.vaeTiling then ", tiling = tiling(tile_size_w = 32, tile_size_h = 32)" else ""
    lines += s"return {output = vae_decode(vae, unet_to_vae_latents(vae, latent)$tiling)}"
    val source = s"fn main(${inputs.keys.mkString(",")}) {\n${lines.result().mkString("\n")}\n}"
    if isVideo then engine.dglFrames(source, inputs.asJava, "output").asScala.toVector
    else Seq(engine.dglImage(source, inputs.asJava, "output"))
