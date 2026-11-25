package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.Locale
import scala.jdk.CollectionConverters.*
import sdfx.core.protocol.{GenerationRequest, LoadModelRequest}

private[vulkan] object VulkanConditioning:
  import VulkanProtocol.{invalid, unsupported}

  def file(path: String, kind: String): String =
    val p = Path.of(path).toAbsolutePath.normalize()
    if !Files.isRegularFile(p) then invalid(s"Missing $kind file: $p")
    p.toString

  def stem(path: String): String = Path.of(path).getFileName.toString.replaceFirst("\\.[^.]+$", "")

  def activeEmbeddings(req: GenerationRequest): Seq[String] =
    val prompts = Seq(req.params.prompt, req.params.negativePrompt)
    req.embeddingPaths.filter { path =>
      val escaped = java.util.regex.Pattern.quote(stem(path))
      val token = java.util.regex.Pattern.compile(s"(?iu)(?<![\\p{L}\\p{N}_])$escaped(?![\\p{L}\\p{N}_])")
      prompts.exists(text => token.matcher(text).find())
    }

  def options(req: LoadModelRequest, device: String, control: String = "", embeddings: Seq[String] = Nil,
      hooks: Map[String,ujson.Value] = Map.empty): VulkanInference.ModelOptions =
    val sampling = activeHook(hooks, "model_sampling")
    val prediction = sampling.flatMap(_.obj.get("sampling")).map(_.str).getOrElse("default") match
      case "default" => ""
      case "eps" => "eps"
      case "v_prediction" => "v"
      case name => unsupported(s"prediction $name")
    if sampling.flatMap(_.obj.get("zsnr")).exists(_.bool) then unsupported("zero-terminal-SNR schedule")
    val offload = activeHook(hooks, "neveroom").flatMap(_.obj.get("unet_offload")).exists(_.bool)
    val weights = req.dtype match
      case "fp16" | "float16" => "f16"
      case "fp32" | "float32" => "f32"
      case "bf16" | "bfloat16" => "bf16"
      case other => invalid(s"Unknown precision: $other")
    val placement = req.vramMode match
      case "auto" => "" // Native auto-fit chooses residency from device memory budgets.
      case "gpu" => device
      case "model_offload" | "medvram" => "cpu"
      case "sequential_offload" | "lowvram" | "cpu" => "disk"
      case other => invalid(s"Unknown VRAM mode: $other")
    val rules = Seq(Option.when(req.fp8Storage)("^model\\.diffusion_model\\.=f8_e4m3"),
      Option.when(req.vaeFp16)("^(first_stage_model|vae)\\.=f16")).flatten.mkString(",")
    val entries = embeddings.distinct.sorted.map { path =>
      new VulkanInference.Embedding(stem(path), file(path, "embedding"))
    }
    entries.groupBy(_.name().toLowerCase(Locale.ROOT)).find(_._2.size > 1).foreach {
      (name, _) => invalid(s"Ambiguous embedding name: $name")
    }
    val residency = if !offload || Set("cpu", "disk")(placement) then placement
      else Seq(placement, "diffusion=cpu").filter(_.nonEmpty).mkString(",")
    new VulkanInference.ModelOptions(weights, residency, req.vaePath.map(file(_, "VAE")).getOrElse(""),
      rules, control, entries.asJava, prediction)

  private def activeHook(hooks: Map[String,ujson.Value], key: String): Option[ujson.Value] =
    hooks.get(key).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))

  private[vulkan] def sampleArgs(p: sdfx.core.model.GenerationParams): String =
    val args = Vector.newBuilder[String]
    activeHook(p.hooks,"cfg_schedule").filter(_ => p.cfgScale > 1.0).foreach { o =>
      def num(k: String, default: Double) = o.obj.get(k).fold(default)(_.num)
      val schedule = sdfx.vulkan.tools.CfgSchedule(num("start_cfg",7),num("end_cfg",3),
        o.obj.get("curve").fold("linear")(_.str),num("start",0),num("end",1),num("active_start",0))
      val count = p.steps
      val values = (0 until count).map(i => schedule.at(i.toDouble / (count-1).max(1)))
      args += "guidance_schedule=" + values.map(scale => s"${scale}x1").mkString("+")
    }
    activeHook(p.hooks,"apg").foreach { o =>
      def num(k: String, default: Double) = o.obj.get(k).fold(default)(_.num)
      val momentum = num("momentum",0.5)
      if (momentum != 0 && num("adaptive_momentum",0.18) != 0) ||
          o.obj.get("guidance_limiter").exists(_.bool) then unsupported("APG adaptive momentum and sigma limiter")
      val eta = num("eta",1)
      val threshold = num("norm_threshold",15)
      require(Seq(momentum,eta,threshold).forall(_.isFinite) && threshold >= 0, "Invalid APG settings")
      args ++= Seq(s"apg_momentum=$momentum",s"apg_eta=$eta",s"apg_norm_threshold=$threshold")
    }
    args.result().mkString(",")

  def prepare(req: GenerationRequest, load: LoadModelRequest, cancelled: () => Boolean = () => false): (String, VulkanInference.Conditioning) =
    val p = req.params
    if p.hires.enabled && !VulkanImageInputs.separateHires(req) && req.hiresLoras.exists(_ != p.loras) then unsupported("different LoRAs for the hires pass")
    val loras = p.loras.map { l =>
      if !l.weight.toFloat.isFinite then invalid(s"Invalid LoRA weight: ${l.name}")
      if l.unet.exists(_ != l.weight) || l.lbw.nonEmpty || l.start.nonEmpty || l.stop.nonEmpty || l.dyn.nonEmpty then
        unsupported(s"separate text/U-Net, block, timing or dynamic LoRA settings for ${l.name}")
      new VulkanInference.Lora(file(req.loraPaths.getOrElse(l.name, invalid(s"Unresolved LoRA: ${l.name}")), "LoRA"), l.weight.toFloat)
    }
    val units = p.controlnet.filter(_.enabled)
    if units.size > 1 then unsupported("multiple simultaneous ControlNet units")
    val control = units.headOption.map { u =>
      if u.guidanceStart != 0 || u.guidanceEnd != 1 || u.controlMode != "Balanced" then
        unsupported("ControlNet guidance intervals and unbalanced control modes")
      if !u.weight.toFloat.isFinite then invalid("Invalid ControlNet weight")
      val model = file(req.controlnetPaths.getOrElse(u.model, invalid(s"Unresolved ControlNet: ${u.model}")), "ControlNet")
      val source = u.image.orElse(p.img2img.map(_.initImage)).getOrElse(invalid("ControlNet requires a condition image"))
      val image = sdfx.vulkan.tools.VulkanTools.controlImage(u, source, p.width, p.height, cancelled)
      (model, image, u.weight)
    }
    val tiling = activeHook(p.hooks,"tiling")
    def tile(key: String) = tiling.exists(_.obj.get(key).forall(_.bool))
    tiling.foreach { options =>
      val start = options.obj.get("start_step").fold(0)(_.num.toInt)
      val stop = options.obj.get("stop_step").fold(-1)(_.num.toInt)
      if start > 0 || (stop >= 0 && stop < p.steps) then unsupported("step-limited circular padding")
    }
    val tiledVae = activeHook(p.hooks,"neveroom").flatMap(_.obj.get("vae_tiling")).exists(_.bool)
    (control.fold("")(_._1), new VulkanInference.Conditioning(loras.asJava, control.map(_._2).orNull,
      control.fold(1.0)(_._3), load.vaeTiling || tiledVae, load.vaeAuto, tile("tile_x"), tile("tile_y"),sampleArgs(p)))
