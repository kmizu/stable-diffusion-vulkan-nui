package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import sdfx.core.infotext.HardwareInfo
import sdfx.core.protocol.*
import sdfx.vulkan.VulkanProtocol
import ujson.{Obj, Value}

private[tools] final class NativeBackendError(val code: String, message: String) extends RuntimeException(message)

private[tools] final class NativeWorkContext(
    val cancelled: () => Boolean,
    val beginPublication: () => Unit,
    val event: (String, Map[String, NativeJson]) => Unit):
  def checkCancelled(): Unit = if cancelled() then throw CancellationException()

private[vulkan] object VulkanTools:
  final class Session:
    private var embeddingToken = Option.empty[String]
    private var openedEmbeddings = Vector.empty[String]
    def embeddingPaths(model: LoadedModel): Vector[String] =
      if model.modelToken != embeddingToken then
        embeddingToken = model.modelToken
        openedEmbeddings = Vector.empty
      openedEmbeddings
    private[tools] def opened(model: LoadedModel,path: String): Unit =
      openedEmbeddings = (embeddingPaths(model) :+ path).distinct
    private[tools] var pendingSubmission = Option.empty[(Long,Map[String,NativeJson])]
    private[tools] var keywordState = ModelKeyword.State()
  val commands = Set("system_submission_preview", "system_submit", "embedding_inspector", "embedding_editor", "embedding_merge", "animatediff_encode", "model_keyword", "model_keyword_plan", "tokenize", "merge_models", "convert_model", "list_files", "preprocess", "image_tool", "metadata_utils", "inpaint_difference_preview", "upscale")
  val features = Vector("clip_interrogator","embedding_inspector", "embedding_editor", "embedding_merge", "ip_adapter", "latent_mirroring", "resharpen", "anti_burn", "save_intermediate", "steps_animation", "apg", "cfg_schedule", "tiling", "neveroom", "model_sampling", "model_keyword", "inpaint_difference", "pnginfo_injection", "controlnet_preprocess_extras", "metadata_utils", "loopback_scaler")
  val preprocessors = NativePreprocess.valid
  def differenceMask(raw: Value, source: BufferedImage, cancelled: () => Boolean): BufferedImage =
    val options = NativeInpaintDifference.parse(NativeJson.parse(ujson.write(raw)))
    val base = DiffusionImageInput.readRgb(Path.of(options.basePath), maxBytes = 200000000L)
    val bytes = NativeInpaintDifference.mask(base, source, options.threshold, options.dilation, options.erosion,
      cancelled, options.contours)
    val mask = new BufferedImage(source.getWidth, source.getHeight, BufferedImage.TYPE_INT_RGB)
    mask.setRGB(0, 0, mask.getWidth, mask.getHeight, bytes.map(_ * 0x010101), 0, mask.getWidth)
    mask

  def controlImage(unit: sdfx.core.model.ControlNetUnit, path: String, width: Int, height: Int,
      cancelled: () => Boolean): BufferedImage =
    val source = DiffusionImageInput.readRgb(Path.of(path))
    val name = if unit.preprocessor.isEmpty then "none" else unit.preprocessor
    if !preprocessors.contains(name) then VulkanProtocol.unsupported(s"ControlNet preprocessor $name")
    require(unit.resizeMode >= 0 && unit.resizeMode <= 2, "invalid ControlNet resize mode")
    val scaleX = width.toDouble / source.getWidth
    val scaleY = height.toDouble / source.getHeight
    val scale = if unit.resizeMode == 2 then scaleX.min(scaleY) else scaleX.max(scaleY)
    val res = if unit.pixelPerfect then math.rint(math.min(source.getWidth, source.getHeight) * scale).toInt.max(1)
      else if unit.processorRes >= 64 && unit.processorRes <= 2048 then unit.processorRes else 512
    val detectAtResolution = unit.pixelPerfect || unit.sourceCompatibleResize || unit.processorRes != -1
    val args = unit.preprocessorArgs.map((k,v) => k -> NativeJson.Number(BigDecimal(v)))
    val map = NativeControlNet.preprocessImage(name, source, args, detectAtResolution, res, cancelled)
    if unit.sourceCompatibleResize then NativeControlNetImageOps.sourceCompatibleResize(map,width,height,unit.resizeMode,cancelled)
    else DiffusionImageResize(map,width,height,unit.resizeMode)

  type ModelUpscale = (Path, BufferedImage, Int, Int) => BufferedImage

  def execute(kind: String, raw: Value, cancelled: () => Boolean, publish: () => Unit,
      event: (String, Value) => Unit, device: String, modelKind: Option[String], modelUpscale: ModelUpscale, session: Session = new Session,
      loadedModel: Option[LoadedModel] = None): Value =
    import NativeJson.*
    val context = new NativeWorkContext(cancelled, publish,
      (name, fields) => event(name, ujson.read(NativeJson.render(NativeJson.Obj(fields)))))
    val params = NativeJson.parse(ujson.write(raw)).asInstanceOf[NativeJson.Obj].fields
    try
      val value = kind match
        case "embedding_inspector" | "embedding_editor" | "embedding_merge" =>
          val model = loadedModel.getOrElse(throw NativeBackendError("no_model","Load a checkpoint first"))
          val result = VulkanEmbeddings.execute(kind,params,model,context,session.embeddingPaths(model))
          if kind == "embedding_editor" && params.get("operation").contains(Str("open")) then
            session.opened(model,result("path").asInstanceOf[Str].value)
          result
        case "model_keyword" | "model_keyword_plan" =>
          val pathKey = if kind == "model_keyword" then "model_path" else "checkpoint"
          val file = raw.obj.get(pathKey).flatMap(_.strOpt).map(Path.of(_)).getOrElse(Path.of("."))
          if raw.obj.get(pathKey).isDefined && Files.isDirectory(file) then
            VulkanProtocol.unsupported("keyword mappings for directory-format checkpoints")
          def directoryHash(): String = VulkanProtocol.unsupported("directory-format checkpoints")
          if kind == "model_keyword" then NativeKeywordMappings.execute(params, file, directoryHash, context)
          else
            val (value, next) = NativeKeywordMappings.planRequest(params, file, directoryHash, session.keywordState, context)
            session.keywordState = next
            value
        case "animatediff_encode" => return VulkanVideo.encode(raw, context)
        case "system_submission_preview" =>
          val preview = VulkanSubmission.submissionPreview(params)
          session.pendingSubmission = Some(System.nanoTime() -> preview)
          preview
        case "system_submit" =>
          val pending = session.pendingSubmission
          session.pendingSubmission = None
          context.beginPublication()
          VulkanSubmission.submit(params, pending)
        case "tokenize" => NativeTokenizerInspection.inspect(params, modelKind, context)
        case "list_files" =>
          val dirs = VulkanProtocol.decode(raw.obj.get("dirs").toVector.flatMap(_.arr.map(_.str)))
          val category = VulkanProtocol.decode(raw("kind").str)
          Map("files" -> Arr(NativeFileCatalog.scan(category, dirs, () => context.checkCancelled())))
        case "preprocess" => NativePreprocess.execute(params, context)
        case "merge_models" => NativeModelMerge.execute(params, context)
        case "convert_model" => NativeModelConvert.execute(params, context)
        case "metadata_utils" => NativeMetadataUtils.execute(params, context)
        case "image_tool" => NativeCasImageTool.execute(params, context)
        case "inpaint_difference_preview" => NativeInpaintDifferencePreview.execute(params, context)
        case "upscale" => return upscale(raw, context, device, modelUpscale)
        case other => VulkanProtocol.unsupported(other)
      ujson.read(NativeJson.render(NativeJson.Obj(value)))
    catch case e: NativeBackendError => throw EngineError(e.code, e.getMessage)

  def upscaleImage(name: String, paths: Map[String,String], image: BufferedImage, width: Int, height: Int,
      modelUpscale: ModelUpscale): BufferedImage =
    require(width > 0 && height > 0 && width <= 16384 && height <= 16384 && width.toLong * height <= 16777216L, "upscale output exceeds pixel limit")
    name match
      case "Lanczos" | "None" => DiffusionImageResize.lanczos(image, width, height)
      case "Nearest" =>
        val result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val g = result.createGraphics()
        try
          g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR)
          g.drawImage(image, 0, 0, width, height, null)
        finally g.dispose()
        result
      case name =>
        val path = paths.get(name).map(Path.of(_)).filter(Files.isRegularFile(_))
          .getOrElse(VulkanProtocol.unsupported(s"upscaler $name without compatible model weights"))
        modelUpscale(path, image, width, height)

  private def upscale(raw: Value, context: NativeWorkContext, device: String, modelUpscale: ModelUpscale): Value =
    val req = VulkanProtocol.decode(Json.fromJs[UpscaleRequest](raw))
    VulkanProtocol.checkFields(raw, Json.toJs(req))
    require(req.scale.isFinite && req.scale > 0 && req.scale <= 16, "scale must be in (0, 16]")
    require(req.imagePath.nonEmpty && req.outputDir.nonEmpty, "image_path and output_dir are required")
    val start = System.nanoTime()
    val image = DiffusionImageInput.readRgb(Path.of(req.imagePath))
    val width = math.max(1, math.rint(image.getWidth * req.scale).toInt)
    val height = math.max(1, math.rint(image.getHeight * req.scale).toInt)
    require(width <= 16384 && height <= 16384 && width.toLong * height <= 16777216L, "upscale output exceeds pixel limit")
    val resized = upscaleImage(req.upscaler, req.upscalerPaths, image, width, height, modelUpscale)
    context.checkCancelled()
    val dir = Files.createDirectories(Path.of(req.outputDir).toAbsolutePath.normalize())
    val file = Files.createTempFile(dir, "upscale-", ".png")
    try
      require(ImageIO.write(resized, "png", file.toFile), "PNG writer unavailable")
      context.beginPublication()
      val hardware = Option.when(req.hardwareInfo.enabled)(HardwareInfo.Record(device,
        f"${(System.nanoTime()-start)/1e9}%.2fs", Option(req.hardwareInfo.remark).filter(_.nonEmpty)))
      Json.toJs(ImagePathResult(file.toString, hardware))
    catch case e: Throwable =>
      Files.deleteIfExists(file)
      throw e
