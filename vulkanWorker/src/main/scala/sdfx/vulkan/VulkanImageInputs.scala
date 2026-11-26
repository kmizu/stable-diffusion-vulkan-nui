package sdfx.vulkan

import java.awt.Rectangle
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import sdfx.vulkan.tools.{DiffusionImageInput, DiffusionImageResize, NativeInpaint}
import sdfx.core.model.Img2ImgParams
import sdfx.core.protocol.GenerationRequest
import sdfx.vulkan.VulkanProtocol.{invalid, unsupported}

private[vulkan] object VulkanImageInputs:
  def img2imgSteps(steps: Int, strength: Double): Int =
    math.min(steps, (steps * strength).toInt + 1)

  val upscalers = Map("Latent" -> "latent", "Latent (antialiased)" -> "latent_antialiased",
    "Latent (bicubic)" -> "latent_bicubic", "Latent (bicubic antialiased)" -> "latent_bicubic_antialiased",
    "Latent (nearest)" -> "latent_nearest", "Latent (nearest-exact)" -> "latent_nearest_exact",
    "Nearest" -> "nearest", "Lanczos" -> "lanczos")

  def separateHires(req: GenerationRequest): Boolean =
    val p = req.params
    val h = p.hires
    h.enabled && (
      (h.prompt.nonEmpty && h.prompt != p.prompt) || (h.negativePrompt.nonEmpty && h.negativePrompt != p.negativePrompt) ||
      h.cfgScale.exists(_ != p.cfgScale) || h.samplerName.exists(_ != p.samplerName) ||
      h.scheduler.exists(_ != p.scheduler) || req.hiresLoras.exists(_ != p.loras))

  def hires(req: GenerationRequest): Option[VulkanInference.Hires] = Option.when(req.params.hires.enabled) {
    val p = req.params
    val h = p.hires
    if !h.scale.isFinite || h.scale <= 0 || h.resizeX < 0 || h.resizeY < 0 ||
        h.resizeX > 65536 || h.resizeY > 65536 || h.secondPassSteps < 0 || h.secondPassSteps > 150 then
      invalid("Invalid hires scale, size or step count")
    val effectiveSteps = if h.secondPassSteps == 0 then p.steps else h.secondPassSteps
    val zero = h.denoisingStrength <= 0
    val strength = if zero then 1.0 / effectiveSteps else h.denoisingStrength
    if !h.denoisingStrength.isFinite || h.denoisingStrength < 0 || h.denoisingStrength > 1 ||
      h.cfgScale.exists(v => !v.isFinite || v < 0) then invalid("Invalid hires denoising strength or CFG scale")
    val (w, ht) = h.targetSize(p.width,p.height)
    if w.toLong * ht > 64L * 1024 * 1024 then invalid("Hires target exceeds the pixel limit")
    val native = upscalers.getOrElse(h.upscaler, if req.upscalerPaths.contains(h.upscaler) then "model" else unsupported(h.upscaler))
    val path = Option.when(native == "model") {
      val file = Path.of(req.upscalerPaths(h.upscaler))
      if !Files.isRegularFile(file) then invalid(s"Missing upscaler: $file")
      file.toAbsolutePath.toString
    }
    new VulkanInference.Hires(native, path.orNull, w, ht, if zero then 1 else h.secondPassSteps, strength)
  }

  def read(path: String): BufferedImage = DiffusionImageInput.readRgb(Path.of(path))

  def resize(image: BufferedImage, width: Int, height: Int, mode: Int): BufferedImage =
    DiffusionImageResize(image, width, height, mode)

  private def transform(width: Int, height: Int)(color: (Int,Int) => Int): BufferedImage =
    val out = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until height do
      val row = Array.tabulate(width)(x => color(x,y))
      out.setRGB(0,y,width,1,row,0,width)
    out

  private def mix(a: Int,b: Int,alpha: Int): Int =
    def channel(shift: Int) = ((((a >> shift) & 255) * (255-alpha) + ((b >> shift) & 255) * alpha + 127) / 255) << shift
    channel(16) | channel(8) | channel(0)

  private def grayMask(image: BufferedImage, invert: Boolean): BufferedImage =
    transform(image.getWidth,image.getHeight) { (x,y) =>
      val rgb = image.getRGB(x,y)
      val value = (299*((rgb >> 16)&255) + 587*((rgb >> 8)&255) + 114*(rgb&255) + 500)/1000
      0x010101 * (if invert then 255 - value else value)
    }

  private def blur(mask: BufferedImage, radius: Int, cancelled: () => Boolean): BufferedImage =
    if radius == 0 then return mask
    val w = mask.getWidth
    val h = mask.getHeight
    val samples = mask.getRGB(0,0,w,h,null,0,w).map(_ & 255)
    val blurred = NativeInpaint.gaussianBytes(samples,w,h,radius,cancelled)
    transform(w,h)((x,y) => blurred(y*w+x) * 0x010101)

  private def crop(mask: BufferedImage,w: Int,h: Int,pad: Int): Option[Rectangle] =
    val bounds = new Rectangle()
    var hasPixels = false
    for y <- 0 until mask.getHeight; x <- 0 until mask.getWidth if (mask.getRGB(x,y)&255) != 0 do
      if !hasPixels then bounds.setBounds(x,y,1,1)
      else
        bounds.add(x,y)
        bounds.add(x+1,y+1)
      hasPixels = true
    Option.when(hasPixels) {
      bounds.grow(pad,pad)
      val b = bounds.intersection(new Rectangle(0,0,mask.getWidth,mask.getHeight))
      val rw = math.max(b.width,b.height.toDouble*w/h).toInt.min(mask.getWidth)
      val rh = math.max(b.height,b.width.toDouble*h/w).toInt.min(mask.getHeight)
      new Rectangle((b.x-(rw-b.width)/2).max(0).min(mask.getWidth-rw),
        (b.y-(rh-b.height)/2).max(0).min(mask.getHeight-rh),rw,rh)
    }

  case class Inputs(init: BufferedImage, mask: Option[BufferedImage], original: BufferedImage,
                    overlay: Option[BufferedImage], region: Option[Rectangle]):
    def finish(image: BufferedImage): BufferedImage = overlay match
      case None => image
      case Some(alpha) =>
        val w = if region.isDefined then original.getWidth else image.getWidth
        val h = if region.isDefined then original.getHeight else image.getHeight
        val base = resize(original,w,h,0)
        val weights = resize(alpha,w,h,0)
        val box = region.getOrElse(new Rectangle(0,0,w,h))
        val patch = resize(image,box.width,box.height,0)
        transform(w,h) { (x,y) =>
          val color = base.getRGB(x,y)
          if box.contains(x,y) then mix(color,patch.getRGB(x-box.x,y-box.y),weights.getRGB(x,y)&255) else color
        }

  def maskConditioning(config: Option[ujson.Value], strength: Double): Double =
    val mode = config.filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
      .flatMap(_.obj.get("mode")).fold("disabled")(_.str)
    mode match
      case "disabled" | "empty" => 1.0
      case "lowest" => 1 - strength / 4
      case "low" => 1 - strength / 2
      case "medium" => .5
      case "high" => strength / 2
      case "highest" => strength / 4
      case "full" => 0.0
      case other => invalid(s"Unknown mask conditioning mode $other")

  def prepare(ip: Img2ImgParams, w: Int,h: Int,seed: Long,
      difference: Option[(BufferedImage, BufferedImage)] = None, cancelled: () => Boolean = () => false): Inputs =
    if ip.maskBlur < 0 || ip.maskBlur > 64 || ip.inpaintFullResPadding < 0 || ip.inpaintFullResPadding > 65536 then
      invalid("Invalid mask blur or inpaint padding")
    val source = difference.map(_._1).getOrElse(read(ip.initImage))
    val init = resize(source,w,h,ip.resizeMode)
    difference.map(_._2).orElse(ip.mask.map(path => DiffusionImageInput.readMask(Path.of(path)))) match
      case None => Inputs(init,None,init,None,None)
      case Some(rawMask) =>
        val aligned = resize(grayMask(rawMask,ip.inpaintingMaskInvert),source.getWidth,source.getHeight,0)
        val full = if ip.inpaintFullRes then Some(blur(aligned,ip.maskBlur,cancelled)) else None
        val box = full.flatMap(crop(_,w,h,ip.inpaintFullResPadding))
        def portion(im: BufferedImage, r: Rectangle) = resize(im.getSubimage(r.x,r.y,r.width,r.height),w,h,2)
        val image = box.fold(init)(portion(source,_))
        val mask = box.fold(blur(resize(aligned,w,h,ip.resizeMode),ip.maskBlur,cancelled))(portion(full.get,_))
        val filled = NativeInpaint.applyFill(image,mask,ip.inpaintingFill,seed,cancelled)
        Inputs(filled,Some(mask),if box.isDefined then source else image,
          if box.isDefined then full else Some(mask),box)
