package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.util.concurrent.{CompletableFuture, TimeUnit}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.vulkan.FrontalFaces
import ujson.Value

private[vulkan] object VulkanFaceRestore:
  private val size = 512
  private val plane = size * size
  private def bin = Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR","."))).toAbsolutePath
  def available: Boolean = Files.isRegularFile(bin.resolve("sdv-ncnn.exe")) && Files.isRegularFile(bin.resolve("ncnn.dll"))

  private def run(image: BufferedImage, param: Path, weights: Path, device: String, weight: Option[Double], check: () => Unit): BufferedImage =
    val pixels = DiffusionImageResize.lanczos(image,size,size).getRGB(0,0,size,size,null,0,size)
    val bytes = ByteBuffer.allocate(plane * 12).order(ByteOrder.LITTLE_ENDIAN)
    for shift <- Seq(16,8,0); rgb <- pixels do bytes.putFloat(((rgb >>> shift) & 255) / 127.5f - 1f)
    val command = Vector(bin.resolve("sdv-ncnn.exe").toString,device,param.toString,weights.toString,
      "in0","out0","512","512","3") ++ weight.toSeq.flatMap(w => Seq("in1",w.toString))
    val process = new ProcessBuilder(command*).redirectError(ProcessBuilder.Redirect.INHERIT).start()
    try
      val io = CompletableFuture.supplyAsync(() => {
        Using.resource(process.getOutputStream)(_.write(bytes.array()))
        Using.resource(process.getInputStream)(_.readNBytes(plane * 12 + 1))
      })
      while !process.waitFor(100,TimeUnit.MILLISECONDS) do check()
      check()
      require(process.exitValue() == 0, s"GFPGAN inference failed (exit ${process.exitValue()})")
      val result = io.get(5,TimeUnit.SECONDS)
      require(result.length == plane * 12, "GFPGAN output shape mismatch")
      val tensor = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      for i <- pixels.indices do
        var rgb = 0
        for c <- 0 until 3 do
          val v = tensor.get(c * plane + i)
          require(v.isFinite, "Non-finite GFPGAN pixel")
          rgb = (rgb << 8) | math.round((v.max(-1f).min(1f) + 1f) * 127.5f)
        pixels(i) = rgb
      val out = new BufferedImage(size,size,BufferedImage.TYPE_INT_RGB)
      out.setRGB(0,0,size,size,pixels,0,size)
      DiffusionImageResize.lanczos(out,image.getWidth,image.getHeight)
    finally
      if process.isAlive then
        process.destroyForcibly()
        process.waitFor()

  def apply(source: BufferedImage, raw: Value, device: String, check: () => Unit): BufferedImage =
    require(available,"GFPGAN requires the bundled ncnn Vulkan runtime")
    val selected = Path.of(raw("model").str).toAbsolutePath
    val name = selected.getFileName.toString
    val codeformer = name.toLowerCase.contains("codeformer")
    val weight = Option.when(codeformer)(raw.obj.get("codeformer_weight").fold(0.5)(_.num))
    require(weight.forall(w => w >= 0 && w <= 1),"CodeFormer weight must be in [0,1]")
    require((codeformer || name.toLowerCase.contains("gfpgan")) && (name.endsWith(".param") || name.endsWith(".bin")),
      "Select a converted GFPGAN or CodeFormer .ncnn.bin model with its matching .param file")
    val stem = name.take(name.lastIndexOf('.'))
    val param = selected.resolveSibling(stem + ".param")
    val weights = selected.resolveSibling(stem + ".bin")
    require(Files.isRegularFile(param) && Files.isRegularFile(weights),"Missing GFPGAN .param or .bin")
    val visibility = raw.obj.get("visibility").fold(1.0)(_.num)
    require(visibility >= 0 && visibility <= 1,"Visibility must be in [0,1]")
    val out = new BufferedImage(source.getWidth,source.getHeight,BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    try g.drawImage(source,0,0,null) finally g.dispose()
    if visibility > 0 then
      for box <- FrontalFaces.detect(source, () => check()).asScala do
        check()
        val side = (box.width.max(box.height) * 1.6).toInt.min(out.getWidth).min(out.getHeight)
        val x = (box.getCenterX - side * .5).toInt.max(0).min(out.getWidth - side)
        val y = (box.getCenterY - side * .5).toInt.max(0).min(out.getHeight - side)
        val crop = out.getSubimage(x,y,side,side)
        val face = run(crop,param,weights,device,weight,check)
        val pad = (side / 12).max(2)
        val radius = pad * 3
        val kernel = (-radius to radius).map(j => math.exp(-0.5 * j.toDouble * j / (pad.toDouble * pad)))
        val sum = kernel.sum
        val axis = Array.tabulate(side)(i => (-radius to radius).iterator.filter(j => i+j >= pad && i+j <= side-pad).map(j => kernel(j+radius)).sum / sum)
        for row <- 0 until side do
          check()
          for col <- 0 until side do
            val alpha = axis(row) * axis(col) * visibility
            val old = crop.getRGB(col,row)
            val replaced = face.getRGB(col,row)
            var mixed = 0
            for shift <- Seq(16,8,0) do
              val a = (old >>> shift) & 255
              val b = (replaced >>> shift) & 255
              mixed |= math.round(a + (b - a) * alpha).toInt << shift
            crop.setRGB(col,row,mixed)
    out
