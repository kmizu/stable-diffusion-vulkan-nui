package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import scala.collection.mutable.ArrayBuffer
import scala.util.Using
import sdfx.core.image.Gif
import sdfx.core.protocol.EngineError
import ujson.{Obj, Value}

private[tools] object VulkanVideo:
  def encode(raw: Value, ctx: NativeWorkContext): Value =
    val p = raw.obj
    val inputs = p("images").arr.map(i => Path.of(i.str)).toVector
    require(inputs.nonEmpty && inputs.size <= 512, "Choose 1 to 512 frames")
    val closed = p.get("closed_loop").fold("N")(_.str)
    require(Set("N","R-P","R+P","A")(closed), "Unknown closed loop mode")
    val batch = p.get("batch_size").fold(16)(_.num.toInt)
    val frames = if closed == "A" && inputs.size <= batch then inputs ++ inputs.drop(1).dropRight(1).reverse else inputs
    val fps = p.get("fps").fold(8)(_.num.toInt)
    val loops = p.get("loop_number").fold(0)(_.num.toInt)
    require(fps > 0 && fps <= 120 && loops >= 0 && loops <= 65535, "Invalid playback rate or loop count")
    val formats = p.get("formats").toSeq.flatMap(_.arr.map(_.str.toUpperCase(java.util.Locale.ROOT))).distinct
    val requested = if formats.isEmpty then Seq("GIF") else formats
    require(requested.forall(Set("GIF","WEBP","WEBM","MP4")), "Unknown animation format")
    val dir = Files.createDirectories(Path.of(p.get("output_dir").fold(".")(_.str)))
    val stem = p.get("name").flatMap(_.strOpt).filter(_.nonEmpty).getOrElse("animatediff")
    require(stem.length <= 160 && !stem.exists(c => c < ' ' || "\\/:*?\"<>|".contains(c)), "Invalid animation name")
    val files = ArrayBuffer.empty[Path]
    val outputs = Obj()
    try
      for format <- requested do
        val path = Files.createTempFile(dir, stem + "-", "." + format.toLowerCase(java.util.Locale.ROOT))
        files += path
        var dimensions = Option.empty[(Int,Int)]
        val images = frames.iterator.map { f =>
          ctx.checkCancelled()
          val image = DiffusionImageInput.readRgb(f)
          val size = (image.getWidth, image.getHeight)
          require(dimensions.forall(_ == size), "Animation frames have different dimensions")
          dimensions = Some(size)
          image
        }
        if format == "GIF" then Gif.writeSequence(images, path, math.round(1000.0 / fps).toInt, Some(loops))
        else ffmpeg(images, path, format, fps, loops, ctx)
        outputs(format) = path.toAbsolutePath.toString
      ctx.beginPublication()
      Obj("outputs" -> outputs, "frames" -> frames.size)
    catch case error: Throwable =>
      files.foreach(Files.deleteIfExists)
      throw error

  private def ffmpeg(frames: Iterator[BufferedImage], dst: Path, format: String, fps: Int, loops: Int, ctx: NativeWorkContext): Unit =
    val first = frames.next()
    val codecs = format match
      case "WEBP" => Seq("-c:v","libwebp_anim","-quality","90","-loop",loops.toString)
      case "WEBM" => Seq("-c:v","libvpx-vp9","-pix_fmt","yuv420p","-b:v","0","-crf","30")
      case "MP4" => Seq("-c:v","libx264","-pix_fmt","yuv420p","-crf","18")
    val args = Seq("ffmpeg","-nostdin","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
      "-s",s"${first.getWidth}x${first.getHeight}","-framerate",fps.toString,"-i","pipe:0") ++ codecs :+ dst.toString
    val builder = new ProcessBuilder(args*)
    builder.redirectOutput(ProcessBuilder.Redirect.DISCARD)
    builder.redirectError(ProcessBuilder.Redirect.INHERIT)
    val proc = try builder.start() catch case _: java.io.IOException =>
      throw EngineError("unsupported", s"$format encoding requires ffmpeg on PATH")
    try
      Using.resource(new java.io.BufferedOutputStream(proc.getOutputStream)) { stream =>
        val rgb = new Array[Int](first.getWidth)
        val row = new Array[Byte](first.getWidth * 3)
        (Iterator.single(first) ++ frames).foreach { img =>
          for y <- 0 until img.getHeight do
            img.getRGB(0,y,rgb.length,1,rgb,0,rgb.length)
            var i = 0
            while i < rgb.length do
              row(3*i) = (rgb(i) >>> 16).toByte
              row(3*i+1) = (rgb(i) >>> 8).toByte
              row(3*i+2) = rgb(i).toByte
              i += 1
            stream.write(row)
        }
      }
      while !proc.waitFor(200, java.util.concurrent.TimeUnit.MILLISECONDS) do ctx.checkCancelled()
      require(proc.exitValue() == 0, s"ffmpeg could not encode $format")
    finally if proc.isAlive then proc.destroyForcibly()
