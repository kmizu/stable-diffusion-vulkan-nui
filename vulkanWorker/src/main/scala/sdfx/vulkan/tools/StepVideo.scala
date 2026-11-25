package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{CancellationException, TimeUnit}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer

private[tools] object StepVideo:
  private val codecs = Map("h264" -> "libx264", "h265" -> "libx265", "vp9" -> "libvpx-vp9", "av1" -> "libaom-av1", "prores" -> "prores_ks")
  def extension(codec: String): String = Map("vp9" -> "webm", "prores" -> "mov").getOrElse(codec,"mp4")

  def directory(root: Path, relative: String): Path =
    val base = Files.createDirectories(root).toRealPath()
    val target = relative.split("/").foldLeft(base) { (parent, component) =>
      val next = parent.resolve(component)
      if Files.isSymbolicLink(next) then throw IllegalArgumentException("Steps Animation directories must stay under the output folder")
      Files.createDirectories(next).toRealPath()
    }
    require(target.startsWith(base),"Steps Animation directory escapes the output folder")
    target

  def encode(root: Path, capture: Path, frames: Vector[IntermediateFrameRecord], s: StepsAnimationConfig, cancelled: () => Boolean): Path =
    val videoDir = directory(root,s.outputDir)
    val stage = Files.createTempDirectory(videoDir,".encode-")
    val inputs = ArrayBuffer.empty[Path]
    val ext = extension(s.codec)
    val temp = stage.resolve("video." + ext)
    val target = videoDir.resolve(capture.getFileName.toString + "." + ext)
    val width = ((frames.map(_.width).max+1)/2)*2
    val height = ((frames.map(_.height).max+1)/2)*2
    def check(): Unit = if cancelled() then throw CancellationException("Steps Animation export interrupted")
    try
      frames.zipWithIndex.foreach { (frame,idx) =>
        check()
        val input = stage.resolve(f"$idx%06d.png")
        inputs += input
        if frame.width == width && frame.height == height then Files.copy(Path.of(frame.path),input)
        else
          val img = DiffusionImageInput.readRgb(Path.of(frame.path))
          val scale = math.min(width.toDouble/frame.width,height.toDouble/frame.height)
          val fitted = DiffusionImageResize.lanczos(img,math.max(1,(frame.width*scale).round.toInt),math.max(1,(frame.height*scale).round.toInt))
          val canvas = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
          val g = canvas.createGraphics()
          try g.drawImage(fitted,(width-fitted.getWidth)/2,(height-fitted.getHeight)/2,null) finally g.dispose()
          require(ImageIO.write(canvas,"png",input.toFile),"PNG encoder unavailable")
      }
      val rate = frames.size / s.duration
      val interp = if frames.size == 1 || s.interpolation == "none" then s"fps=${s.fps}"
        else s"minterpolate=fps=${s.fps}:mi_mode=${s.interpolation}"
      val filter = s"tpad=stop_mode=clone:stop_duration=${2/rate},$interp,trim=duration=${s.duration},setpts=PTS-STARTPTS"
      val pixel = if s.codec == "prores" then "yuv422p10le" else "yuv420p"
      val log = capture.resolve("ffmpeg.log")
      val command = Seq("ffmpeg","-nostdin","-v","error","-n","-framerate",rate.toString,"-i",stage.resolve("%06d.png").toString,
        "-vf",filter,"-t",s.duration.toString,"-c:v",codecs(s.codec),"-pix_fmt",pixel,temp.toString)
      val process = new ProcessBuilder(command*).redirectErrorStream(true).redirectOutput(log.toFile).start()
      try
        while !process.waitFor(150,TimeUnit.MILLISECONDS) do check()
        require(process.exitValue()==0,s"ffmpeg failed; see $log")
      finally if process.isAlive then process.destroyForcibly().waitFor()
      check()
      Files.move(temp,target)
    finally
      inputs.foreach(Files.deleteIfExists(_))
      Files.deleteIfExists(temp)
      Files.deleteIfExists(stage)
