package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import scala.util.Using
import ujson.{Arr,Obj}
import sdfx.vulkan.VulkanProtocol

private[vulkan] object VulkanVideoExtract:
  private def process[A](args: Seq[String], check: () => Unit)(body: Process => A): A =
    check()
    val native = try new ProcessBuilder(args*).redirectError(ProcessBuilder.Redirect.DISCARD).start()
    catch case _: java.io.IOException => VulkanProtocol.unsupported(s"video extraction requires ${args.head} on PATH")
    native.getOutputStream.close()
    val failure = new AtomicReference[Throwable]()
    val monitor = Thread.ofVirtual().start(() =>
      try
        while native.isAlive do
          check()
          Thread.sleep(100)
      catch
        case _: InterruptedException => ()
        case err: java.util.concurrent.CancellationException =>
          failure.set(err)
          native.destroyForcibly())
    try
      val value = body(native)
      require(native.waitFor(15,TimeUnit.SECONDS) && native.exitValue() == 0,s"${args.head} could not decode the video")
      check()
      value
    finally
      monitor.interrupt()
      native.destroyForcibly()
      native.waitFor(5,TimeUnit.SECONDS)
      Option(failure.get()).foreach(throw _)

  def run(path: Path, requested: Double, write: BufferedImage => Unit, check: () => Unit,
      progress: (Int,Int) => Unit): Obj =
    require(Files.isRegularFile(path),"Missing video")
    val probe = process(Seq("ffprobe","-v","error","-protocol_whitelist","file,pipe","-select_streams","v:0",
      "-show_entries","stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
      "-of","json","-i",path.toAbsolutePath.toString),check) { child =>
        val bytes = Using.resource(child.getInputStream)(_.readNBytes(16385))
        require(bytes.length <= 16384,"Unexpected video metadata size")
        ujson.read(bytes)
    }
    val stream = probe("streams").arr.headOption.getOrElse(throw IllegalArgumentException("Video stream missing"))
    val w = stream("width").num.toInt
    val h = stream("height").num.toInt
    require(w>0 && h>0 && w<=16384 && h<=16384 && w.toLong*h<=2097152,"Video dimensions exceed limit")
    def rate(key: String): Option[Double] = stream.obj.get(key).flatMap(_.strOpt).flatMap { s =>
      s.split("/") match
        case Array(a,b) => a.toDoubleOption.flatMap(n => b.toDoubleOption.filter(_>0).map(n/_))
        case _ => s.toDoubleOption
    }.filter(x => x.isFinite && x > 0)
    val fps = rate("avg_frame_rate").orElse(rate("r_frame_rate")).getOrElse(throw IllegalArgumentException("Video FPS unavailable"))
    val outputFps = if requested == 0 then fps else requested
    require(outputFps <= fps,"Requested FPS exceeds source FPS")
    val knownCount = stream.obj.get("nb_frames").flatMap(_.strOpt).flatMap(_.toIntOption).filter(_>0)
    val duration = stream.obj.get("duration").flatMap(_.strOpt).flatMap(_.toDoubleOption)
      .orElse(probe.obj.get("format").flatMap(_.obj.get("duration")).flatMap(_.strOpt).flatMap(_.toDoubleOption))
    val expected = knownCount.orElse(duration.filter(x=>x.isFinite && x>0).map(x => math.round(x*fps).toInt))
      .getOrElse(throw IllegalArgumentException("Unknown video duration or frame count"))
    require(expected>0 && expected<=10000 && expected.toLong*w*h<=2000000000L,"Video frame count exceeds limit")
    val rgb = new Array[Byte](w*h*3)
    var count = 0
    var next = 0.0
    process(Seq("ffmpeg","-nostdin","-v","error","-protocol_whitelist","file,pipe","-noautorotate","-i",path.toAbsolutePath.toString,
      "-map","0:v:0","-an","-sn","-dn","-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgb24","pipe:1"),check) { child =>
      Using.resource(child.getInputStream) { input =>
        var n = input.readNBytes(rgb,0,rgb.length)
        while n > 0 do
          check()
          require(n == rgb.length && count < 10000 && (count.toLong+1)*w*h <= 2000000000L,"Partial frame or video exceeds size limit")
          if count/fps+1e-9 >= next then
            val image = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
            val packed = Array.tabulate(w*h)(i => ((rgb(3*i)&255)<<16) | ((rgb(3*i+1)&255)<<8) | (rgb(3*i+2)&255))
            image.setRGB(0,0,w,h,packed,0,w)
            write(image)
            next += 1/outputFps
          count += 1
          progress(count,expected)
          n = input.readNBytes(rgb,0,rgb.length)
      }
    }
    require(count > 0 && knownCount.forall(_ == count),"Video decoder returned an incomplete sequence")
    Obj("dimensions" -> Arr(w,h),"fps" -> outputFps,"source_fps" -> fps,"duration" -> (count/fps))
