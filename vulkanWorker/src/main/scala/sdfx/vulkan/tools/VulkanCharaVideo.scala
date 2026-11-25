package sdfx.vulkan.tools

import java.nio.file.{Files,Path}
import java.util.concurrent.TimeUnit
import scala.util.Using

private[tools] object VulkanCharaVideo:
  def frames(file:Path,stage:Path,interval:Double,maximum:Int,check:()=>Unit)(process:(Path,Int)=>Unit):Unit =
    require(interval.isFinite && interval>0 && maximum>=1 && maximum<=2000,"Invalid frame interval or frame limit")
    val ffmpeg=sys.env.getOrElse("SDFX_FFMPEG","ffmpeg")
    val ffprobe=sys.env.getOrElse("SDFX_FFPROBE",if ffmpeg=="ffmpeg" then "ffprobe" else Path.of(ffmpeg).resolveSibling("ffprobe.exe").toString)
    val fpsProc=new ProcessBuilder(ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=avg_frame_rate","-of","csv=p=0",file.toString)
      .redirectError(ProcessBuilder.Redirect.INHERIT).start()
    def waitFor(p:Process):Unit =
      try
        while !p.waitFor(100,TimeUnit.MILLISECONDS) do check()
        check();require(p.exitValue()==0,"Could not decode video")
      finally if p.isAlive then p.destroyForcibly()
    waitFor(fpsProc)
    val rate=Using.resource(fpsProc.getInputStream)(in=>new String(in.readNBytes(128),java.nio.charset.StandardCharsets.US_ASCII).trim.split('/'))
    val fps=rate.headOption.flatMap(_.toDoubleOption).map(_/rate.lift(1).flatMap(_.toDoubleOption).getOrElse(1.0)).filter(v=>v>0 && v<=1000).getOrElse(25.0)
    val step=Math.rint(fps*interval).toInt.max(1)
    val prefix=java.util.UUID.randomUUID.toString
    val command=new ProcessBuilder(ffmpeg,"-nostdin","-v","error","-i",file.toString,"-vf",s"select=not(mod(n\\,$step))","-fps_mode","vfr","-frames:v",maximum.toString,
      "-start_number","0",stage.resolve(prefix+"-%06d.png").toString).redirectError(ProcessBuilder.Redirect.INHERIT).start()
    waitFor(command)
    var index=0
    while index<maximum && Files.isRegularFile(stage.resolve(f"$prefix-$index%06d.png")) do
      val frame=stage.resolve(f"$prefix-$index%06d.png")
      try {check();process(frame,index*step)} finally Files.deleteIfExists(frame)
      index+=1
    require(index>0,"Video contained no decodable frames")
