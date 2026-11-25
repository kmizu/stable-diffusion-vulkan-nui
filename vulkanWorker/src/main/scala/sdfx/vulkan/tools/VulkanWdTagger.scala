package sdfx.vulkan.tools

import java.awt.{Color,Graphics2D}
import java.awt.image.BufferedImage
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import java.util.concurrent.{CompletableFuture,TimeUnit}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import ujson.{Obj,Value}
import sdfx.core.imagebrowser.BrowserFiles

private[vulkan] object VulkanWdTagger:
  val modelId="SmilingWolf/wd-vit-tagger-v3@7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f"
  val modelSha="35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6"
  val tableSha="298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217"
  private val required=Vector("wd.ncnn.param"->"17b3c57ffc6877ae8865004ecfb3b82c59d85adb51f460a3b1932f568cc5d293",
    "wd.ncnn.bin"->"8c93f8b15a7ec2a32ede1edcdf695d8fcd339dca9c73bfed74c14918a5201d27","selected_tags.csv"->tableSha)
  private val emoticons=Set("0_0","(o)_(o)","+_+","+_-","._.","<o>_<o>","<|>_<|>","=_=",">_<","3_3","6_9",">_o","@_@","^_^","o_o","u_u","x_x","|_|","||_||")
  private def bin=Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR","."))).toAbsolutePath
  private def models=sys.env.get("SDFX_VULKAN_WD_DIR").map(Path.of(_)).getOrElse(bin.resolve("wd-vit-v3")).toAbsolutePath
  def available:Boolean=Files.isRegularFile(bin.resolve("sdv-ncnn.exe")) && Files.isRegularFile(bin.resolve("ncnn.dll")) && required.forall((n,_)=>Files.isRegularFile(models.resolve(n)))
  def features:Vector[String]=if available then Vector("caption","batch_interrogate_wd_v3_partial") else Vector.empty
  private def input(image:BufferedImage):Array[Byte] =
    val side=image.getWidth.max(image.getHeight)
    require(side.toLong*side<=32000000,"WD image exceeds square size limit")
    val square=new BufferedImage(side,side,BufferedImage.TYPE_INT_RGB)
    val graphics=square.createGraphics()
    try
      graphics.setColor(Color.WHITE);graphics.fillRect(0,0,side,side)
      graphics.drawImage(image,(side-image.getWidth)/2,(side-image.getHeight)/2,null)
    finally graphics.dispose()
    val resized=DiffusionImageResize.bicubic(square,448,448,32000000)
    val bytes=ByteBuffer.allocate(448*448*3*4).order(ByteOrder.LITTLE_ENDIAN)
    for pixel<-resized.getRGB(0,0,448,448,null,0,448) do
      bytes.putFloat(pixel&255);bytes.putFloat((pixel>>>8)&255);bytes.putFloat((pixel>>>16)&255)
    bytes.array()
  def tags(path:Path,device:String,threshold:Double,batch:Boolean,check:()=>Unit):String =
    require(threshold.isFinite && threshold>=0 && threshold<=1,"Tag threshold must be in [0,1]")
    if !available then throw sdfx.core.protocol.EngineError("unsupported","WD tagger requires the Vulkan ncnn runtime and converted WD v3 weights")
    required.foreach { (n,h) =>
      check();require(BrowserFiles.sha256(models.resolve(n))==h,s"Incorrect digest for $n")
    }
    val pixels=input(DiffusionImageInput.readRgb(path,maxPixels=32000000))
    val command=new ProcessBuilder(bin.resolve("sdv-ncnn.exe").toString,device,models.resolve("wd.ncnn.param").toString,
      models.resolve("wd.ncnn.bin").toString,"in0","out0","3","448","448")
    command.environment().put("PATH",bin.toString+java.io.File.pathSeparator+command.environment().getOrDefault("PATH",""))
    val process=command.redirectError(ProcessBuilder.Redirect.INHERIT).start()
    val output=CompletableFuture.supplyAsync(() => {
      Using.resource(process.getOutputStream)(_.write(pixels))
      Using.resource(process.getInputStream)(_.readNBytes(10861*4+1))
    })
    try
      while !process.waitFor(100,TimeUnit.MILLISECONDS) do check()
      check()
      require(process.exitValue()==0,"WD ncnn inference failed")
      val bytes=output.get(5,TimeUnit.SECONDS)
      require(bytes.length==10861*4,"Unexpected WD score tensor")
      val scores=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      val rows=Files.readAllLines(models.resolve("selected_tags.csv")).asScala.drop(1).toVector
      require(rows.size==scores.remaining)
      val selected=rows.flatMap { line =>
        val last=line.lastIndexOf(',');val category=line.lastIndexOf(',',last-1);val start=line.indexOf(',')
        val name=line.substring(start+1,category).stripPrefix("\"").stripSuffix("\"").replace("\"\"","\"")
        val cat=line.substring(category+1,last).toInt;val score=scores.get()
        require(score.isFinite && score>=0 && score<=1,"Invalid WD tag score")
        Option.when(if batch then (cat==0 || cat==4) && score>threshold else cat==0 && score>=threshold)(
          (if batch && emoticons(name) then name else name.replace('_',' '),score))
      }
      (if batch then selected else selected.sortBy(-_._2)).map(_._1).distinct.mkString(", ")
    finally if process.isAlive then {process.destroyForcibly();process.waitFor()}
  def caption(raw:Value,device:String,check:()=>Unit):Value =
    if !raw.obj.get("method").contains(ujson.Str("wd14")) then throw sdfx.core.protocol.EngineError("unsupported","Vulkan caption supports the wd14 tagger")
    Obj("caption"->tags(Path.of(raw("image_path").str),device,raw.obj.get("threshold").fold(.35)(_.num),false,check))
