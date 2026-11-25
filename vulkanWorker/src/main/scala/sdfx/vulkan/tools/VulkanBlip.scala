package sdfx.vulkan.tools

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import java.util.concurrent.{CompletableFuture,TimeUnit}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.imagebrowser.BrowserFiles
import ujson.{Obj,Value}

private[vulkan] object VulkanBlip:
  private def bin=Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR","."))).toAbsolutePath
  private def baseModels=sys.env.get("SDFX_VULKAN_BLIP_DIR").map(Path.of(_)).getOrElse(bin.resolve("blip-base")).toAbsolutePath
  private val hashes=Vector("vision.ncnn.param"->"7da371282df5d9ea6d29b661eceadde5e4c83eff91356937d1f020d3de270be6",
    "vision.ncnn.bin"->"75fcfd12f40e59a8bd4a7e43f9ce38935da7d49ec92f84e63d9c4eaa1aecf7fd",
    "blip_decoder.ncnn.param"->"d253631a0113221d1cb48c95ffa9115c43903ba8e00732f88f4228dccb5808f3",
    "blip_decoder.ncnn.bin"->"6dce4d756ee2f9cf097833cb0a246cbca726bd659650d8a34ce54418091f6519",
    "vocab.txt"->"07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3")
  private val largeHashes=Vector("vision.ncnn.param"->"9aeb227a7bb2925c1f9df8a1974696c3cf09f03e6f105a19da69a53c6bfbf80c",
    "vision.ncnn.bin"->"4656300cb9619a81f0efee385f40069a5d4aa3e4ff0823bb91d6fa8af1af55db",
    "blip_decoder.ncnn.param"->"4a429dba1d874bf99c3a5df5b20d2ec71f5664b537a02bb02360d112e1c9f7b9",
    "blip_decoder.ncnn.bin"->"2b103055538099eb7ef1e9963b9fbeefc8b9a2ce77b2b3e9312c4a2aeede1d4a",hashes.last)
  private def assets(name:String)=name match
    case "blip-base" => (baseModels,hashes)
    case "blip-large" => (sys.env.get("SDFX_VULKAN_BLIP_LARGE_DIR").map(Path.of(_)).getOrElse(bin.resolve("blip-large")).toAbsolutePath,largeHashes)
    case _ => throw IllegalArgumentException(s"Unsupported caption model: $name")
  def known:Vector[String]=
    if !Files.isRegularFile(bin.resolve("sdv-ncnn.exe")) || !Files.isRegularFile(bin.resolve("ncnn.dll")) then return Vector.empty
    Vector("blip-base","blip-large").filter { name=>
      val (dir,files)=assets(name)
      files.forall((n,_)=>Files.isRegularFile(dir.resolve(n)))
    }
  def available:Boolean=known.nonEmpty
  def caption(path:Path,device:String,check:()=>Unit,maxTokens:Int=60,name:String="blip-base"):String =
    val (models,files)=assets(name)
    require(maxTokens>=1 && maxTokens<=64,"Caption length must be 1 to 64 tokens")
    require(known.contains(name),s"Install the converted weights for $name")
    files.foreach((n,h)=>{check();require(BrowserFiles.sha256(models.resolve(n))==h,s"Incorrect BLIP asset: $n")})
    val image=DiffusionImageResize.bicubic(DiffusionImageInput.readRgb(path,compositeAlpha=false),384,384)
    val rgb=image.getRGB(0,0,384,384,null,0,384)
    val pixels=ByteBuffer.allocate(12*rgb.length).order(ByteOrder.LITTLE_ENDIAN)
    val means=Array(.48145466f,.4578275f,.40821073f)
    val deviations=Array(.26862954f,.26130258f,.27577711f)
    for c<-0 until 3; pixel<-rgb do pixels.putFloat((((pixel>>>(16-8*c))&255)/255f-means(c))/deviations(c))
    val paths=files.take(4).map((n,_)=>models.resolve(n).toString)
    val args=Vector(bin.resolve("sdv-ncnn.exe").toString,"--blip",device)++paths:+maxTokens.toString
    val process=new ProcessBuilder(args.asJava).redirectError(ProcessBuilder.Redirect.INHERIT).start()
    try
      val io=CompletableFuture.supplyAsync(()=>{Using.resource(process.getOutputStream)(_.write(pixels.array()));Using.resource(process.getInputStream)(_.readNBytes(maxTokens*4+1))})
      while !process.waitFor(100,TimeUnit.MILLISECONDS) do check()
      check();require(process.exitValue()==0,s"BLIP Vulkan inference failed (${process.exitValue()})")
      val bytes=io.get(10,TimeUnit.SECONDS);require(bytes.length<=maxTokens*4 && bytes.length%4==0)
      val vocab=Files.readAllLines(models.resolve("vocab.txt")).asScala
      val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
      val words=Vector.fill(buffer.remaining())(buffer.get()).filterNot(Set(0,100,101,102,103,30522,30523)).map(id=>vocab(id))
      words.mkString(" ").replace(" ##","").replace(" .",".").replace(" ?","?").replace(" !","!").replace(" ,",",")
        .replace(" ' ","'").replace(" n't","n't").replace(" 'm","'m").replace(" 's","'s").replace(" 've","'ve").replace(" 're","'re").trim
    finally if process.isAlive then {process.destroyForcibly();process.waitFor()}
