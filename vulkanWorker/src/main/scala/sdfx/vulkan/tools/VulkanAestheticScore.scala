package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import java.util.zip.ZipFile
import scala.util.Using
import sdfx.vulkan.VulkanInference
import ujson.Value

private[vulkan] object VulkanAestheticScore:
  private final case class Head(repo:String,revision:String,file:String,sha:String,widths:Vector[Int])
  private val improved=Head("christophschuhmann/improved-aesthetic-predictor","6934dd81792f086e613a121dbce43082cb8be85e",
    "sac+logos+ava1-l14-linearMSE.pth","21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0",Vector(768,1024,128,64,16,1))
  private val simulacra=Head("crowsonkb/simulacra-aesthetic-models","701f7678b305acf8a0634350365a86beb95dd87d",
    "models/sac_public_2022_06_29_vit_l_14_linear.pth","f77d1d0f4bb04b0ff8e60c5747fd0bfff0cc4a85263a6106627522c06eae85c6",Vector(768,1))

  private val simulacraB16=simulacra.copy(file="models/sac_public_2022_06_29_vit_b_16_linear.pth",
    sha="cdc27b2196565bcb69f2692731084b0189612b98aae74adec033782469ab583c",widths=Vector(512,1))

  private def root:Path = sys.env.get("SDFX_AESTHETIC_CACHE").map(Path.of(_)).getOrElse {
    val home=Path.of(sys.props("user.home"),".cache")
    val old=home.resolve("stablediffusion-nui/aesthetic_score")
    val current=home.resolve("stable-diffusion-nui/aesthetic_score")
    if !Files.exists(current) && Files.exists(old) then old else current
  }
  // The pinned archives contain contiguous float32 storages. Never evaluate the pickle metadata.
  private def predict(head:Head,input:Array[Float],check:()=>Unit):Double =
    val dir=Files.createDirectories(root.resolve(head.repo.replace("/","__")).resolve(head.revision))
    val url=s"https://raw.githubusercontent.com/${head.repo}/${head.revision}/${head.file}"
    val path=VulkanClip.asset(dir,Path.of(head.file).getFileName.toString,url,head.sha,check,(_,_,_)=>())
    Using.resource(new ZipFile(path.toFile)) { zip =>
      def storage(i:Int,n:Int):Array[Float] =
        val entry=zip.getEntry(s"archive/data/$i")
        require(entry!=null && entry.getSize==4L*n,"Invalid aesthetic predictor tensor")
        val bytes=Using.resource(zip.getInputStream(entry))(_.readNBytes(4*n))
        val out=new Array[Float](n)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer.get(out)
        out
      var row=input
      for i<-1 until head.widths.size do
        check()
        val width=head.widths(i-1)
        val count=head.widths(i)
        val weights=storage((i-1)*2,width*count)
        val bias=storage((i-1)*2+1,count)
        row=Array.tabulate(count) { k =>
          var j=0
          var sum=bias(k).toDouble
          while j<width do
            sum+=weights(k*width+j).toDouble*row(j)
            j+=1
          sum.toFloat
        }
      row(0).toDouble
    }
  final case class Result(fields:Vector[(String,String)],chunks:Vector[(String,String)])
  def apply(image:BufferedImage,config:Value,engine:VulkanInference,device:String,check:()=>Unit):Result =
    def flag(k:String,default:Boolean=false)=config.obj.get(k).fold(default)(_.bool)
    require(!flag("force_cpu"),"Aesthetic scoring uses the selected Vulkan adapter")
    require(!Seq("cafe_aesthetic","cafe_style","cafe_waifu").exists(k=>flag(k)),"Cafe BEiT classifiers are unavailable")
    val sim=config.obj.get("simulacra").fold("off")(_.str)
    require(Set("off","vit-l-14","vit-b-16")(sim),"Unknown simulacra model")
    val useImproved=flag("improved",true)
    val fields=Vector.newBuilder[(String,String)]
    val chunks=Vector.newBuilder[(String,String)]
    if useImproved || sim=="vit-l-14" then
      val program=VulkanAesthetic.visionProgram(root,device,check,(_,_,_)=>())
      try
        if useImproved then
          val row=VulkanAesthetic.projection(image,engine,program,false,true)
          val value=String.format(java.util.Locale.ROOT,"%.1f",Double.box(predict(improved,row,check)))
          fields+="aesthetic_score"->value
          chunks+="aesthetic_score"->value
        if sim=="vit-l-14" then
          val row=VulkanAesthetic.projection(image,engine,program,true,true)
          val value=String.format(java.util.Locale.ROOT,"%.2f",Double.box(predict(simulacra,row.map(v=>(v*math.sqrt(768)).toFloat),check)))
          fields+="Aesthetic Scorer ViT-L/14"->value
          chunks+="score"->value
      finally engine.unloadClipVision()
    if sim=="vit-b-16" then
      val program=VulkanBaseClip.program(root,16,device,check)
      try
        val row=VulkanAesthetic.projection(image,engine,program,true,true,512)
        val value=String.format(java.util.Locale.ROOT,"%.2f",Double.box(predict(simulacraB16,row.map(v=>(v*math.sqrt(512)).toFloat),check)))
        fields+="Aesthetic Scorer ViT-B/16"->value
        chunks+="score"->value
      finally engine.unloadClipVision()
    Result(fields.result(),if flag("png_chunks",true) then chunks.result() else Vector.empty)
  def metadata(result:Result,cfg:Value,p:sdfx.core.model.GenerationParams,prompt:String,seed:Long,modelHash:String):Result =
    def flag(k:String,default:Boolean=false)=cfg.obj.get(k).fold(default)(_.bool)
    val extra=if flag("generation_fields") then
      Vector("sampler"->p.samplerName,"cfg_scale"->BigDecimal(p.cfgScale).bigDecimal.stripTrailingZeros.toPlainString,
        "sd_model_hash"->modelHash,"seed"->seed.toString)
    else Vector.empty
    val hashes=if flag("hash") then
      def quoted(s:String)=ujson.write(ujson.Str(s),escapeUnicode=true)
      val entries=Vector("steps"->p.steps.toString,"sampler"->quoted(p.samplerName),"cfg_scale"->p.cfgScale.toString,
        "seed"->seed.toString,"width"->p.width.toString,"height"->p.height.toString,"model"->quoted(modelHash),
        "prompt"->quoted(prompt),"negative_prompt"->quoted(p.negativePrompt))
      val json=entries.map((k,v)=>quoted(k)+": "+v).mkString("{",", ","}")
      val md5=java.security.MessageDigest.getInstance("MD5").digest(json.getBytes(java.nio.charset.StandardCharsets.UTF_8))
      Vector("hash"->java.util.HexFormat.of().formatHex(md5))
    else Vector.empty
    Result(result.fields++hashes,if flag("png_chunks",true) then result.chunks++extra++hashes else Vector.empty)
