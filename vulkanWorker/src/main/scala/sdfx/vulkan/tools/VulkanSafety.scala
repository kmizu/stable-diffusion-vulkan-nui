package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import sdfx.core.config.AppHome
import sdfx.core.image.RegionCensor
import sdfx.vulkan.{VulkanInference,VulkanProtocol}
import ujson.{Obj,Value}

private[vulkan] object VulkanSafety:
  def bin = Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR","."))).toAbsolutePath
  def models = sys.env.get("SDFX_VULKAN_NSFW_DIR").map(Path.of(_)).getOrElse(bin.resolve("nsfw")).toAbsolutePath
  def has(name: String): Boolean = Seq("param","bin").forall(s=>Files.isRegularFile(models.resolve(s"$name.ncnn.$s")))
  def available: Boolean = Files.isRegularFile(bin.resolve("sdv-ncnn.exe")) && has("nudenet")
  val labels=Vector("drawings","hentai","neutral","porn","sexy")
  def forced(key: String)=sys.env.get("SDFX_NSFW_CHECKER_FORCE_FLAGGED").exists(_.split(",").exists(_.trim.equalsIgnoreCase(key)))

private[vulkan] final class VulkanSafety(engine: VulkanInference,device: String,name: String,check: ()=>Unit) extends RegionCensor:
  import VulkanSafety.*
  private val home=AppHome(models)
  private def ncnn(model: String,data: Array[Float],w: Int,h: Int,n: Int)=
    if !has(model) then VulkanProtocol.unsupported(s"Missing converted $model model in $models")
    NcnnTensor.run(bin.resolve("sdv-ncnn.exe"),name,models,model,data,w,h,n,()=>check())

  override def detectMask(img: BufferedImage,home: AppHome,thresholds: Array[Float],expandH: Array[Float],expandV: Array[Float],
      nmsThreshold: Double,maskShape: String,roundRadius: Double,modelPath: Option[Path],log: Option[String=>Unit]): Option[BufferedImage] =
    if modelPath.nonEmpty then VulkanProtocol.unsupported("Custom NudeNet detector paths")
    val predictions=ncnn("nudenet",preprocess(img),320,320,22*2100)
    val boxes=decodeDetections(predictions.grouped(2100).toArray,img.getWidth,img.getHeight,thresholds,expandH,expandV,nmsThreshold)
    boxes.foreach(b=>log.foreach(_(s"${Friendly(b.classIndex)}: ${b.score}")))
    Option.when(boxes.nonEmpty)(renderMask(img.getWidth,img.getHeight,boxes,maskShape,roundRadius))

  private def nsfwjs(image: BufferedImage,level: String): (Boolean,Obj) =
    require(Set("safe","questionable","explicit")(level),"Invalid NSFWJS safety level")
    val info=Obj("checker"->"nsfwjs","safety_level"->level,"label"->"")
    if level=="explicit" then return (false,info)
    val w=image.getWidth
    val h=image.getHeight
    val bytes=Array.tabulate(224*224*3){i=>
      val x=(((i/3%224)+0.5)*w/224).toInt.min(w-1)
      val y=(((i/3/224)+0.5)*h/224).toInt.min(h-1)
      ((image.getRGB(x,y)>>>(16-8*(i%3)))&255)/255f
    }
    val values=ncnn("nsfwjs",bytes,3,224,5)
    val label=labels(values.indices.maxBy(values(_)))
    info("label")=label
    for (key,score)<-labels.zip(values) do info(s"score_$key")=math.rint(score.toDouble*1e6)/1e6
    (Set("hentai","porn")(label) || level=="safe" && label=="sexy" || forced("nsfwjs"),info)

  private def safety(image: BufferedImage): Boolean =
    val model=models.resolve("safety-vision.safetensors")
    val head=models.resolve("concepts.json")
    if !Files.isRegularFile(model) || !Files.isRegularFile(head) then VulkanProtocol.unsupported("SD safety checker weights (scripts/build_vulkan_safety.py)")
    val file=ujson.write(ujson.Str(model.toString))
    val gpu=ujson.write(ujson.Str(device))
    val graph=s"""fn main(input) {
      vision = load_vision_encoder($file,backend = $gpu,params_backend = $gpu)
      return {embedding = encode_vision(vision,input,pooled = true)}
    }"""
    val projected=try VulkanAesthetic.projection(image,engine,graph,lanczos=false,roundCrop=false)
      finally engine.unloadClipVision()
    val weights=ujson.read(Files.readAllBytes(head))
    def scores(key: String)=weights(key).arr.zip(weights(key+"_weights").arr).map{ (row,cutoff)=>
      val vector=row.arr.map(_.num)
      require(vector.size==projected.length,"Safety concept dimension mismatch")
      val norm=math.sqrt(vector.iterator.map(v=>v*v).sum)
      vector.indices.iterator.map(i=>vector(i)*projected(i)).sum/norm-cutoff.num
    }
    var adjustment=0.0
    for score<-scores("special_care_embeds") do
      if math.rint((score+adjustment)*1000)>0 then adjustment=0.01
    scores("concept_embeds").exists(s=>math.rint((s+adjustment)*1000)>0) || forced("safety_checker")

  private def whole(image: BufferedImage,checker: String,level: String,report: Boolean): (BufferedImage,Obj) =
    val (flag,info)=if checker=="safety_checker" then (safety(image),Obj("checker"->checker)) else nsfwjs(image,level)
    info("is_nsfw")=if flag then 1 else 0
    info("detected")=if flag then 1 else 0
    info("censored")=if flag && !report then 1 else 0
    val result=if !flag || report then image else if checker=="safety_checker" then
      new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
    else blurRgb(image,20)
    (result,info)

  def filter(image: BufferedImage,options: Value,preview: Boolean): CensorGenerationResult =
    val cfg=parseGenerationConfig(options.asInstanceOf[Obj])
    val safe=if cfg.safetyChecker && (!preview || cfg.safetyCheckerLivePreview) then whole(image,"safety_checker","safe",false)._1 else image
    val screened=if cfg.nsfwjs then whole(safe,"nsfwjs",cfg.nsfwjsSafetyLevel,false)._1 else safe
    val region=if preview then cfg.copy(genFilterType=cfg.livePreviewFilterType,saveBeforeCensor=false,saveMask=false) else cfg
    censorGeneration(screened,region,home)

  def tool(raw: Value,output: Path,password: Option[String]): Obj =
    val checker=raw.obj.get("checker").fold("nudenet")(_.str)
    if checker=="nudenet" then
      return censorTool(raw.asInstanceOf[Obj],home,output,writeOutput=(image,file)=>sdfx.core.png.ScrambledPng.write(image,file,Seq.empty,password))
    require(checker=="safety_checker" || checker=="nsfwjs","Invalid NSFW checker")
    val source=DiffusionImageInput.readRgb(Path.of(raw("image_path").str))
    val (result,info)=whole(source,checker,raw.obj.get("safety_level").fold("safe")(_.str),raw.obj.get("report_only").exists(_.bool))
    val file=Files.createTempFile(Files.createDirectories(output),"nsfw-",".png")
    sdfx.core.png.ScrambledPng.write(result,file,Seq("parameters"->ujson.write(info)),password)
    Obj("path"->file.toString,"info"->info)
