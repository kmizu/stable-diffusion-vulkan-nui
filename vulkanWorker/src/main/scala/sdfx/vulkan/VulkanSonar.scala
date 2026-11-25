package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.Path
import sdfx.core.model.GenerationParams
import sdfx.vulkan.tools.{DiffusionImageInput,VulkanTools}
import ujson.Value

private[vulkan] final class VulkanSonar(h: Value,p: GenerationParams,seed: Long):
  private def num(key: String,d: Double): Double =
    val v=h.obj.get(key).fold(d)(_.num)
    require(v.isFinite,s"Sonar $key must be finite")
    v
  private def text(key: String,d: String)=h.obj.get(key).fold(d)(_.str)
  private val method=text("sampler","Euler a")
  private val momentum=num("momentum",.95)
  private val historyWeight=num("momentum_hist",.75)
  private val init=text("momentum_hist_init","zero")
  private val sign=text("momentum_sign","pos")
  private val reference=text("ref_img","")
  private val referenceMethod=text("ref_meth","linear")
  require(Set("Euler","Euler a","Naive")(method) && Set("pos","neg","rand")(sign) &&
    Set("zero","rand_init","rand_new")(init) && Set("linear","euler")(referenceMethod),"Invalid Sonar mode")
  require(momentum>=0 && momentum<=1 && historyWeight>=0 && historyWeight<=1,"Invalid Sonar momentum")
  private val strength=num("ref_hgf",.2)/10
  private val total=p.img2img.fold(p.steps)(i=>math.min(p.steps,(p.steps*i.denoisingStrength).toInt+1))
  private def bound(v: Double)=if v>1 then v.toInt else math.rint(v*total).toInt
  private val first=bound(num("ref_min_step",0))
  private val last=bound(num("ref_max_step",.75))

  def image: Option[BufferedImage] = Option.when(method=="Naive" && reference.nonEmpty && strength!=0){
    VulkanImageInputs.resize(DiffusionImageInput.readRgb(Path.of(reference)),p.width,p.height,0)
  }
  def sample(hasReference: Boolean,tiled: String): Seq[String] =
    val lines=Vector.newBuilder[String]
    var serial=0
    def put(expr: String): String =
      serial+=1
      val name=s"sonar$serial"
      lines+=s"$name = $expr"
      name
    lines+=s"sonar_rng = make_rng(\"cpu\", seed = ${seed & 0xffffffffL})"
    val rnd=new java.util.Random(seed)
    val ref=Option.when(hasReference){
      val encoded=put(s"vae_to_unet_latents(vae,vae_encode(vae,image::to_f32(sonar_ref),rng = sonar_rng$tiled))")
      val channels=(0 until 4).map{ c =>
        val v=put(s"tensor::index_select($encoded,axis = 2,indices = [$c])")
        put(s"tensor::div(tensor::add($v,-tensor::stat_f32($v,mode = \"mean\")),neutral_safe(tensor::stat_f32($v,mode = \"std\")))")
      }
      put(s"tensor::concat([${channels.mkString(",")}],axis = 2)")
    }
    var current="start"
    var history=init match
      case "rand_init" => Some(current)
      case "rand_new" => Some(put("tensor::randn_like(start,rng = sonar_rng)"))
      case _ => None
    for step<-0 until total do
      val sigma=s"sigmas[$step]"
      val next=s"sigmas[${step+1}]"
      val clean=put(s"sample($current,tensor::vector([$sigma,0],dtype = \"f32\"),predict = guided,method = \"euler\",rng = rng)")
      val derivative=put(s"tensor::div(tensor::add($current,tensor::mul($clean,-1)),$sigma)")
      val fixed=if momentum==1 then derivative else
        val positive=sign=="pos" || sign=="rand" && rnd.nextBoolean()
        val weight=(1-momentum)*(if positive then 1 else -1)
        val value=history.fold(put(s"tensor::mul($derivative,${1-weight})"))(prior=>put(s"tensor::lerp($derivative,$prior,$weight)"))
        val update=(1-historyWeight)*(if positive then 1 else -1)
        history=Some(history.fold(value)(prior=>put(s"tensor::lerp($prior,$value,$update)")))
        value
      val (down,up)=if method=="Euler a" then
        val additive=put(s"cads_sqrt($next * $next * ($sigma * $sigma - $next * $next) / ($sigma * $sigma))")
        val remaining=put(s"cads_sqrt($next * $next - $additive * $additive)")
        if step+1==total then ("0","0") else (remaining,additive)
      else (next,"0")
      current=put(s"tensor::add($current,tensor::mul($fixed,$down - $sigma))")
      if method=="Euler a" && step+1<total then
        current=put(s"tensor::add($current,tensor::mul(tensor::randn_like($current,rng = rng),$up))")
      for v<-ref if first<=step && step<=last do
        val source=if referenceMethod=="euler" then clean else current
        val target=put(s"tensor::add(tensor::mul($v,tensor::stat_f32($source,mode = \"std\")),tensor::stat_f32($source,mode = \"mean\"))")
        val pull=if referenceMethod=="euler" then s"($sigma - $next) * $strength / $sigma" else strength.toString
        current=put(s"tensor::lerp($current,$target,$pull)")
      if method=="Naive" then current=put(s"tensor::add($current,tensor::mul(tensor::randn_like($current,rng = sonar_rng),0.00001))")
    lines+=s"latent = $current"
    lines.result()

  def upscale(input: BufferedImage,paths: Map[String,String],model: VulkanTools.ModelUpscale): BufferedImage =
    val algorithm=text("upscaler","Lanczos")
    if !h.obj.get("upscale").exists(_.bool) || algorithm=="None" then return input
    var w=num("upscale_width",0).toInt
    var ht=num("upscale_height",0).toInt
    val ratio=num("upscale_ratio",1)
    require(w>=0 && ht>=0 && ratio>=1,"Invalid Sonar target size")
    if w==0 && ht==0 then
      w=math.rint(input.getWidth*ratio).toInt
      ht=math.rint(input.getHeight*ratio).toInt
    else if w==0 then w=math.rint(input.getWidth.toDouble*ht/input.getHeight).toInt
    else if ht==0 then ht=math.rint(input.getHeight.toDouble*w/input.getWidth).toInt
    if input.getWidth==w && input.getHeight==ht then return input
    val scale=math.max(w.toDouble/input.getWidth,ht.toDouble/input.getHeight)
    val cw=math.ceil(input.getWidth*scale).toInt.max(w)
    val ch=math.ceil(input.getHeight*scale).toInt.max(ht)
    val up=VulkanTools.upscaleImage(algorithm,paths,input,cw,ch,model)
    up.getSubimage((cw-w)/2,(ch-ht)/2,w,ht)
