package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream,ByteArrayOutputStream}
import javax.imageio.ImageIO
import javax.imageio.stream.{MemoryCacheImageInputStream,MemoryCacheImageOutputStream}
import scala.util.Using
import ujson.{Arr,Obj,Value}

private[tools] object FastloadUnits:
  import FastloadPickle.*
  private val resize=Vector("Just Resize","Crop and Resize","Resize and Fill")
  private val resizeAlias=Map("Inner Fit (Scale to Fit)"->1,"Scale to Fit (Inner Fit)"->1,"Outer Fit (Shrink to Fit)"->2,"Envelope (Outer Fit)"->2)
  private val modes=Set("Balanced","My prompt is more important","ControlNet is more important")
  private val resolution=Set("none","invert","canny")
  private val defaults=Map("input_mode"->str("simple"),"output_dir"->str(""),"loopback"->bool(false),"animatediff_batch"->bool(false),
    "union_control_type"->str("Unknown"),"batch_modifiers"->seq("list",Nil),"batch_image_files"->seq("list",Nil),"hr_option"->str("Both"),
    "use_preview_as_input"->bool(false),"guess_mode"->bool(false),"mask"->nil)
  private def num(x:Value):Double = kind(x) match
    case "int" => int(x).toDouble
    case "float" => x("value").num
    case _ => throw IllegalArgumentException("Invalid ControlNet numeric value")
  private def flag(x:Value):Boolean = {require(kind(x)=="bool");x("value").bool}
  final case class Mapped(params:Map[String,Value],image:Value,mask:Value,unapplied:Map[String,Value],original:Value):
    def wire:Value=dict(Map("params"->dict(params),"image"->image,"mask"->mask,"unapplied"->dict(unapplied),"original"->original))
    def plainParams:Value =
      def plain(x:Value):Value=kind(x) match
        case "dict" => Obj.from(fields(x).map((k,v)=>k->plain(v)))
        case "int" => ujson.Num(int(x).toDouble)
        case _ => x("value")
      plain(dict(params))
  def mapUnit(unit:Value):Mapped =
    val f=fields(unit)
    def at(k:String,d:Value)=f.getOrElse(k,d)
    def identity(k:String,d:String)=at(k,nil) match
      case v if v==nil || v==str("") => d
      case v => text(v)
    def range(k:String,d:Double,max:Double)=
      val n=num(at(k,number(d)));require(n.isFinite && n>=0 && n<=max,s"Invalid ControlNet $k");number(n)
    val module=identity("module","none")
    val start=range("guidance_start",0,1);val end=range("guidance_end",1,1);require(num(start)<=num(end))
    val r=at("resize_mode",str(resize(1)))
    val ri=if kind(r)=="int" then {val n=int(r);require(n>=0 && n<=3);n.toInt%3}
      else resizeAlias.getOrElse(text(r),resize.indexOf(text(r)))
    require(ri>=0,"Unknown ControlNet resize mode")
    val used=scala.collection.mutable.Set("enabled","model","module","weight","guidance_start","guidance_end","image","resize_mode","_sdfx_source_compatible_resize")
    val p=scala.collection.mutable.LinkedHashMap("enabled"->bool(flag(at("enabled",bool(false)))),"model"->str(identity("model","")),
      "preprocessor"->str(module),"weight"->range("weight",1,2),"guidance_start"->start,"guidance_end"->end,"resize_mode"->integer(ri),
      "source_compatible_resize"->bool(flag(at("_sdfx_source_compatible_resize",bool(true)))),"control_mode"->str("Balanced"),
      "processor_res"->integer(-1),"pixel_perfect"->bool(false),"preprocessor_args"->dict(Nil))
    val mode=at("control_mode",str("Balanced"))
    if kind(mode)=="string" && modes(text(mode)) then {p("control_mode")=mode;used+="control_mode"}
    if resolution(module) then
      val res=at("processor_res",integer(-1));val perfect=at("pixel_perfect",bool(false))
      if Set("int","float")(kind(res)) && num(res).isFinite && kind(perfect)=="bool" then
        val n=num(res);p("processor_res")=integer(if n>=64 && n<=2048 && n.isWhole then n.toInt else -1)
        p("pixel_perfect")=perfect;used++=Seq("processor_res","pixel_perfect")
      if module!="canny" then used++=Seq("threshold_a","threshold_b")
    for k<-Seq("low_vram","save_detected_map","is_ui","pulid_mode","inpaint_crop_input_image") if f.get(k).exists(v=>Set("bool","string")(kind(v))) do used+=k
    for (k,d)<-defaults if f.get(k).contains(d) do used+=k
    for k<-Seq("advanced_weighting","effective_region_mask","ipadapter_input","batch_images","batch_mask_dir","batch_keyframe_idx","generated_image") if f.get(k).contains(nil) do used+=k
    if module=="canny" then
      val args=for (k,t)<-Vector("threshold_a"->"low_threshold","threshold_b"->"high_threshold");v<-f.get(k);n=num(v) if {used+=k;n != -1} yield
        require(n>=0 && n<=255);t->number(n)
      p("preprocessor_args")=dict(args)
    var image=at("image",nil);var mask=nil:Value
    val extras=scala.collection.mutable.Map.empty[String,Value]
    if kind(image)=="dict" then
      val container=fields(image)
      image=container.getOrElse("image",nil);mask=container.getOrElse("mask",nil)
      container.filterNot((k,_)=>Set("image","mask")(k)).foreach((k,v)=>extras("image."+k)=v)
    else if Set("tuple","list")(kind(image)) && items(image).headOption.exists(v=>kind(v)=="dict" && fields(v).contains("image")) then
      extras("image.batch")=image
    else if Set("tuple","list")(kind(image)) && items(image).size==2 then
      val both=items(image);image=both.head;mask=both.last
    else if at("mask",nil)!=nil then {mask=f("mask");used+="mask"}
    if mask!=nil && Seq("inpaint","clip","ip-adapter").exists(module.contains) then extras("image.mask")=mask
    Mapped(p.toMap,image,mask,f.filterNot((k,_)=>used(k))++extras,unit)

  def fromNative(item:Value):Value =
    val fi=fields(item);require(fi.keySet.subsetOf(Set("params","image","mask")))
    val p=fields(fi("params"));require(p.keySet.subsetOf(Set("enabled","model","preprocessor","weight","guidance_start","guidance_end","preprocessor_args",
      "resize_mode","source_compatible_resize","control_mode","processor_res","pixel_perfect")),"Unsupported native ControlNet field")
    val module=text(p.getOrElse("preprocessor",str("none")))
    val args=fields(p.getOrElse("preprocessor_args",dict(Nil)))
    require(args.keySet.subsetOf(if module=="canny" then Set("low_threshold","high_threshold") else Set.empty))
    val r=int(p.getOrElse("resize_mode",integer(1)));require(r>=0 && r<3)
    val mode=p.getOrElse("control_mode",str("Balanced"));require(modes(text(mode)))
    val unit=scala.collection.mutable.LinkedHashMap("enabled"->p.getOrElse("enabled",bool(false)),"model"->p.getOrElse("model",str("")),
      "module"->str(module),"weight"->p.getOrElse("weight",integer(1)),"guidance_start"->p.getOrElse("guidance_start",integer(0)),
      "guidance_end"->p.getOrElse("guidance_end",integer(1)),"resize_mode"->str(resize(r.toInt)),"control_mode"->mode)
    if !flag(p.getOrElse("source_compatible_resize",bool(false))) then unit("_sdfx_source_compatible_resize")=bool(false)
    val res=int(p.getOrElse("processor_res",integer(-1)));val perfect=flag(p.getOrElse("pixel_perfect",bool(false)))
    require(res == -1 || res>=64 && res<=2048);require(resolution(module) || res == -1 && !perfect)
    if resolution(module) then {unit("processor_res")=integer(res);unit("pixel_perfect")=bool(perfect)}
    args.foreach((k,v)=>unit(if k=="low_threshold" then "threshold_a" else "threshold_b")=v)
    def image(key:String):Value=fi.getOrElse(key,nil) match
      case v if kind(v)=="bytes" => readImage(binary(v("base64").str));str(v("base64").str)
      case v if kind(v)=="array" || v==nil => v
      case _ => throw IllegalArgumentException("Invalid native image")
    val im=image("image");val mask=image("mask")
    require(im!=nil || mask==nil)
    if im!=nil then unit("image")=if mask==nil then im else dict(Seq("image"->im,"mask"->mask))
    val result=dict(unit);mapUnit(result);result

  private def readImage(bytes:Array[Byte]):BufferedImage =
    Using.resource(new MemoryCacheImageInputStream(new ByteArrayInputStream(bytes))) { stream =>
      val readers=ImageIO.getImageReaders(stream);require(readers.hasNext,"Unreadable Fastload image")
      val r=readers.next()
      try
        r.setInput(stream,true,true);require(r.getWidth(0).toLong*r.getHeight(0)<=32000000,"Fastload image too large");r.read(0)
      finally r.dispose()
    }
  private def image(x:Value):Option[BufferedImage] =kind(x) match
    case "null" => None
    case "string" =>
      val s=text(x);Some(readImage(binary(if s.startsWith("data:image/") && s.contains(',') then s.drop(s.indexOf(',')+1) else s)))
    case "bytes" => Some(readImage(binary(x("base64").str)))
    case "array" =>
      val s=shape(x);val bs=binary(x("data").str);require(expected(x)==bs.length && x("dtype").str=="|u1" && s.size>=2)
      val h=s(0);val w=s(1);val ch=s.lift(2).getOrElse(1)
      require(w.toLong*h<=32000000 && (s.size==2 || Set(3,4)(ch)))
      val out=new BufferedImage(w,h,if ch==1 then BufferedImage.TYPE_BYTE_GRAY else if ch==4 then BufferedImage.TYPE_INT_ARGB else BufferedImage.TYPE_INT_RGB)
      val fortran=x("order").str=="F"
      for y<-0 until h; xx<-0 until w do
        def v(c:Int)=bs(if fortran then y+h*(xx+w*c) else (y*w+xx)*ch+c)&255
        if ch==1 then out.getRaster.setSample(xx,y,0,v(0))
        else out.setRGB(xx,y,((if ch==4 then v(3) else 255)<<24)|(v(0)<<16)|(v(1)<<8)|v(2))
      Some(out)
    case _ => throw IllegalArgumentException("Unsupported Fastload image type")
  private def png(im:BufferedImage):Value =
    val out=new ByteArrayOutputStream
    Using.resource(new MemoryCacheImageOutputStream(out))(stream=>ImageIO.write(im,"png",stream))
    require(out.size<=Limit);ujson.Str(base.encodeToString(out.toByteArray))
  private def rgb(im:BufferedImage,x:Int,y:Int):Int =
    if im.getType==BufferedImage.TYPE_BYTE_GRAY then im.getRaster.getSample(x,y,0)*0x010101 else
      val p=im.getRGB(x,y);val a=(p>>>24)/255f
      def c(shift:Int)=(((p>>>shift)&255)*a+255f*(1f-a)).toInt
      (c(16)<<16)|(c(8)<<8)|c(0)
  def preview(unit:Mapped,native:Boolean):Value =
    var im=image(unit.image);var mask=image(unit.mask)
    val consume=mask.isDefined && !unit.unapplied.contains("image.mask")
    if native && im.isDefined && (consume || flag(unit.params("source_compatible_resize"))) then
      val a=im.get;val w=a.getWidth;val h=a.getHeight;val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
      val scribble=consume && mask.exists { b =>
        require(w==b.getWidth && h==b.getHeight,"Mask dimensions differ")
        var low=255;var high=0
        for y<-0 until h; x<-0 until w do
          val v=(rgb(b,x,y)>>>16)&255;low=low.min(v);high=high.max(v)
        low<250 && high>5
      }
      for y<-0 until h;x<-0 until w do out.setRGB(x,y,if scribble then ((rgb(mask.get,x,y)>>>16)&255)*0x010101 else rgb(a,x,y))
      im=Some(out)
      if consume then mask=None
    val result=Obj("image"->im.map(png).getOrElse[ujson.Value](ujson.Null),"mask"->mask.map(png).getOrElse[ujson.Value](ujson.Null))
    if native then
      val module=text(unit.params("preprocessor"))
      val extra=Option.when(flag(unit.params("enabled")) && module.nonEmpty && !VulkanTools.preprocessors.contains(module))(s"module=$module (unavailable)")
      result("params")=unit.plainParams;result("unapplied")=Arr.from((unit.unapplied.keys.toVector.sorted++extra).map(ujson.Str(_)))
    result
