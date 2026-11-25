package sdfx.vulkan.tools

import java.awt.{Color,Rectangle}
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import scala.collection.mutable
import sdfx.core.model.{GenerationParams,Img2ImgParams}
import sdfx.vulkan.{VulkanImageInputs,VulkanInference,VulkanProtocol}
import ujson.{Arr,Obj,Value}

private[vulkan] object VulkanADetailer:
  type Refine = (String,String,Int,Int,Int,Double,String,Long,VulkanInference.ImageOptions) => BufferedImage

  def facePass(h: Value, currentPath: String, currentHash: String): Value =
    def number(k:String,d:Double) = h.obj.get(k).fold(d)(_.num)
    def text(k:String,d:String) = h.obj.get(k).fold(d)(_.str)
    val checkpoint=text("checkpoint","")
    if checkpoint.nonEmpty && !Files.isSameFile(Path.of(checkpoint),Path.of(currentPath)) then
      VulkanProtocol.unsupported("Batch Face Swap using a separate checkpoint")
    val hash=text("checkpoint_hash","")
    require(hash.isEmpty || hash.equalsIgnoreCase(currentHash),"Face checkpoint hash differs from recorded hash")
    val side=number("resolution",0).toInt/8*8
    val steps=number("steps",0);val cfg=number("cfg_scale",0);val sampler=text("sampler","")
    val unit=Obj("model"->text("detector","face_yolov8n.pt"),"confidence"->number("confidence",0.3),
      "mask_k_largest"->number("max_faces",0),"mask_shape"->text("mask_shape","ellipse"),"mask_scale"->number("mask_size",1.2),
      "dilate_erode"->0,"mask_blur"->number("mask_blur",8),"denoising_strength"->number("denoising_strength",0.5),
      "inpaint_only_masked"->true,"inpaint_padding"->number("padding",32),"use_inpaint_width_height"->(side>0),
      "inpaint_width"->side,"inpaint_height"->side,"prompt"->text("prompt",""),"negative_prompt"->text("negative_prompt",""),
      "use_steps"->(steps>0),"steps"->steps,"use_cfg_scale"->(cfg>0),"cfg_scale"->cfg,"use_sampler"->sampler.nonEmpty,
      "sampler"->sampler,"color_correction"->h.obj.get("color_correction").fold(true)(_.bool))
    Obj("units"->Arr(unit))

  private def matchColors(output: BufferedImage, reference: BufferedImage, cancelled: () => Boolean): BufferedImage =
    val (w,h) = (output.getWidth,output.getHeight)
    def lab(im: BufferedImage): Array[Int] =
      val colors = im.getRGB(0,0,w,h,null,0,w)
      NativeRgbLab.toLab(colors.flatMap(c => Array((c >>> 16)&255,(c >>> 8)&255,c&255)),cancelled)
    val a=lab(output); val b=lab(reference)
    val count = w.toDouble*h
    def stats(values:Array[Int]) =
      val sums=Array.ofDim[Double](3);val squares=Array.ofDim[Double](3)
      var i=0
      while i<values.length do
        val c=i%3;val v=values(i)
        sums(c)+=v;squares(c)+=v*v;i+=1
      val means=sums.map(_/count)
      (means,Array.tabulate(3)(c=>math.sqrt(math.max(0,squares(c)/count-means(c)*means(c)))))
    val (ma,sa)=stats(a);val (mb,sb)=stats(b)
    for i <- a.indices do
      val c=i%3
      a(i)=((a(i)-ma(c))*sb(c)/math.max(sa(c),1e-6)+mb(c)).toInt.max(0).min(255)
    val rgb=NativeRgbLab.toRgb(a,cancelled)
    val packed=Array.tabulate(w*h)(i=>(rgb(3*i)<<16)|(rgb(3*i+1)<<8)|rgb(3*i+2))
    val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    out.setRGB(0,0,w,h,packed,0,w)
    out

  def apply(engine: VulkanInference, device: String, modelDir: Path, first: BufferedImage, config: Value,
      p: GenerationParams, positive: String, seed: Long, cancelled: () => Boolean, detected: (Int,Int) => Unit,
      refine: Refine): BufferedImage =
    val units = config("units").arr
    require(units.size <= 4,"ADetailer supports up to four detection units")
    def check(): Unit = if cancelled() then throw new java.util.concurrent.CancellationException()
    var canvas=first
    for (u,unit) <- units.zipWithIndex do
      check()
      def str(k:String,d:String) = u.obj.get(k).fold(d)(_.str)
      def flag(k:String,d:Boolean) = u.obj.get(k).fold(d)(_.bool)
      def num(k:String,d:Double,lo:Double,hi:Double):Double =
        val v=u.obj.get(k).fold(d)(_.num)
        require(v.isFinite && v>=lo && v<=hi,s"Invalid ADetailer $k")
        v
      val weights=VulkanYoloWeights.resolve(str("model","face_yolov8n.pt"),modelDir,() => check())
      val confidence=num("confidence",0.3,0,1)
      val min=num("mask_min_ratio",0,0,1);val max=num("mask_max_ratio",1,min,1)
      val top=num("mask_k_largest",0,0,100).toInt
      def quote(s:String)=ujson.write(ujson.Str(s))
      val source=s"""fn main(input) {
        detector = load_detector(${quote(weights.toAbsolutePath.toString)}, backend = ${quote(device)})
        found = detections::detect(detector, input, confidence = $confidence)
        selected = detections::filter(input, found, top_k = $top, min_area_ratio = $min, max_area_ratio = $max, sort_by = "left_to_right")
        return {boxes = selected.xyxy}
      }"""
      val boxes=engine.detectBoxes(source,canvas).grouped(4).toVector
      detected(unit,boxes.size)
      val w=canvas.getWidth;val h=canvas.getHeight
      val x=num("x_offset",0,-200,200).toInt;val y= -num("y_offset",0,-200,200).toInt
      val k=num("dilate_erode",4,-128,128).toInt
      val left = if k>0 then (k-1)/2 else k/2
      val right = if k>0 then k/2 else (k+1)/2
      val shape=str("mask_shape","box")
      val maskScale=num("mask_scale",1,0.1,5)
      require(Set("ellipse","box")(shape),"Unknown face mask shape")
      val rectangles=boxes.flatMap { b =>
        val cx=(b(0)+b(2))/2;val cy=(b(1)+b(3))/2
        val hw=(b(2)-b(0))*maskScale/2;val hh=(b(3)-b(1))*maskScale/2
        val x0=(cx-hw).toInt+x-left;val y0=(cy-hh).toInt+y-left
        val x1=(cx+hw).toInt+x+right;val y1=(cy+hh).toInt+y+right
        val box=new Rectangle(x0,y0,x1-x0+1,y1-y0+1).intersection(new Rectangle(w,h))
        Option.when(!box.isEmpty)(box)
      }
      val merge=str("mask_merge_invert","None")
      require(Set("None","Merge","Merge and Invert")(merge),"Invalid ADetailer mask merge mode")
      val groups= if merge=="None" then rectangles.map(r=>Vector(r)) else Vector(rectangles).filter(_.nonEmpty)
      val only=flag("inpaint_only_masked",true)
      val width=if flag("use_inpaint_width_height",false) then num("inpaint_width",512,64,4096).toInt else p.width
      val height=if flag("use_inpaint_width_height",false) then num("inpaint_height",512,64,4096).toInt else p.height
      val steps=if flag("use_steps",false) then num("steps",28,1,1000).toInt else p.steps
      val guidance=if flag("use_cfg_scale",false) then num("cfg_scale",7,0,100) else p.cfgScale
      val sampler=if flag("use_sampler",false) then str("sampler",p.samplerName) else p.samplerName
      val denoise=num("denoising_strength",0.4,0,1)
      val blur=num("mask_blur",4,0,64).toInt
      val pad=num("inpaint_padding",32,0,4096).toInt
      def prompt(key:String,main:String,idx:Int)=
        val text=Option(str(key,"").trim).filter(_.nonEmpty).getOrElse(main)
        val parts=text.split("\\[SEP\\]",-1)
        parts(math.min(idx,parts.length-1)).trim.replace("[PROMPT]",main)
      for (rects,idx) <- groups.zipWithIndex if denoise>0 do
        check()
        val mask=new BufferedImage(w,h,BufferedImage.TYPE_BYTE_GRAY)
        val graphics=mask.createGraphics()
        try
          val invert=merge=="Merge and Invert"
          graphics.setColor(if invert then Color.WHITE else Color.BLACK)
          graphics.fillRect(0,0,w,h)
          graphics.setColor(if invert then Color.BLACK else Color.WHITE)
          rects.foreach { r =>
            if shape=="ellipse" then graphics.fillOval(r.x,r.y,r.width,r.height)
            else graphics.fillRect(r.x,r.y,r.width,r.height)
          }
        finally graphics.dispose()
        val ip=Img2ImgParams("",denoisingStrength=denoise,maskBlur=blur,inpaintFullRes=only,inpaintFullResPadding=pad)
        val prepared=VulkanImageInputs.prepare(ip,width,height,seed+idx,Some(canvas -> mask),cancelled)
        val options=new VulkanInference.ImageOptions(prepared.init,prepared.mask.orNull,denoise,null)
        val result=refine(prompt("prompt",positive,idx),prompt("negative_prompt",p.negativePrompt,idx),
          width,height,steps,guidance,sampler,seed+idx,options)
        val corrected = if flag("color_correction",true) then matchColors(result,prepared.init,cancelled) else result
        val overlay = if prepared.region.nonEmpty then prepared
          else prepared.copy(original=canvas,region=Some(new Rectangle(w,h)))
        canvas=overlay.finish(corrected)
    canvas
