package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.US_ASCII
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import sdfx.vulkan.VulkanProtocol
import ujson.{Arr, Obj, Value}

private[vulkan] object VulkanCnAnimation:
  private def text(p: Value,k: String,d: String="") = p.obj.get(k).flatMap(_.strOpt).filter(_.nonEmpty).getOrElse(d)
  private def number(p: Value,k: String,d: Double,lo: Double,hi: Double): Double =
    val value=p.obj.get(k).fold(d)(_.num)
    require(value.isFinite && value>=lo && value<=hi,s"$k must be within $lo..$hi")
    value
  private def integer(p: Value,k: String,d: Int,lo: Int,hi: Int): Int =
    val v=number(p,k,d,lo,hi)
    require(v==v.toInt,s"$k must be a whole number")
    v.toInt
  private def path(p: Value,k: String): Path =
    require(text(p,k).nonEmpty,s"$k is required")
    Path.of(text(p,k)).toAbsolutePath.normalize()
  private def image(p: Value,k: String) = DiffusionImageInput.readRgb(path(p,k))
  private def write(im: BufferedImage,p: Path): Unit =
    Files.createDirectories(p.getParent)
    require(ImageIO.write(im,"png",p.toFile),"PNG writer unavailable")
  private def reflect(i: Int,n: Int,repeat: Boolean): Int =
    if n==1 then return 0
    val period=if repeat then 2*n else 2*(n-1)
    val k=Math.floorMod(i,period)
    if k<n then k else period-k-(if repeat then 1 else 0)
  private def pixels(im: BufferedImage) = im.getRGB(0,0,im.getWidth,im.getHeight,null,0,im.getWidth)
  private def packed(w: Int,h: Int,rgb: Array[Int]) =
    val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    out.setRGB(0,0,w,h,rgb,0,w)
    out
  private val shifts=Array(16,8,0)
  private def channel(rgb: Int,s: Int) = (rgb>>>s)&255
  private def compose(f: Int => Double): Int = shifts.foldLeft(0)((rgb,s)=>rgb | (f(s).toInt.max(0).min(255)<<s))
  private def verify(im: BufferedImage,w: Int,h: Int): Unit =
    require(im.getWidth==w && im.getHeight==h,"Frames have different dimensions")
  private def alphaFile(file: Path,w: Int,h: Int): Array[Float] =
    val n=w*h
    require(Files.size(file)<=4L*n+65546,"Alpha file exceeds expected size")
    val buf=ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN)
    require(buf.remaining>=10 && buf.get()==0x93.toByte && (0 until 5).map(_=>buf.get().toChar).mkString=="NUMPY" &&
      buf.get()==1 && buf.get()==0,"Expected a NumPy v1 alpha image")
    val len=buf.getShort() & 65535
    require(buf.remaining==len+4*n,"Alpha array size mismatch")
    val head=new Array[Byte](len)
    buf.get(head)
    val header=new String(head,US_ASCII).replace(" ","")
    require(header.contains("'descr':'<f4'") && header.contains("'fortran_order':False") &&
      header.contains(s"'shape':($h,$w)"),"Invalid alpha type or shape")
    val out=new Array[Float](n)
    buf.asFloatBuffer().get(out)
    require(out.forall(_.isFinite),"Non-finite alpha")
    out
  private def writeAlpha(file: Path,w: Int,h: Int,data: Array[Float]): Unit =
    val meta=s"{'descr': '<f4', 'fortran_order': False, 'shape': ($h, $w), }"
    val header=(meta+" "*Math.floorMod(-11-meta.length,64)+"\n").getBytes(US_ASCII)
    val buf=ByteBuffer.allocate(10+header.length+4*data.length).order(ByteOrder.LITTLE_ENDIAN)
    buf.put(0x93.toByte).put("NUMPY".getBytes(US_ASCII)).put(1.toByte).put(0.toByte).putShort(header.length.toShort).put(header)
    buf.asFloatBuffer().put(data)
    Files.write(file,buf.array())

  def run(p: Value,device: String,ctx: NativeWorkContext): Value =
    ctx.checkCancelled()
    text(p,"operation") match
      case "release" => Obj("released"->true)
      case "encode" =>
        val args=Obj.from(p.obj)
        args("closed_loop")="N"
        VulkanVideo.encode(args,ctx)
      case "prepare" =>
        val w=integer(p,"width",512,64,2048)
        val h=integer(p,"height",512,64,2048)
        require(w%8==0 && h%8==0,"Frame dimensions must be multiples of 8")
        val limit=integer(p,"max_frames",0,0,5000) match
          case 0 => 5000
          case x => x
        val stride=1+integer(p,"frame_skip",0,0,100)
        val dir=Files.createDirectories(path(p,"output_dir"))
        val frames=Arr()
        var i=0
        val meta=VulkanVideoExtract.run(path(p,"video"),0,frame=>{
          if i%stride==0 then
            val file=dir.resolve(f"source-${frames.arr.size}%05d.png")
            val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
            val gfx=out.createGraphics()
            try
              if frame.getWidth>w || frame.getHeight>h then
                gfx.drawImage(frame.getScaledInstance(w,h,java.awt.Image.SCALE_AREA_AVERAGING),0,0,null)
              else
                gfx.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
                gfx.drawImage(frame,0,0,w,h,null)
            finally gfx.dispose()
            write(out,file)
            frames.value+=ujson.Str(file.toString)
          i+=1
        },()=>ctx.checkCancelled(),(_,_)=>(),(limit-1)*stride+1)
        Obj("frames"->frames,"source_size"->meta("dimensions"),"fps"->(meta("fps").num/stride),
          "source_fps"->meta("fps"),"source_frames"->meta("source_frames"),"decoded"->meta("decoded"))
      case "flow" => flow(p,device,ctx)
      case "finish" => finish(p,ctx)
      case other => throw IllegalArgumentException(s"Unknown cn_animation operation $other")

  private def blur(raw: Array[Float],w: Int,h: Int,sigma: Double): Array[Float] =
    val radius=((w.min(h)/15)|1)/2
    if sigma==0 || radius==0 then return raw
    val kernel=(-radius to radius).map(x=>math.exp(-x*x/(2*sigma*sigma))).toArray
    val sum=kernel.sum
    for j<-kernel.indices do kernel(j)/=sum
    def pass(src: Array[Float],vertical: Boolean) = Array.tabulate(w*h){ i =>
      var value=0.0
      for j<-kernel.indices do
        val pos=if vertical then reflect(i/w+j-radius,h,true)*w+i%w else i/w*w+reflect(i%w+j-radius,w,true)
        value+=kernel(j)*src(pos)
      value.toFloat
    }
    pass(pass(raw,false),true)

  private def flow(p: Value,device: String,ctx: NativeWorkContext): Value =
    val previous=image(p,"prev_frame")
    val current=image(p,"cur_frame")
    val styled=image(p,"prev_styled")
    val w=current.getWidth
    val h=current.getHeight
    val n=w*h
    verify(previous,w,h)
    verify(styled,w,h)
    require(w<=2048 && h<=2048,"RAFT dimensions exceed 2048")
    val kind=text(p,"raft_model","large")
    require(Set("small","large")(kind),"Unknown RAFT variant")
    val iters=integer(p,"raft_iters",20,1,64)
    val sigma=number(p,"occlusion_blur",3,0,10)
    val flowMult=number(p,"flow_multiplier",5,0,10)
    val difo=number(p,"difo_multiplier",2,0,10)
    val difs=number(p,"difs_multiplier",0,0,10)
    val blend=number(p,"blend_alpha",1,0,1)
    val trailing=p.obj.get("trailing").forall(_.bool)
    val prior=Option.when(text(p,"prev_alpha").nonEmpty)(alphaFile(path(p,"prev_alpha"),w,h))
    val bin=Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR",".")))
    val folder=text(p,"raft_weights") match
      case "" => Path.of(sys.env.getOrElse("SDFX_VULKAN_RAFT_DIR",bin.toString)).resolve(s"raft-$kind")
      case custom => Path.of(custom)
    val bridge=bin.resolve("sdv-ncnn.exe").toAbsolutePath
    if !Files.isRegularFile(bridge) || !Seq("encoder","update").forall(part=>
      Seq("param","bin").forall(ext=>Files.isRegularFile(folder.resolve(s"$part.ncnn.$ext")))) then
      VulkanProtocol.unsupported(s"RAFT $kind requires the ncnn bridge and converted RAFT weights in $folder (scripts/build_vulkan_raft.py)")
    val model=RaftFlow(w,h,kind=="small",device,bridge,folder.toAbsolutePath,()=>ctx.checkCancelled())
    val a=model.encode(previous)
    val b=model.encode(current)
    val forward=model.predict(a,b,iters,w,h)
    val backward=model.predict(b,a,iters,w,h)
    val prev=pixels(previous)
    val cur=pixels(current)
    val style=pixels(styled)
    val warped=new Array[Int](n)
    var flowSum=0.0
    val raw=Array.tabulate(n){ i =>
      val xx=reflect(math.rint(i%w+backward(i)).toInt,w,false)
      val yy=reflect(math.rint(i/w+backward(n+i)).toInt,h,false)
      val j=yy*w+xx
      warped(i)=style(j)
      def distance(other: Int)=shifts.iterator.map(s=>math.abs(channel(cur(i),s)-channel(other,s))).max/255.0
      val fx=forward(i)/w
      val fy=forward(n+i)/h
      val bx=backward(i)/w
      val by=backward(n+i)/h
      val consistency=math.hypot(fx+bx,fy+by)*(1-20*math.hypot(bx,by)).max(0)
      flowSum+=math.hypot(forward(i),forward(n+i))
      math.max(consistency*10*flowMult,math.max(distance(prev(j))*difo,distance(style(j))*difs)).toFloat
    }
    val alpha=blur(raw,w,h,sigma)
    if trailing then prior.foreach(old=>alpha.indices.foreach(i=>alpha(i)+=old(i)*0.5f))
    val mask=new BufferedImage(w,h,BufferedImage.TYPE_BYTE_GRAY)
    val init=new Array[Int](n)
    var maskSum=0.0
    var maskMax=0.0
    for i<-0 until n do
      val t=alpha(i).max(0f).min(1f)
      val old=warped(i)
      def fixed(s: Int)=channel(cur(i),s)*t+channel(old,s)*(1f-t)
      mask.getRaster.setSample(i%w,i/w,0,(t*255).toInt)
      init(i)=compose(s=>fixed(s)*(1-blend)+channel(cur(i),s)*blend)
      warped(i)=compose(s=>fixed(s).toDouble)
      maskSum+=t
      maskMax=maskMax.max(t)
    val dir=Files.createDirectories(path(p,"output_dir"))
    val name=text(p,"name","frame")
    require(name.length<=160 && !name.exists(c=>c<' ' || "\\/:*?\"<>|".contains(c)),"Invalid frame name")
    def out(suffix: String) = dir.resolve(s"$name-$suffix")
    ctx.beginPublication()
    write(mask,out("mask.png"))
    write(packed(w,h,warped),out("warped.png"))
    write(packed(w,h,init),out("init.png"))
    writeAlpha(out("alpha.npy"),w,h,alpha)
    Obj("mask"->out("mask.png").toString,"init"->out("init.png").toString,"warped"->out("warped.png").toString,
      "alpha"->out("alpha.npy").toString,"flow_mean"->(flowSum/n),"mask_mean"->(maskSum/n),"mask_max"->maskMax)

  private def finish(p: Value,ctx: NativeWorkContext): Value =
    val input=image(p,"image")
    val reference=image(p,"reference")
    val w=input.getWidth
    val h=input.getHeight
    verify(reference,w,h)
    val src=pixels(input)
    def histogram(values: Array[Int]) =
      val hist=new Array[Int](256)
      values.foreach(rgb=>shifts.foreach(s=>hist(channel(rgb,s))+=1))
      hist
    val srcHist=histogram(src)
    val refHist=histogram(pixels(reference))
    val bins=refHist.indices.filter(refHist(_)>0).toVector
    val cumulative=bins.scanLeft(0)((sum,j)=>sum+refHist(j)).tail
    var srcTotal=0
    val lut=srcHist.map{ count =>
      srcTotal+=count
      val hi=cumulative.indexWhere(_>=srcTotal)
      if hi<=0 then bins.head.toFloat
      else bins(hi-1)+(bins(hi)-bins(hi-1))*(srcTotal-cumulative(hi-1)).toFloat/(cumulative(hi)-cumulative(hi-1))
    }
    val overlay=Option.when(text(p,"alpha").nonEmpty){
      val warped=image(p,"warped")
      verify(warped,w,h)
      (pixels(warped),alphaFile(path(p,"alpha"),w,h))
    }
    val out=Array.tabulate(src.length){ i =>
      compose{ s =>
        val matched=lut(channel(src(i),s))
        overlay.fold(matched){ (v,alpha) =>
          val t=alpha(i).max(0f).min(1f)
          matched*t+channel(v(i),s)*(1f-t)
        }.toDouble
      }
    }
    val file=path(p,"output")
    ctx.beginPublication()
    write(packed(w,h,out),file)
    Obj("path"->file.toString)
