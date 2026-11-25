package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import java.util.concurrent.{CompletableFuture,TimeUnit}
import scala.util.Using
import sdfx.core.imagebrowser.BrowserFiles

private[tools] object VulkanCharaMasks:
  private val hashes=Map("person"->("a7324706595da41a484aac7c140a68c60ad9805457e1c084454fa6eaa041bb17","e0d4e89c822c1915b431abab6b59945911a99b898674bc373568276e8d638bf3"),
    "seg"->("2e6dbb845310177b3ce3f6858d8f876aba8e20a10c4118a324ff60f65bc269d6","3823cf7a137b09410b7e9d356617791f15a9a9a32bbf8ef4fd021c3985390abb"))
  private def bin=Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR","."))).toAbsolutePath
  private def models=sys.env.get("SDFX_VULKAN_CHARA_DIR").map(Path.of(_)).getOrElse(bin.resolve("chara-searcher")).toAbsolutePath
  def available:Boolean=Files.isRegularFile(bin.resolve("sdv-ncnn.exe")) && hashes.keys.forall(k=>Vector("bin","param").forall(x=>Files.isRegularFile(models.resolve(s"$k.ncnn.$x"))))
  def verify():Unit =
    require(available,"Character import requires converted anime person detection and ISNet models in SDFX_VULKAN_CHARA_DIR")
    hashes.foreach { (key,pair) =>
      require(BrowserFiles.sha256(models.resolve(s"$key.ncnn.param"))==pair._1 && BrowserFiles.sha256(models.resolve(s"$key.ncnn.bin"))==pair._2,s"Incorrect character model: $key")
    }
  private case class Box(x:Double,y:Double,r:Double,b:Double,score:Double):
    def area:Double=(r-x)*(b-y)
    def iou(other:Box):Double =
      val intersect=(r.min(other.r)-x.max(other.x)).max(0)*(b.min(other.b)-y.max(other.y)).max(0)
      intersect/(area+other.area-intersect).max(1e-9)
    def distance(px:Double,py:Double,pad:Double):Double =
      val dx=(px-(x+r)/2).abs/((r-x)/2).max(1)
      val dy=(py-(y+b)/2).abs/((b-y)/2).max(1)
      if dx<=1+pad && dy<=1+pad then dx*dx+dy*dy else Double.PositiveInfinity
  private case class Input(bytes:Array[Byte],scale:Double,left:Int,top:Int,width:Int,height:Int)
  private def input(image:BufferedImage,size:Int,pad:Int):Input =
    val scale=size.toDouble/image.getWidth.max(image.getHeight)
    val w=Math.rint(image.getWidth*scale).toInt.max(1);val h=Math.rint(image.getHeight*scale).toInt.max(1)
    val left=(size-w)/2;val top=(size-h)/2
    val canvas=new BufferedImage(size,size,BufferedImage.TYPE_INT_RGB)
    val g=canvas.createGraphics()
    try
      g.setColor(new java.awt.Color(pad,pad,pad));g.fillRect(0,0,size,size)
      g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      val opaque=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
      opaque.setRGB(0,0,image.getWidth,image.getHeight,image.getRGB(0,0,image.getWidth,image.getHeight,null,0,image.getWidth),0,image.getWidth)
      g.drawImage(opaque,left,top,w,h,null)
    finally g.dispose()
    val pixels=canvas.getRGB(0,0,size,size,null,0,size)
    val bytes=ByteBuffer.allocate(12*size*size).order(ByteOrder.LITTLE_ENDIAN)
    for shift<-Seq(16,8,0); rgb<-pixels do bytes.putFloat(((rgb>>>shift)&255)/255f)
    Input(bytes.array(),scale,left,top,w,h)
  private def predict(name:String,device:String,size:Int,input:Input,count:Int,check:()=>Unit):Array[Float] =
    val proc=new ProcessBuilder(bin.resolve("sdv-ncnn.exe").toString,device,models.resolve(name+".ncnn.param").toString,models.resolve(name+".ncnn.bin").toString,
      "in0","out0",size.toString,size.toString,"3").redirectError(ProcessBuilder.Redirect.INHERIT).start()
    try
      val io=CompletableFuture.supplyAsync(()=>{Using.resource(proc.getOutputStream)(_.write(input.bytes));Using.resource(proc.getInputStream)(_.readNBytes(count*4+1))})
      while !proc.waitFor(100,TimeUnit.MILLISECONDS) do check()
      check()
      require(proc.exitValue()==0,s"Character $name inference failed")
      val data=io.get(10,TimeUnit.SECONDS)
      require(data.length==count*4,s"Unexpected $name tensor")
      val arr=new Array[Float](count)
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer.get(arr)
      require(arr.forall(_.isFinite),"Nonfinite model output")
      arr
    finally if proc.isAlive then {proc.destroyForcibly();proc.waitFor()}
  def extract(image:BufferedImage,device:String,threshold:Double,maskThreshold:Double,pad:Double,check:()=>Unit):Vector[BufferedImage] =
    val det=input(image,640,114)
    val n=8400
    val raw=predict("person",device,640,det,n*5,check)
    val candidates=(0 until n).filter(i=>raw(4*n+i)>=threshold).map { i =>
      val x=(raw(i)-det.left)/det.scale;val y=(raw(n+i)-det.top)/det.scale
      val w=raw(2*n+i)/(2*det.scale);val h=raw(3*n+i)/(2*det.scale)
      Box((x-w).max(0),(y-h).max(0),(x+w).min(image.getWidth),(y+h).min(image.getHeight),raw(4*n+i))
    }.sortBy(-_.score)
    val picked=scala.collection.mutable.ArrayBuffer.empty[Box]
    candidates.foreach(b=>if picked.forall(_.iou(b)<=.7) then picked+=b)
    val boxes=picked.filter(b=>b.r-b.x>=8 && b.b-b.y>=8).sortBy(_.x).toVector
    boxes.zipWithIndex.flatMap { (box,index) =>
      val bw=box.r-box.x;val bh=box.b-box.y
      val x=(box.x-pad*bw).toInt.max(0);val y=(box.y-pad*bh).toInt.max(0)
      val w=(box.r+pad*bw+.999).toInt.min(image.getWidth)-x;val h=(box.b+pad*bh+.999).toInt.min(image.getHeight)-y
      val crop=image.getSubimage(x,y,w,h)
      val seg=input(crop,1024,0)
      val probs=predict("seg",device,1024,seg,1024*1024,check)
      val alpha=new Array[Float](w*h)
      for j<-0 until h; i<-0 until w do
        val u=((i+.5)*seg.width/w-.5).max(0).min(seg.width-1)+seg.left
        val v=((j+.5)*seg.height/h-.5).max(0).min(seg.height-1)+seg.top
        val u0=u.toInt;val v0=v.toInt;val u1=(u0+1).min(1023);val v1=(v0+1).min(1023)
        val dx=u-u0;val dy=v-v0
        alpha(j*w+i)=((1-dy)*((1-dx)*probs(v0*1024+u0)+dx*probs(v0*1024+u1))+dy*((1-dx)*probs(v1*1024+u0)+dx*probs(v1*1024+u1))).toFloat
      isolate(alpha,w,h,box.x-x,box.y-y,box.r-x,box.b-y)
      val result=new BufferedImage(w,h,BufferedImage.TYPE_INT_ARGB)
      var visible=0
      for j<-0 until h; i<-0 until w do
        val owns=boxes.size==1 || {
          val ds=boxes.map(_.distance(x+i+.5,y+j+.5,pad))
          ds(index).isFinite && ds.indexOf(ds.min)==index
        }
        val pixel=crop.getRGB(i,j)
        val a=if owns then Math.rint(((alpha(j*w+i)-maskThreshold+.15)/.3).max(0).min(1)*(pixel>>>24)).toInt else 0
        if a>127 then visible+=1
        result.setRGB(i,j,(pixel&0xffffff)|(a<<24))
      Option.when(visible>=.02*w*h)(result)
    }
  private def isolate(alpha:Array[Float],w:Int,h:Int,x1:Double,y1:Double,x2:Double,y2:Double):Unit =
    val labels=new Array[Int](alpha.length);val queue=new Array[Int](alpha.length)
    val keep=scala.collection.mutable.ArrayBuffer(false)
    for i<-alpha.indices if alpha(i)>.5f && labels(i)==0 do
      val id=keep.size;var head=0;var tail=1;var inside=0
      queue(0)=i;labels(i)=id
      while head<tail do
        val p=queue(head);head+=1
        val x=p%w;val y=p/w
        if x>=Math.rint(x1) && x<Math.rint(x2) && y>=Math.rint(y1) && y<Math.rint(y2) then inside+=1
        for v<-(y-1).max(0) to (y+1).min(h-1); u<-(x-1).max(0) to (x+1).min(w-1) do
          val k=v*w+u
          if labels(k)==0 && alpha(k)>.5f then {queue(tail)=k;tail+=1;labels(k)=id}
      keep+=(inside.toLong*5>=tail)
    if keep.size>2 && keep.contains(true) then
      for i<-alpha.indices if !keep(labels(i)) do
        val x=i%w;val y=i/w
        if !((y-2).max(0) to (y+2).min(h-1)).exists(v=>((x-2).max(0) to (x+2).min(w-1)).exists(u=>keep(labels(v*w+u)))) then alpha(i)=0f
