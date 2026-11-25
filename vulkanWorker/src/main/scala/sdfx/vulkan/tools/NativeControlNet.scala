// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import java.awt.image.BufferedImage

private[tools] object NativeControlNet:
  import NativeJson.*
  private def canny(image: BufferedImage,low: Int,high: Int,cancelled: () => Boolean): BufferedImage=
    val width=image.getWidth;val height=image.getHeight
    val (lowThreshold,highThreshold)=if low<=high then (low,high) else (high,low)
    val count=Math.multiplyExact(width,height)
    val magnitude=new Array[Int](count)
    val dxs=new Array[Int](count);val dys=new Array[Int](count)
    def rgb(x: Int,y: Int): Int=image.getRGB(math.max(0,math.min(width-1,x)),math.max(0,math.min(height-1,y)))
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet Canny preprocessing cancelled")
      for x <- 0 until width do
        val i=y*width+x
        val p0=rgb(x-1,y-1);val p1=rgb(x,y-1);val p2=rgb(x+1,y-1)
        val p3=rgb(x-1,y);val p4=rgb(x+1,y)
        val p5=rgb(x-1,y+1);val p6=rgb(x,y+1);val p7=rgb(x+1,y+1)
        var strongest=0;var selectedX=0;var selectedY=0
        var channel=0
        while channel<3 do
          val shift=16-channel*8
          val a=(p0>>>shift)&255;val b=(p1>>>shift)&255;val c=(p2>>>shift)&255
          val d=(p3>>>shift)&255;val f=(p4>>>shift)&255
          val g=(p5>>>shift)&255;val h=(p6>>>shift)&255;val j=(p7>>>shift)&255
          val gx= -a+c-2*d+2*f-g+j
          val gy= -a-2*b-c+g+2*h+j
          val strength=math.abs(gx)+math.abs(gy)
          if strength>strongest then
            strongest=strength
            selectedX=gx
            selectedY=gy
          channel+=1
        dxs(i)=selectedX;dys(i)=selectedY;magnitude(i)=strongest
    val suppressed=new Array[Int](count)
    def magnitudeAt(x: Int,y: Int): Int=if x<0 || x>=width || y<0 || y>=height then 0 else magnitude(y*width+x)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet Canny preprocessing cancelled")
      for x <- 0 until width do
        val i=y*width+x;val gx=dxs(i);val gy=dys(i)
        val horizontal=math.abs(gy)*1000<=math.abs(gx)*414
        val vertical=math.abs(gx)*1000<=math.abs(gy)*414
        val (before,after)=
          if horizontal then (magnitudeAt(x-1,y),magnitudeAt(x+1,y))
          else if vertical then (magnitudeAt(x,y-1),magnitudeAt(x,y+1))
          else if (gx<0)==(gy<0) then (magnitudeAt(x-1,y-1),magnitudeAt(x+1,y+1))
          else (magnitudeAt(x+1,y-1),magnitudeAt(x-1,y+1))
        val value=magnitude(i)
        if value>before && (if horizontal || vertical then value>=after else value>after) then suppressed(i)=value
    val edges=new Array[Boolean](count)
    val queue=new java.util.ArrayDeque[Int]()
    var i=0
    while i<count do
      if suppressed(i)>highThreshold then
        edges(i)=true
        queue.addLast(i)
      i+=1
    var visits=0
    while !queue.isEmpty do
      if (visits&8191)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
        throw new java.util.concurrent.CancellationException("ControlNet Canny preprocessing cancelled")
      visits+=1
      val current=queue.removeFirst();val x=current%width;val y=current/width
      var dy= -1
      while dy<=1 do
        var dx= -1
        while dx<=1 do
          val nx=x+dx;val ny=y+dy
          if nx>=0 && nx<width && ny>=0 && ny<height then
            val next=ny*width+nx
            if !edges(next) && suppressed(next)>lowThreshold then
              edges(next)=true
              queue.addLast(next)
          dx+=1
        dy+=1
    val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    i=0
    while i<count do
      output.setRGB(i%width,i/width,if edges(i) then 0xffffffff.toInt else 0xff000000.toInt)
      i+=1
    output

  private def sourceCanny(image: BufferedImage,resolution: Int,low: Int,high: Int,
      cancelled: () => Boolean): BufferedImage=
    require(resolution>0,"ControlNet Canny resolution must be positive")
    val scale=resolution.toDouble/math.min(image.getWidth,image.getHeight)
    val width=math.max(1,math.rint(image.getWidth*scale).toInt)
    val height=math.max(1,math.rint(image.getHeight*scale).toInt)
    val interpolation=if scale>1.0 then "cubic" else "area"
    val resized=NativeControlNetImageOps.cvResize(image,width,height,interpolation,cancelled)
    val paddedWidth=((width+63)/64)*64
    val paddedHeight=((height+63)/64)*64
    require(paddedWidth.toLong*paddedHeight<=16777216L,"ControlNet Canny padded image exceeds pixel limit")
    if paddedWidth==width && paddedHeight==height then canny(resized,low,high,cancelled)
    else
      val padded=new BufferedImage(paddedWidth,paddedHeight,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until paddedHeight do
        if cancelled() || Thread.currentThread().isInterrupted then
          throw new java.util.concurrent.CancellationException("ControlNet Canny padding cancelled")
        val sy=math.min(y,height-1)
        for x <- 0 until paddedWidth do padded.setRGB(x,y,resized.getRGB(math.min(x,width-1),sy))
      val edges=canny(padded,low,high,cancelled)
      val cropped=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until height; x <- 0 until width do cropped.setRGB(x,y,edges.getRGB(x,y))
      cropped

  private def recolor(image: BufferedImage,luminance: Boolean,gamma: Double,cancelled: () => Boolean): BufferedImage=
    require(gamma.isFinite,"ControlNet recolor gamma must be finite")
    val output=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until image.getHeight do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet recolor preprocessing cancelled")
      for x <- 0 until image.getWidth do
        val pixel=image.getRGB(x,y)
        val red=((pixel>>>16)&255)/255f
        val green=((pixel>>>8)&255)/255f
        val blue=(pixel&255)/255f
        val intensity=if luminance then 0.2126f*red+0.7152f*green+0.0722f*blue else math.max(red,math.max(green,blue))
        val corrected=math.pow(intensity.toDouble,gamma)*255.0
        val level=if corrected.isNaN then 0 else math.max(0,math.min(255,corrected.toInt))
        output.setRGB(x,y,0xff000000.toInt|(level<<16)|(level<<8)|level)
    output

  /** Preprocess without loading ControlNet weights, with NUI standalone resolution semantics. */
  def preprocessImage(name: String,source: BufferedImage,args: Map[String,NativeJson],standalone: Boolean,
      resolution: Int,cancelled: () => Boolean): BufferedImage =
    def numeric(key: String,default: Double): Double=args.get(key) match
      case None => default
      case Some(Number(value)) if value.toDouble.isFinite => value.toDouble
      case _ => throw IllegalArgumentException(s"ControlNet $key must be finite and numeric")
    name match
      case "none" | "inpaint" => source
      case "invert" =>
        val output=new BufferedImage(source.getWidth,source.getHeight,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until source.getHeight do
          if cancelled() then throw new java.util.concurrent.CancellationException("preprocess cancelled")
          for x <- 0 until source.getWidth do output.setRGB(x,y,0xff000000.toInt | (~source.getRGB(x,y)&0xffffff))
        output
      case "canny" =>
        val low=numeric("low_threshold",100).toInt;val high=numeric("high_threshold",200).toInt
        if standalone then sourceCanny(source,resolution,if low>=1 && low<=255 then low else 100,
          if high>=1 && high<=255 then high else 200,cancelled)
        else canny(source,low,high,cancelled)
      case "recolor_luminance" | "recolor_intensity" =>
        recolor(source,name=="recolor_luminance",numeric("gamma_correction",1.0),cancelled)
      case "tile_resample" =>
        val down=numeric("down_sampling_rate",1.0)
        if down<=1 then source else
          val small=NativeControlNetImageOps.bicubic(source,math.max(1,(source.getWidth/down).toInt),
            math.max(1,(source.getHeight/down).toInt),cancelled)
          NativeControlNetImageOps.bicubic(small,source.getWidth,source.getHeight,cancelled)
      case "tile_colorfix" => NativeControlNetImageOps.gaussian(source,math.max(1,numeric("variation",8).toInt).toDouble,cancelled)
      case "blur_gaussian" => NativeControlNetImageOps.gaussian(source,numeric("sigma",9.0),cancelled)
      case other => throw new NativeBackendError("unsupported",s"preprocessor '$other' is not available")
