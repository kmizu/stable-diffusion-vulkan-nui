// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import java.awt.image.BufferedImage

private[vulkan] object NativeInpaint:
  def gaussianBytes(input: Array[Int],width: Int,height: Int,radius: Int,cancelled: () => Boolean): Array[Int] =
    if radius==0 then input
    else
      // Pillow's three fractional box passes, including byte rounding at every pass.
      // See Pillow BoxBlur.c; its license is in META-INF/licenses/pillow.txt.
      val sigma2=radius.toFloat*radius.toFloat/3f
      val length=math.sqrt(12.0*sigma2+1.0).toFloat
      val whole=math.floor((length-1.0)/2.0).toFloat
      val fraction=((2f*whole+1f)*(whole*(whole+1f)-3f*sigma2))/(6f*(sigma2-(whole+1f)*(whole+1f)))
      val boxRadius=(whole+fraction).toFloat
      val integerRadius=boxRadius.toInt
      val scale=1L<<24
      val weight=(scale.toFloat/(boxRadius*2f+1f)).toLong
      val edgeWeight=(scale-(integerRadius*2+1)*weight)/2
      var pixels=input
      def pass(horizontal: Boolean): Unit =
        val output=new Array[Int](pixels.length)
        val lines=if horizontal then height else width
        val count=if horizontal then width else height
        for line <- 0 until lines do
          if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("inpainting cancelled")
          def index(position: Int): Int =
            val at=math.max(0,math.min(count-1,position))
            if horizontal then line*width+at else at*width+line
          var sum=0L
          for position <- -integerRadius to integerRadius do sum+=pixels(index(position))
          for position <- 0 until count do
            val far=pixels(index(position-integerRadius-1))+pixels(index(position+integerRadius+1))
            output(index(position))=((sum*weight+far.toLong*edgeWeight+(1L<<23))>>24).toInt
            sum+=pixels(index(position+integerRadius+1))-pixels(index(position-integerRadius))
        pixels=output
      for _ <- 0 until 3 do pass(horizontal=true)
      for _ <- 0 until 3 do pass(horizontal=false)
      pixels

  private final class PythonRandom(seed: Long):
    private val n=624
    private val state=new Array[Int](n)
    private var index=n
    private def initialize(seed: Long): Unit =
      var value=BigInt(seed).abs
      val key=scala.collection.mutable.ArrayBuffer.empty[Int]
      while value>0 do
        key+= (value & BigInt("ffffffff",16)).toLong.toInt
        value=value>>32
      if key.isEmpty then key+=0
      state(0)=19650218
      for i <- 1 until n do state(i)=1812433253*(state(i-1)^(state(i-1)>>>30))+i
      var i=1;var j=0
      var k=math.max(n,key.size)
      while k>0 do
        state(i)=(state(i)^((state(i-1)^(state(i-1)>>>30))*1664525))+key(j)+j
        i+=1;j+=1
        if i>=n then { state(0)=state(n-1);i=1 }
        if j>=key.size then j=0
        k-=1
      k=n-1
      while k>0 do
        state(i)=(state(i)^((state(i-1)^(state(i-1)>>>30))*1566083941))-i
        i+=1
        if i>=n then { state(0)=state(n-1);i=1 }
        k-=1
      state(0)=0x80000000
      index=n
    initialize(seed)
    private def word(): Int =
      if index>=n then
        for i <- 0 until n do
          val y=(state(i)&0x80000000)|(state((i+1)%n)&0x7fffffff)
          state(i)=state((i+397)%n)^(y>>>1)^(if (y&1)!=0 then 0x9908b0df else 0)
        index=0
      var y=state(index);index+=1
      y^=y>>>11
      y^=(y<<7)&0x9d2c5680
      y^=(y<<15)&0xefc60000
      y^=y>>>18
      y
    def bytes(length: Int,cancelled: () => Boolean=() => false): Array[Byte] =
      require(length>=0 && length<=50331648,"inpaint noise exceeds allocation limit")
      val output=new Array[Byte](length)
      var offset=0
      while offset<length do
        if (offset&65535)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
          throw new java.util.concurrent.CancellationException("inpainting noise generation cancelled")
        val count=math.min(4,length-offset)
        val bits=if count==4 then word() else word() >>> (32-count*8)
        for byte <- 0 until count do output(offset+byte)=(bits >>> (byte*8)).toByte
        offset+=count
      output

  /** Pillow-compatible multiscale premultiplied-alpha fill. */
  private def blurredColorFill(image: BufferedImage,mask: BufferedImage,cancelled: () => Boolean): BufferedImage =
    require(image.getWidth==mask.getWidth && image.getHeight==mask.getHeight,"inpaint fill image/mask size mismatch")
    val width=image.getWidth;val height=image.getHeight;val count=width*height
    val pixels=image.getRGB(0,0,width,height,null,0,width)
    val levels=mask.getRGB(0,0,width,height,null,0,width)
    val premultiplied=Array.fill(4)(new Array[Int](count))
    def div255(value: Int): Int =
      val rounded=value+128
      ((rounded>>>8)+rounded)>>>8
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("inpainting blur-fill preparation cancelled")
      for x <- 0 until width do
        val i=y*width+x
        val alpha=255-(levels(i)&255)
        premultiplied(0)(i)=div255(((pixels(i)>>>16)&255)*alpha)
        premultiplied(1)(i)=div255(((pixels(i)>>>8)&255)*alpha)
        premultiplied(2)(i)=div255((pixels(i)&255)*alpha)
        premultiplied(3)(i)=alpha
    val result=Array.fill(4)(new Array[Int](count))
    val radii=Vector((256,1),(64,1),(16,2),(4,4),(2,2),(0,1))
    for (radius,repeats) <- radii do
      val blurred=if radius==0 then premultiplied else premultiplied.map(channel => gaussianBytes(channel,width,height,radius,cancelled))
      for _ <- 0 until repeats do
        for y <- 0 until height do
          if cancelled() || Thread.currentThread().isInterrupted then
            throw new java.util.concurrent.CancellationException("inpainting blur-fill cancelled")
          for x <- 0 until width do
            val i=y*width+x
            val alpha=blurred(3)(i)
            if alpha>0 then
              val outAlpha255=alpha*255+result(3)(i)*(255-alpha)
              val foreground=alpha*255*255*128/outAlpha255
              val background=255*128-foreground
              for channel <- 0 until 3 do
                val value=if alpha==255 then blurred(channel)(i) else math.min(255,blurred(channel)(i)*255/alpha)
                val sum=value*foreground+result(channel)(i)*background+(128<<7)
                result(channel)(i)=(((sum>>>8)+sum)>>>8)>>>7
              val sum=outAlpha255+128
              result(3)(i)=((sum>>>8)+sum)>>>8
    val output=new Array[Int](count)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("inpainting blur-fill compositing cancelled")
      for x <- 0 until width do
        val i=y*width+x
        val amount=levels(i)&255
        def channel(index: Int,shift: Int): Int =
          val original=(pixels(i)>>>shift)&255
          div255(result(index)(i)*amount+original*(255-amount))
        output(i)=0xff000000|(channel(0,16)<<16)|(channel(1,8)<<8)|channel(2,0)
    val filled=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    filled.setRGB(0,0,width,height,output,0,width)
    filled

  def applyFill(image: BufferedImage,mask: BufferedImage,mode: Int,seed: Long,
      cancelled: () => Boolean): BufferedImage =
    require(Set(0,1,2,3)(mode),"invalid inpainting fill mode")
    if mode==0 then blurredColorFill(image,mask,cancelled)
    else if mode==1 then image
    else
      require(image.getWidth==mask.getWidth && image.getHeight==mask.getHeight,"inpaint fill image/mask size mismatch")
      val width=image.getWidth;val height=image.getHeight
      val count=width*height
      val source=image.getRGB(0,0,width,height,null,0,width)
      val levels=mask.getRGB(0,0,width,height,null,0,width)
      val noise=if mode==2 then PythonRandom(seed).bytes(count*3,cancelled) else new Array[Byte](0)
      val output=new Array[Int](count)
      for y <- 0 until height do
        if cancelled() || Thread.currentThread().isInterrupted then
          throw new java.util.concurrent.CancellationException("inpainting fill cancelled")
        for x <- 0 until width do
          val i=y*width+x
          if (levels(i)&255)>127 then
            if mode==2 then
              val j=i*3
              output(i)=0xff000000|((noise(j)&255)<<16)|((noise(j+1)&255)<<8)|(noise(j+2)&255)
            else output(i)=0xff7f7f7f
          else output(i)=source(i)
      val result=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      result.setRGB(0,0,width,height,output,0,width)
      result
