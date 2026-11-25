// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import sdfx.vulkan.VulkanInference.Latent
import java.awt.image.BufferedImage
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path,StandardOpenOption}
import javax.imageio.ImageIO
import scala.util.Using

/** NUI's four-byte-per-pixel latent layout shared by RGBA and CMYK containers. */
private[vulkan] object NativeVaeLatent:
  private val maxElements=16L*1024*1024
  final case class Quantized(width: Int,height: Int,bytes: Array[Byte],clipped: Int)

  private val maxExactBytes=maxElements*4+65536
  /** Uncompressed CMYK TIFF keeps the four quantized channels unchanged. */
  def writeCmykTiff(latent: Latent,path: Path): Quantized =
    val packed=packQuantized(latent)
    val tags=11;val bitsOffset=8+2+tags*12+4;val pixelOffset=bitsOffset+8
    val out=ByteBuffer.allocate(Math.addExact(pixelOffset,packed.bytes.length)).order(ByteOrder.LITTLE_ENDIAN)
    out.put('I'.toByte).put('I'.toByte).putShort(42.toShort).putInt(8).putShort(tags.toShort)
    def field(tag: Int,kind: Int,count: Int,value: Int): Unit =
      out.putShort(tag.toShort).putShort(kind.toShort).putInt(count).putInt(value)
    field(256,4,1,packed.width);field(257,4,1,packed.height)
    field(258,3,4,bitsOffset);field(259,3,1,1);field(262,3,1,5)
    field(273,4,1,pixelOffset);field(277,3,1,4);field(278,4,1,packed.height)
    field(279,4,1,packed.bytes.length);field(284,3,1,1);field(332,3,1,1)
    out.putInt(0)
    for _<-0 until 4 do out.putShort(8.toShort)
    out.put(packed.bytes)
    Files.write(path,out.array(),StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    packed

  def writeCmykJpeg(latent: Latent,path: Path): Unit =
    val packed = packQuantized(latent)
    // JPEG stores complemented CMYK samples. Passing a raster avoids an RGB/YCC conversion.
    val data = new java.awt.image.DataBufferByte(packed.bytes.map(b => (~b).toByte),packed.bytes.length)
    val raster = java.awt.image.Raster.createInterleavedRaster(data,packed.width,packed.height,packed.width*4,4,Array(0,1,2,3),null)
    val writer = ImageIO.getImageWritersByFormatName("JPEG").next()
    try
      val options = writer.getDefaultWriteParam
      options.setCompressionMode(javax.imageio.ImageWriteParam.MODE_EXPLICIT)
      options.setCompressionQuality(1.0f)
      Using.resource(ImageIO.createImageOutputStream(path.toFile)) { stream =>
        writer.setOutput(stream)
        writer.write(null,new javax.imageio.IIOImage(raster,null,null),options)
      }
    finally writer.dispose()

  def readCmyk(path: Path,channels: Int): Latent =
    require(Files.isRegularFile(path) && Files.size(path)<=maxElements*4+65536,"latent TIFF is missing or too large")
    val input=ImageIO.createImageInputStream(path.toFile)
    require(input!=null,"TIFF input unavailable")
    Using.resource(input) { stream =>
      val readers=ImageIO.getImageReaders(stream)
      require(readers.hasNext,"invalid TIFF")
      val reader=readers.next()
      try
        val format = reader.getFormatName.toLowerCase(java.util.Locale.ROOT)
        require(Set("jpeg","jpg","tif","tiff")(format),"latent image must be CMYK TIFF or JPEG")
        reader.setInput(stream,false,false)
        val width=reader.getWidth(0);val height=reader.getHeight(0)
        require(width>0 && height>0 && width.toLong*height*4<=maxElements,"latent TIFF exceeds limit")
        if Set("tif","tiff")(format) then
          val directory=javax.imageio.plugins.tiff.TIFFDirectory.createFromMetadata(reader.getImageMetadata(0))
          def scalar(tag: Int,default: Int): Int=Option(directory.getTIFFField(tag)).map(_.getAsInt(0)).getOrElse(default)
          val bits=directory.getTIFFField(258)
          require(scalar(262,-1)==5 && scalar(277,1)==4 && scalar(332,1)==1 && bits!=null &&
            bits.getCount==4 && (0 until 4).forall(i=>bits.getAsInt(i)==8),"latent TIFF must be 8-bit CMYK")
        val image=reader.read(0)
        require(image.getColorModel.getColorSpace.getType==java.awt.color.ColorSpace.TYPE_CMYK,"Image reader did not preserve CMYK")
        val raster=image.getRaster
        require(raster.getNumBands==4,"invalid CMYK channels")
        val bytes=new Array[Byte](width*height*4)
        val row=new Array[Int](width*4)
        for y<-0 until height do
          raster.getPixels(0,y,width,1,row)
          for i<-row.indices do bytes(y*row.length+i)=row(i).toByte
        unpackQuantized(width,height,bytes,channels)
      finally reader.dispose()
    }

  private def validateExact(latent: Latent, channels: Int): Unit =
    require(channels == 4 && latent.values.forall(_.isFinite), "invalid four-channel float latent")

  def safetensors(latent: Latent, metadata: String = "{}"): Array[Byte] =
    validateExact(latent, 4)
    val dataSize = Math.multiplyExact(4, latent.values.length)
    val json = ujson.Obj("tensor" -> ujson.Obj("dtype" -> "F32",
      "shape" -> ujson.Arr(1, 4, latent.height, latent.width), "data_offsets" -> ujson.Arr(0,dataSize)),
      "__metadata__" -> ujson.Obj("sdfx_vae_tools" -> metadata))
    val header = ujson.write(json).getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val padded = (header.length + 7) / 8 * 8
    val out = ByteBuffer.allocate(8+padded+dataSize).order(ByteOrder.LITTLE_ENDIAN)
    out.putLong(padded).put(header)
    while out.position < 8+padded do out.put(32.toByte)
    out.asFloatBuffer().put(latent.values)
    out.array()

  def parseTensor(bytes: Array[Byte], preserveRaw: Boolean): (Latent, Boolean) =
    require(bytes.length >= 8 && bytes.length <= maxExactBytes, "invalid latent file size")
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val length = Math.toIntExact(buf.getLong())
    require(length >= 2 && length <= 65536 && length <= buf.remaining(), "invalid latent header size")
    val header = new Array[Byte](length)
    buf.get(header)
    val json = ujson.read(new String(header,java.nio.charset.StandardCharsets.UTF_8))
    require(json.obj.keySet.filterNot(_ == "__metadata__") == Set("tensor"), "expected one latent tensor")
    val tensor = json("tensor")
    val shape = tensor("shape").arr.map(v => { require(v.num.isValidInt); v.num.toInt })
    require(shape.length == 4 && shape.take(2).toSeq == Seq(1,4) && shape.forall(_ > 0), "invalid latent shape")
    val count = shape.foldLeft(1)(Math.multiplyExact)
    require(count <= maxElements, "latent exceeds element limit")
    val dtype = tensor("dtype").str
    val size = dtype match
      case "F32" => 4
      case "F16" | "BF16" => 2
      case "F64" => 8
      case other => throw IllegalArgumentException(s"Invalid latent type $other")
    val dataSize = Math.multiplyExact(size,count)
    require(buf.remaining() == dataSize && tensor("data_offsets").arr.map(_.num).toSeq == Seq(0.0,dataSize.toDouble),
      "invalid latent tensor offsets")
    val values = Array.fill(count)(dtype match
      case "F32" => buf.getFloat()
      case "F16" => java.lang.Float.float16ToFloat(buf.getShort())
      case "BF16" => java.lang.Float.intBitsToFloat(buf.getShort().toInt << 16)
      case _ => buf.getDouble().toFloat)
    val provenance = json.obj.get("__metadata__").flatMap(_.obj.get("sdfx_vae_tools")).map(v => ujson.read(v.str)).getOrElse(ujson.Obj())
    val raw = provenance.obj.get("coordinate").map(_.str).getOrElse("diffusion_scaled") match
      case "diffusion_scaled" => false
      case "vae_decoder_input" => true
      case _ => throw IllegalArgumentException("Unknown latent coordinate space")
    if raw && !preserveRaw then
      val scale = provenance("scaling_factor").num
      require(scale.isFinite && scale > 0, "invalid VAE scaling factor")
      val shift = provenance.obj.get("shift_factor").flatMap(_.numOpt).getOrElse(0.0)
      val mean = provenance.obj.get("latents_mean").flatMap(_.arrOpt)
      val std = provenance.obj.get("latents_std").flatMap(_.arrOpt)
      for i <- values.indices do
        val c = i / (count/4)
        values(i) = ((if mean.isDefined && std.isDefined then (values(i)-mean.get(c).num)/std.get(c).num else values(i)-shift)*scale).toFloat
    val latent = new Latent(shape(3),shape(2),values)
    validateExact(latent,4)
    (latent,raw && preserveRaw)

  def readSafetensorsWithCoordinate(path: Path, channels: Int, preserveRaw: Boolean): (Latent, Boolean) =
    require(channels == 4 && Files.isRegularFile(path) && Files.size(path) <= maxExactBytes, "latent file missing or too large")
    parseTensor(Files.readAllBytes(path),preserveRaw)

  /** NUI's lossless L PNG stores a big-endian length followed by safetensors bytes. */
  def writeExactPng(latent: Latent,path: Path): Unit =
    val dims=Vector(1L,4L,latent.height.toLong,latent.width.toLong)
    require(dims.size==4 && Set(4L,16L)(dims(1)),"invalid latent shape")
    validateExact(latent,dims(1).toInt)
    val payload=safetensors(latent)
    val count=Math.addExact(payload.length,4)
    val side=math.ceil(math.sqrt(count.toDouble)).toInt
    require(side.toLong*side<=maxExactBytes,"latent image exceeds limit")
    val pixels=new Array[Byte](side*side)
    ByteBuffer.wrap(pixels).order(ByteOrder.BIG_ENDIAN).putInt(payload.length).put(payload)
    val image=new BufferedImage(side,side,BufferedImage.TYPE_BYTE_GRAY)
    val row=new Array[Int](side)
    for y<-0 until side do
      for x<-0 until side do row(x)=pixels(y*side+x)&255
      image.getRaster.setPixels(0,y,side,1,row)
    val output=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    try
      try require(ImageIO.write(image,"png",output),"PNG writer unavailable")
      finally output.close()
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error

  def readExactPng(path: Path,channels: Int): Latent =
    require(Files.isRegularFile(path) && Files.size(path)<=maxExactBytes+1048576,"latent PNG is missing or too large")
    val header=Using.resource(Files.newInputStream(path))(_.readNBytes(29))
    require(header.length==29 && header.take(8).sameElements(Array[Byte](137.toByte,80,78,71,13,10,26,10)),"invalid latent PNG")
    val data=ByteBuffer.wrap(header).order(ByteOrder.BIG_ENDIAN)
    require(data.getInt(8)==13 && data.getInt(12)==0x49484452,"invalid PNG header")
    val width=data.getInt(16);val height=data.getInt(20)
    require(width>0 && height>0 && width.toLong*height<=maxExactBytes &&
      (header(24)&255)==8 && (header(25)&255)==0,"latent PNG must be bounded 8-bit grayscale")
    val image=Using.resource(Files.newInputStream(path))(ImageIO.read)
    require(image!=null && image.getWidth==width && image.getHeight==height,"invalid grayscale latent image")
    readExactImage(image,channels)

  def readExactImage(image: BufferedImage,channels: Int): Latent =
    require(image!=null && image.getWidth>0 && image.getHeight>0 && image.getWidth.toLong*image.getHeight<=maxExactBytes &&
      image.getColorModel.getColorSpace.getType==java.awt.color.ColorSpace.TYPE_GRAY &&
      !image.getColorModel.hasAlpha && image.getRaster.getNumBands==1 && image.getSampleModel.getSampleSize.forall(_==8),
      "invalid grayscale latent image")
    val width=image.getWidth;val height=image.getHeight
    val pixels=new Array[Byte](width*height)
    val row=new Array[Int](width)
    for y<-0 until height do
      image.getRaster.getPixels(0,y,width,1,row)
      for x<-0 until width do pixels(y*width+x)=row(x).toByte
    require(pixels.length>=4,"truncated latent PNG")
    val size=ByteBuffer.wrap(pixels).order(ByteOrder.BIG_ENDIAN).getInt()&0xffffffffL
    require(size>=8 && size<=maxExactBytes && size<=pixels.length-4,"invalid latent payload length")
    val latent=parseTensor(java.util.Arrays.copyOfRange(pixels,4,4+size.toInt),false)._1
    validateExact(latent,channels)
    latent

  /** Writes a new RGBA PNG without premultiplying or replacing an existing file. */
  def writeRgbaPng(latent: Latent,path: Path): Quantized =
    val packed=packQuantized(latent)
    val image=new BufferedImage(packed.width,packed.height,BufferedImage.TYPE_4BYTE_ABGR)
    val raster=image.getRaster
    val row=new Array[Int](packed.width*4)
    for y<-0 until packed.height do
      for i<-row.indices do row(i)=packed.bytes(y*row.length+i)&255
      raster.setPixels(0,y,packed.width,1,row)
    val output=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    try
      try require(ImageIO.write(image,"png",output),"PNG writer unavailable")
      finally output.close()
      packed
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error

  /** Reads exact 8-bit RGBA channels without converting through RGB or dropping alpha. */
  def readRgbaPng(path: Path,channels: Int): Latent =
    require(Files.isRegularFile(path) && Files.size(path)<=maxElements*4+65536,"latent PNG is missing or too large")
    val header=Using.resource(Files.newInputStream(path))(_.readNBytes(29))
    require(header.length==29 && header.take(8).sameElements(Array[Byte](137.toByte,80,78,71,13,10,26,10)),"invalid latent PNG")
    val data=ByteBuffer.wrap(header).order(ByteOrder.BIG_ENDIAN)
    require(data.getInt(8)==13 && data.getInt(12)==0x49484452,"invalid PNG header")
    val width=data.getInt(16);val height=data.getInt(20)
    require(width>0 && height>0 && BigInt(width)*height*4<=maxElements &&
      (header(24)&255)==8 && (header(25)&255)==6,"latent PNG must be bounded 8-bit RGBA")
    require(Set(4,16)(channels) && (channels==4 || (width%2==0 && height%2==0)),"invalid latent channels or grid")
    val image=Using.resource(Files.newInputStream(path))(ImageIO.read)
    require(image!=null && image.getWidth==width && image.getHeight==height &&
      image.getRaster.getNumBands==4 && image.getSampleModel.getSampleSize.forall(_==8) &&
      !image.isAlphaPremultiplied,"invalid RGBA latent image")
    val bytes=new Array[Byte](width*height*4)
    val row=new Array[Int](width*4)
    for y<-0 until height do
      image.getRaster.getPixels(0,y,width,1,row)
      for i<-row.indices do bytes(y*row.length+i)=row(i).toByte
    unpackQuantized(width,height,bytes,channels)

  def packQuantized(latent: Latent): Quantized =
    val dims=Vector(1L,4L,latent.height.toLong,latent.width.toLong)
    require(dims.size==4 && dims(0)==1 &&
      Set(4L,16L)(dims(1)) && dims(2)>0 && dims(3)>0 && latent.values.length<=maxElements,
      "latent must be one bounded CPU Float32 image with 4 or 16 channels")
    val values=latent.values
    require(values.forall(_.isFinite),"latent contains nonfinite values")
    val channels=dims(1).toInt;val height=dims(2).toInt;val width=dims(3).toInt
    val grid=if channels==16 then 2 else 1
    val outWidth=Math.multiplyExact(width,grid);val outHeight=Math.multiplyExact(height,grid)
    val bytes=new Array[Byte](Math.multiplyExact(Math.multiplyExact(outWidth,outHeight),4))
    var clipped=0
    for c<-0 until channels; y<-0 until height; x<-0 until width do
      val value=values((c*height+y)*width+x)
      if math.abs(value)>10f then clipped+=1
      val scaled=(value+10f)*12.75f
      val quantized=math.max(0d,math.min(255d,math.rint(scaled.toDouble))).toInt
      val tile=c/4
      val destination=(((tile/2)*height+y)*outWidth+(tile%2)*width+x)*4+c%4
      bytes(destination)=quantized.toByte
    Quantized(outWidth,outHeight,bytes,clipped)

  def unpackQuantized(width: Int,height: Int,bytes: Array[Byte],channels: Int): Latent =
    require(width>0 && height>0 && Set(4,16)(channels) && bytes!=null,
      "invalid latent image dimensions or channels")
    val grid=if channels==16 then 2 else 1
    require(width%grid==0 && height%grid==0,"latent grid dimensions are invalid")
    val byteCount=BigInt(width)*height*4
    require(byteCount<=maxElements && byteCount==bytes.length,"latent image byte count exceeds its limit or is invalid")
    val latentWidth=width/grid;val latentHeight=height/grid
    val result=new Array[Float](byteCount.toInt)
    for c<-0 until channels; y<-0 until latentHeight; x<-0 until latentWidth do
      val tile=c/4
      val source=(((tile/2)*latentHeight+y)*width+(tile%2)*latentWidth+x)*4+c%4
      result((c*latentHeight+y)*latentWidth+x)=((bytes(source)&255).toFloat/255f)*20f-10f
    new Latent(latentWidth,latentHeight,result)
