package sdfx.vulkan.tools

import java.io.{ByteArrayOutputStream,DataOutputStream,File}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path,StandardCopyOption}
import java.util.zip.{ZipEntry,ZipOutputStream}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.vulkan.VulkanInference
import ujson.{Arr,Obj,Value}

private[vulkan] object VulkanAesthetic:
  private val revision = "32bd64288804d66eefd0ccbe215aa642df71cc41"
  private val sha = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"
  private val repo = "openai/clip-vit-large-patch14"
  def cache:Path = sys.env.get("SDFX_AESTHETIC_GRADIENTS_CACHE").map(Path.of(_)).getOrElse {
    val root=Path.of(sys.props("user.home"),".cache")
    val current=root.resolve("stable-diffusion-nui/aesthetic_gradients")
    val old=root.resolve("stablediffusion-nui/aesthetic_gradients")
    if !Files.exists(current) && Files.exists(old) then old else current
  }
  def folders:Vector[Path] =
    sys.env.get("SDFX_AESTHETIC_EMBEDDINGS").toVector.flatMap(_.split(File.pathSeparator)).filter(_.trim.nonEmpty).map(Path.of(_)) :+ cache.resolve("embeddings")
  private def files(dir:Path):Vector[Path] =
    Using.resource(Files.list(dir))(_.iterator.asScala.filter(Files.isRegularFile(_)).toVector.sortBy(_.toString))
  private def ext(p:Path):String = p.getFileName.toString.toLowerCase(java.util.Locale.ROOT).split("\\.").last
  private def list():Value =
    val names=folders.filter(Files.isDirectory(_)).flatMap(files).filter(p=>Set("pt","safetensors")(ext(p)))
      .map(_.getFileName.toString.replaceFirst("\\.[^.]+$","")).distinct.sorted
    Obj("embeddings"->Arr.from(names.map(ujson.Str(_))),"folders"->Arr.from(folders.map(p=>ujson.Str(p.toString))))

  /** Fixed torch.save tensor schema. No pickle is evaluated while exporting. */
  private def saveTensor(path:Path,row:Array[Float]):Unit =
    val buf=new ByteArrayOutputStream
    val p=new DataOutputStream(buf)
    def ascii(s:String):Unit=p.writeBytes(s)
    def string(s:String):Unit = {p.writeByte('X');p.writeInt(Integer.reverseBytes(s.length));ascii(s)}
    def dim():Unit={p.writeByte('M');p.writeShort(java.lang.Short.reverseBytes(768.toShort).toInt)}
    p.write(Array(0x80.toByte,2.toByte))
    ascii("ctorch._utils\n_rebuild_tensor_v2\n((")
    string("storage");ascii("ctorch\nFloatStorage\n");string("0");string("cpu");dim()
    ascii("tQ");p.write(Array('K'.toByte,0.toByte,'K'.toByte,1.toByte));dim();p.writeByte(0x86)
    dim();p.write(Array('K'.toByte,1.toByte,0x86.toByte,0x89.toByte))
    ascii("ccollections\nOrderedDict\n)RtR.")
    val bytes=ByteBuffer.allocate(row.length*4).order(ByteOrder.LITTLE_ENDIAN)
    row.foreach(bytes.putFloat)
    Using.resource(new ZipOutputStream(Files.newOutputStream(path))) { zip =>
      def entry(name:String,content:Array[Byte]):Unit =
        zip.putNextEntry(new ZipEntry("archive/"+name));zip.write(content);zip.closeEntry()
      entry("data.pkl",buf.toByteArray);entry("data/0",bytes.array());entry("version","3\n".getBytes(UTF_8));entry("byteorder","little".getBytes(UTF_8))
    }

  def referenceWeights(root:Path,check:()=>Unit,progress:(String,Int,Int)=>Unit):Path =
    val cacheDir=Files.createDirectories(root.resolve("openai__clip-vit-large-patch14").resolve(revision))
    VulkanClip.asset(cacheDir,"model.safetensors",s"https://huggingface.co/$repo/resolve/$revision/model.safetensors",sha,check,progress)

  def visionProgram(root:Path,device:String,check:()=>Unit,progress:(String,Int,Int)=>Unit):String =
    val weights=referenceWeights(root,check,progress)
    val file=ujson.write(ujson.Str(weights.toAbsolutePath.toString));val adapter=ujson.write(ujson.Str(device))
    s"""fn main(input) {
      vision = load_vision_encoder($file, backend=$adapter, params_backend=$adapter)
      return {embedding = encode_vision(vision,input,pooled=true)}
    }"""

  def projection(source:java.awt.image.BufferedImage,engine:VulkanInference,program:String,lanczos:Boolean,roundCrop:Boolean,dimensions:Int=768,roundSize:Boolean=false,normalize:Boolean=true):Array[Float] =
    val short=source.getWidth.min(source.getHeight)
    def side(x:Int)=if roundSize then Math.rint(224.0*x/short).toInt else (x.toLong*224/short).toInt
    val w=side(source.getWidth)
    val h=side(source.getHeight)
    val resized=if lanczos then DiffusionImageResize.lanczos(source,w,h) else DiffusionImageResize.bicubic(source,w,h)
    def offset(x:Int)=if roundCrop then Math.rint((x-224)/2.0).toInt else (x-224)/2
    val crop=resized.getSubimage(offset(w),offset(h),224,224)
    val row=engine.clipVision(program,crop,true,dimensions)
    val n=math.sqrt(row.iterator.map(v=>v.toDouble*v).sum)
    require(n.isFinite && n>0,"Invalid CLIP image projection")
    if normalize then row.map(v=>(v/n).toFloat) else row

  def execute(raw:Value,engine:VulkanInference,device:String,check:()=>Unit,publish:()=>Unit,progress:(String,Int,Int)=>Unit):Value =
    val op=raw.obj.get("op").fold("create")(_.str)
    if op=="list" then return list()
    require(op=="create","Unknown aesthetic embedding operation")
    val name=raw("name").str.trim
    require(name.nonEmpty && !name.startsWith(".") && !name.exists("\\/:*?\"<>|".contains(_)),"Aesthetic embedding name must be a plain filename")
    val overwrite=raw.obj.get("overwrite").exists(_.bool)
    val dir=raw.obj.get("output_dir").map(_.str).filter(_.nonEmpty).fold(folders.head)(Path.of(_))
    val output=dir.resolve(if name.toLowerCase(java.util.Locale.ROOT).endsWith(".pt") then name else name+".pt")
    require(overwrite || !Files.exists(output),"Aesthetic embedding output already exists")
    val paths=raw("images").strOpt.toVector++raw("images").arrOpt.toVector.flatMap(_.map(_.str))
    val images=paths.flatMap { s =>
      val path=Path.of(s)
      if Files.isDirectory(path) then files(path).filter(p=>Set("png","jpg","jpeg","bmp","webp")(ext(p)))
      else {require(Files.isRegularFile(path),s"Image not found: $path");Vector(path)}
    }
    require(images.nonEmpty,"No images to embed")
    val program=visionProgram(cache,device,check,progress)
    val sum=Array.fill[Double](768)(0)
    try
      for (path,i)<-images.zipWithIndex do
        check()
        val source=DiffusionImageInput.readRgb(path)
        val row=projection(source,engine,program,false,false)
        row.indices.foreach(j=>sum(j)+=row(j))
        progress("aesthetic_embedding",i+1,images.size)
    finally engine.unloadClipVision()
    val norm=math.sqrt(sum.iterator.map(v=>v*v).sum)/images.size
    require(norm.isFinite && norm>0,"Embeddings cancel out")
    check()
    Files.createDirectories(dir)
    val stage=Files.createTempFile(dir,".aesthetic-",".tmp")
    try
      saveTensor(stage,sum.map(x=>(x/images.size/norm).toFloat))
      check();publish()
      if overwrite then Files.move(stage,output,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
      else Files.createLink(output,stage)
    finally Files.deleteIfExists(stage)
    Obj("path"->output.toString,"count"->images.size,"norm_of_mean"->norm)
