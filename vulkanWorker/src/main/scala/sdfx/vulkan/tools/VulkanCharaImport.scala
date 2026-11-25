package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files,Path,StandardCopyOption,StandardOpenOption}
import java.nio.charset.StandardCharsets.{US_ASCII,UTF_8}
import java.util.zip.{ZipEntry,ZipOutputStream}
import javax.imageio.ImageIO
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.charasearch.Npy
import sdfx.vulkan.VulkanInference
import ujson.{Arr,Obj,Value}

private[vulkan] object VulkanCharaImport:
  def available:Boolean=VulkanWdTagger.available && VulkanCharaMasks.available
  private val images=Set("png","jpg","jpeg","webp","bmp")
  private val videos=Set("mp4","mov","mkv","avi","webm","m4v","flv","wmv")
  private def ext(p:Path)=p.getFileName.toString.toLowerCase(java.util.Locale.ROOT).split("\\.").last
  private def walk(p:Path):Vector[Path]=Using.resource(Files.walk(p))(_.iterator.asScala.filter(Files.isRegularFile(_)).toVector.sortBy(_.toString))
  private def npz(path:Path,rows:Vector[Array[Float]]):Unit =
    val dict=s"{'descr': '<f4', 'fortran_order': False, 'shape': (${rows.size}, 512), }"
    val header=(dict+" "*((64-(11+dict.length)%64)%64)+"\n").getBytes(US_ASCII)
    Using.resource(new ZipOutputStream(Files.newOutputStream(path))) { zip =>
      zip.putNextEntry(new ZipEntry("embedding.npy"))
      zip.write(Array(147.toByte)++"NUMPY".getBytes(US_ASCII)++Array(1.toByte,0.toByte,(header.length&255).toByte,(header.length>>>8).toByte))
      zip.write(header)
      val bytes=ByteBuffer.allocate(512*4).order(ByteOrder.LITTLE_ENDIAN)
      for row<-rows do
        require(row.length==512 && row.forall(_.isFinite),"Invalid character embedding")
        bytes.asFloatBuffer.put(row);zip.write(bytes.array())
      zip.closeEntry()
    }
  def execute(raw:Value,engine:VulkanInference,device:String,display:String,check:()=>Unit,publish:()=>Unit,progress:(Int,Int)=>Unit):Value =
    VulkanCharaMasks.verify()
    require(VulkanWdTagger.available,"Character tagging requires the WD v3 weights")
    val config=raw.obj.getOrElse("options",Obj())
    def num(key:String,default:Double)=config.obj.get(key).fold(default)(_.num)
    val mask=num("mask_threshold",.6);val det=num("instance_threshold",.3);val tag=num("tag_threshold",.35);val pad=num("padding",.1)
    require(Seq(mask,det,tag).forall(v=>v>=0 && v<1) && pad>=0 && pad<=1,"Invalid character thresholds or padding")
    val excluded=config.obj.get("exclude_tags").fold("")(_.str).split(",").map(_.trim.toLowerCase(java.util.Locale.ROOT).replace("\\(","(").replace("\\)",")").replace('_',' ')).toSet
    val sources=raw("paths").arr.toVector.map(p=>Path.of(p.str).toAbsolutePath.normalize()).flatMap(p=>if Files.isDirectory(p) then walk(p) else Vector(p))
      .filter(p=>images(ext(p)) || videos(ext(p))).distinct
    require(sources.nonEmpty && sources.size<=5000,"Select between one and 5000 image or video sources")
    val dataset=Files.createDirectories(Path.of(raw("dataset_dir").str).toAbsolutePath.normalize())
    val staging=Files.createTempDirectory(dataset,".chara-import-")
    val pending=mutable.LinkedHashMap.empty[Path,Path]
    val created=mutable.ArrayBuffer.empty[String]
    val errors=mutable.ArrayBuffer.empty[String]
    def temp(suffix:String)=Files.createTempFile(staging,"stage-",suffix)
    def handle(path:Path,stem:String):Unit =
      val input=DiffusionImageInput.readRgb(path,compositeAlpha=false,preserveAlpha=true)
      val chars=VulkanCharaMasks.extract(input,display,det,mask,pad,check)
      chars.zipWithIndex.foreach { (img,i) =>
        var name=s"${stem}_$i.png"
        var counter=0
        while Files.exists(dataset.resolve(name)) || pending.contains(dataset.resolve(name)) do
          counter+=1;name=s"$stem-${counter}_$i.png"
        val f=temp(".png");require(ImageIO.write(img,"png",f.toFile))
        pending(dataset.resolve(name))=f;created+=name
      }
    try
      Using.resource(FileChannel.open(dataset.resolve(".chara-import.lock"),StandardOpenOption.CREATE,StandardOpenOption.WRITE)) { channel =>
        val lock=channel.tryLock();require(lock!=null,"Another import is running for this dataset")
        try
          sources.zipWithIndex.foreach { (file,i) =>
            check();progress(i,sources.size)
            val base=file.getFileName.toString.replaceFirst("\\.[^.]*$","")
            val stem=(file.getParent.getFileName.toString+"_"+base).replaceAll("(?U)[^\\w.\\-]+","_").replaceAll("^[._]+|[._]+$","")
            if videos(ext(file)) then VulkanCharaVideo.frames(file,staging,num("frame_interval",4),num("max_frames",2000).toInt,check) { (frame,index) =>
              handle(frame,f"${stem}_$index%09d")
            }
            else
              try handle(file,stem)
              catch case ex:java.io.IOException=>errors+=s"${file.getFileName}: ${ex.getMessage}"
          }
          val ip=dataset.resolve("embedding.json");val np=dataset.resolve("embedding.npz")
          require(Files.isRegularFile(ip)==Files.isRegularFile(np),"Incomplete character dataset index")
          val entries=if Files.exists(ip) then ujson.read(Files.readString(ip))("files").arr.toVector else Vector.empty
          val rows=if Files.exists(np) then Npy.readNpz(np).toVector else Vector.empty
          require(entries.size==rows.size,"Inconsistent character dataset")
          val known=entries.map(_("path").str).toSet
          val paths=(walk(dataset).filter(p=> !p.startsWith(staging) && ext(p)=="png")++pending.keys).distinct
          val todo=paths.map(p=>(dataset.relativize(p).toString.replace('\\','/'),p)).filterNot((n,_)=>known(n)).sortBy(_._1)
          val added=mutable.ArrayBuffer.empty[Value];val vectors=mutable.ArrayBuffer.empty[Array[Float]]
          if todo.nonEmpty then
            val program=VulkanBaseClip.searchProgram(device,check)
            try todo.foreach { (relative,output) =>
              check()
              val source=pending.getOrElse(output,output)
              val image=DiffusionImageInput.readRgb(source)
              val caption=VulkanWdTagger.tags(source,display,tag,false,check,true,excluded)
              val txt=temp(".txt");Files.writeString(txt,caption,UTF_8)
              pending(output.resolveSibling(output.getFileName.toString.dropRight(4)+".txt"))=txt
              vectors+=VulkanAesthetic.projection(image,engine,program,false,false,512,true,false)
              added+=Obj("path"->relative,"width"->image.getWidth,"height"->image.getHeight)
            }
            finally engine.unloadClipVision()
          if added.nonEmpty || !Files.exists(ip) then
            val json=temp(".json");Files.writeString(json,ujson.write(Obj("files"->Arr.from(entries++added))),UTF_8)
            val archive=temp(".npz");npz(archive,rows++vectors)
            pending(np)=archive;pending(ip)=json
          check();publish()
          val installed=mutable.ArrayBuffer.empty[(Path,Option[Path])]
          try pending.foreach { (dest,stage) =>
            val prior=Option.when(Files.isRegularFile(dest)) { val backup=temp(".bak");Files.copy(dest,backup,StandardCopyOption.REPLACE_EXISTING);backup }
            Files.move(stage,dest,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
            installed+=((dest,prior))
          }
          catch case ex:Throwable =>
            installed.reverseIterator.foreach((dest,prior)=>prior.fold[Unit]({Files.deleteIfExists(dest);()})(p=>{Files.move(p,dest,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);()}))
            throw ex
          progress(sources.size,sources.size)
          Obj("dataset"->dataset.toString,"sources"->sources.size,"created"->Arr.from(created.map(ujson.Str(_))),
            "new"->added.size,"errors"->Arr.from(errors.map(ujson.Str(_))),"total"->(entries.size+added.size))
        finally lock.release()
      }
    finally
      Using.resource(Files.list(staging))(_.iterator.asScala.foreach(Files.deleteIfExists(_)))
      Files.deleteIfExists(staging)
