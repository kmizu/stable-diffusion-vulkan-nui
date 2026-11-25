package sdfx.vulkan.tools

import java.nio.file.{Files,Path,LinkOption}
import java.nio.file.attribute.BasicFileAttributes
import java.util.UUID
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import ujson.{Arr,Obj,Value}
import sdfx.core.imagebrowser.BrowserFiles
import sdfx.core.png.ScrambledPng
import sdfx.core.protocol.{EngineError,LoadedModel}

private[vulkan] final class VulkanWdBatch:
  private case class Snapshot(path:Path,stamp:BasicFileAttributes,sha:String)
  private case class Plan(owner:String,token:String,deadline:Long,items:Vector[Snapshot],generation:String,destination:Path)
  private val pending=mutable.Map.empty[String,Plan]
  private def stamp(path:Path):BasicFileAttributes=Files.readAttributes(path,classOf[BasicFileAttributes],LinkOption.NOFOLLOW_LINKS)
  private def stable(a:BasicFileAttributes,b:BasicFileAttributes):Boolean=a.fileKey==b.fileKey && a.size==b.size && a.lastModifiedTime==b.lastModifiedTime
  private def checkedPath(p:Path):Path =
    val abs=p.toAbsolutePath.normalize()
    Iterator.iterate(abs)(_.getParent).takeWhile(_!=null).foreach(a=>require(!Files.isSymbolicLink(a),"Batch paths cannot contain links"))
    abs
  private def readable(p:Path):Snapshot =
    val before=stamp(p)
    require(before.isRegularFile && before.size>0 && before.size<=128L*1024*1024,"Invalid WD image size")
    DiffusionImageInput.readRgb(p,maxPixels=32000000,maxBytes=128L*1024*1024)
    val hash=BrowserFiles.sha256(p)
    require(stable(before,stamp(p)),"Batch image changed during preparation")
    Snapshot(p,before,hash)
  def prepare(raw:Value,owner:String,model:LoadedModel,check:()=>Unit,publish:()=>Unit):Value =
    require(owner.nonEmpty && owner.length<=256 && raw.obj.keySet==Set("images_root","generation","output_parent","output_name","checkpoint_token"))
    require(VulkanWdTagger.available,"WD assets unavailable")
    val token=raw("checkpoint_token").str
    require(model.modelToken.contains(token),"Loaded checkpoint changed")
    val root=checkedPath(Path.of(raw("images_root").str));val parent=checkedPath(Path.of(raw("output_parent").str))
    require(Files.isDirectory(root) && Files.isDirectory(parent))
    val name=raw("output_name").str
    require(name.nonEmpty && name!="." && name!=".." && !name.endsWith(".") && !name.endsWith(" ") && !name.exists("\\/:*?\"<>|".contains(_)))
    val dest=parent.resolve(name);require(!Files.exists(dest,LinkOption.NOFOLLOW_LINKS),"Output already exists")
    val g=raw("generation");val params=g("params");val frozen=ujson.write(g)
    require(frozen.length<=2000000 && !g.obj.get("vae_recovery_dir").exists(_.strOpt.exists(_.nonEmpty)) &&
      !params.obj.get("hooks").exists(_.obj.nonEmpty) && !params.obj.get("hires").exists(_.obj.get("enabled").exists(_.bool)),"Hooks, recovery and hires unavailable for WD batches")
    val paths=Using.resource(Files.list(root)) { stream=>
      val entries=stream.iterator.asScala.take(10001).toVector
      require(entries.size<=10000 && entries.forall(p => !Files.isSymbolicLink(p)),"Invalid WD input directory")
      entries.filter(p=>Files.isRegularFile(p) && p.getFileName.toString.toLowerCase(java.util.Locale.ROOT).matches(".*\\.(jpg|jpeg|png|webp)")).sortBy(_.getFileName.toString.toLowerCase(java.util.Locale.ROOT))
    }
    require(paths.nonEmpty && paths.size<=1000 && paths.map(_.getFileName.toString.toLowerCase(java.util.Locale.ROOT).replaceFirst("\\.[^.]+$","")).distinct.size==paths.size,"Empty batch or ambiguous image names")
    val batch=params.obj.get("batch_size").fold(1.0)(_.num)
    val count=params.obj.get("n_iter").fold(1.0)(_.num)
    val w=params.obj.get("width").fold(512.0)(_.num)
    val h=params.obj.get("height").fold(512.0)(_.num)
    require(Seq(batch,count,w,h).forall(x=>x.isWhole && x>0) && batch<=16 && count<=100 && w<=4096 && h<=4096)
    val outputs=paths.size*batch*count
    require(outputs<=10000 && outputs*(4*w*h+1048576)<=8L*1024*1024*1024,"Estimated WD batch outputs exceed limits")
    var bytes=0L
    val inputs=paths.map { p=>check();val snap=readable(p);bytes+=snap.stamp.size;require(bytes<=2L*1024*1024*1024);snap }
    pending.filterInPlace((_,v)=>v.deadline>System.nanoTime());require(pending.size<8,"Too many pending WD batches")
    val id=UUID.randomUUID.toString.replace("-","")
    check();publish();pending(id)=Plan(owner,token,System.nanoTime()+300L*1000000000,inputs,frozen,dest)
    Obj("token"->id,"image_count"->inputs.size,"backend"->"wd_ext","partial"->true)

  def execute(raw:Value,owner:String,model:LoadedModel,device:String,password:Option[String],check:()=>Unit,publish:()=>Unit,generate:Value=>Value):Value =
    require(raw.obj.keySet==Set("token"));val id=raw("token").str;require(id.matches("[0-9a-f]{32}"))
    val plan=pending.get(id).filter(p=>p.owner==owner && p.deadline>System.nanoTime()).getOrElse(throw IllegalArgumentException("WD preparation unavailable or expired"))
    check();pending.remove(id)
    require(model.modelToken.contains(plan.token),"Loaded checkpoint changed")
    val dest=checkedPath(plan.destination);require(!Files.exists(dest,LinkOption.NOFOLLOW_LINKS),"WD output already exists")
    val stage=Files.createTempDirectory(dest.getParent,".sdv-wd-")
    def cleanup(dir:Path):Unit=if Files.isDirectory(dir,LinkOption.NOFOLLOW_LINKS) then
      Using.resource(Files.walk(dir))(_.iterator.asScala.toVector.reverse.foreach(Files.deleteIfExists))
    try
      val rows=plan.items.zipWithIndex.map { (snap,i)=>
        check()
        val original=snap.path
        require(stable(snap.stamp,stamp(checkedPath(original))) && BrowserFiles.sha256(original)==snap.sha,"Batch input changed after preparation")
        val bound=stage.resolve(s".source-$i-"+original.getFileName)
        Files.copy(original,bound)
        require(BrowserFiles.sha256(bound)==snap.sha,"Batch input changed during binding")
        val item=stage.resolve(f"item-$i%04d")
        try
          val tags=VulkanWdTagger.tags(bound,device,.35,true,check)
          val req=ujson.read(plan.generation);val p=req("params")
          val submitted=p.obj.get("prompt").fold("")(_.str)
          val positive=Vector(tags,submitted).filter(_.nonEmpty).mkString(", ")
          val negative=p.obj.get("negative_prompt").fold("")(_.str)
          p("prompt")=positive;p("negative_prompt")=negative
          val settings=p.obj.get("img2img").filterNot(_==ujson.Null).getOrElse(Obj())
          settings("init_image")=bound.toString;p("img2img")=settings
          req("output_dir")=item.toString
          val result=generate(req)
          require(result("images").arr.nonEmpty)
          val provenance=Obj("backend"->"wd_ext","model_id"->VulkanWdTagger.modelId,"model_sha256"->VulkanWdTagger.modelSha,"tags_sha256"->VulkanWdTagger.tableSha,
            "threshold"->0.35,"remove_underscores"->true,"append_ratings"->false,"rating_threshold"->0.5,"preprocessing"->"exif-white-alpha-square-pillow-bicubic-bgr-f32-v1")
          val gp=sdfx.vulkan.VulkanProtocol.generation(req).request.params
          result("images").arr.foreach { image=>
            check()
            val file=Path.of(image("path").str).toAbsolutePath.normalize();require(file.getParent==item && Files.isRegularFile(file,LinkOption.NOFOLLOW_LINKS))
            val png=ScrambledPng.read(file,password)
            import sdfx.core.infotext.{ImageFacts,Infotext}
            val extra=scala.collection.immutable.ListMap("WD interrogators"->ujson.write(Arr(provenance)),"WD input"->original.toString,"WD input SHA256"->snap.sha)
            val facts=ImageFacts(image("seed").num.toLong,image.obj.get("subseed").fold(-1L)(_.num.toLong),
              Some(model.name),Some(model.hash),model.vaeName,model.vaeHash,scheduler=result.obj.get("scheduler").flatMap(_.strOpt),extra=extra)
            val text=Infotext.render(positive,negative,Infotext.fields(gp,facts))
            ScrambledPng.write(png.image,file,Seq("parameters"->text),password)
            image("path")=dest.resolve(item.getFileName).resolve(file.getFileName).toString
          }
          Obj("status"->"complete","input"->original.toString,"positive"->positive,"negative"->negative,"result"->result,"interrogators"->Arr(provenance))
        catch
          case e:java.util.concurrent.CancellationException=>throw e
          case e:EngineError if Set("oom","interrupted")(e.code)=>throw e
          case NonFatal(e)=>cleanup(item);Obj("status"->"error","input"->original.toString,"error"->Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
        finally Files.deleteIfExists(bound)
      }
      check();publish()
      if rows.exists(_("status").str=="complete") then Files.move(stage,dest)
      Obj("rows"->Arr.from(rows),"partial"->true)
    finally cleanup(stage)
