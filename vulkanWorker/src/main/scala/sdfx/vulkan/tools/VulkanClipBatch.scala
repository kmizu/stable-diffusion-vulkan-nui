package sdfx.vulkan.tools

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,LinkOption,Path,StandardCopyOption}
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import ujson.{Arr,Obj,Value}
import sdfx.vulkan.VulkanInference

private[tools] object VulkanClipBatch:
  val formats=Vector("Text file for each image","Single text file with all prompts","csv file with columns for filenames and prompts")
  def execute(raw:Value,engine:VulkanInference,device:String,check:()=>Unit,publish:()=>Unit,progress:(String,Int,Int)=>Unit):Value =
    val dir=Path.of(raw("folder").str)
    require(Files.isDirectory(dir),"CLIP batch folder does not exist")
    val mode=raw.obj.get("mode").fold(VulkanClip.modes.head)(_.str)
    require(VulkanClip.modes.contains(mode),s"Unsupported Vulkan CLIP batch mode $mode")
    val format=raw.obj.get("output_mode").fold(formats.head)(_.str)
    require(formats.contains(format),"Unknown CLIP batch output mode")
    val overwrite=raw.obj.get("overwrite").exists(_.bool)
    val files=Using.resource(Files.list(dir))(_.iterator.asScala.filter { p=>
      val n=p.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
      Files.isRegularFile(p) && Seq(".jpg",".jpeg",".png").exists(n.endsWith)
    }.toVector.sortBy(_.getFileName.toString))
    require(files.nonEmpty,"Folder has no PNG or JPEG images")
    def individual(p:Path):Path=p.resolveSibling(p.getFileName.toString.replaceFirst("\\.[^.]+$",".txt"))
    val separate=format==formats.head
    val aggregate=dir.resolve(if format==formats(1) then "batch.txt" else "batch.csv")
    require(separate || overwrite || !Files.exists(aggregate,LinkOption.NOFOLLOW_LINKS),"Batch output already exists")
    val (skipped,todo)=files.partition(p=>separate && !overwrite && Files.exists(individual(p),LinkOption.NOFOLLOW_LINKS))
    require(todo.nonEmpty,"Every image already has a text output")
    if separate then require(todo.map(individual).distinct.size==todo.size,"Two images have the same output filename")
    val options=Obj.from(raw.obj)
    options("operation")="prompt";options("keep_loaded")=true;options("mode")=mode
    val rows=mutable.ArrayBuffer.empty[(Path,String)]
    val failed=mutable.ArrayBuffer.empty[Path]
    try
      for (p,i)<-todo.zipWithIndex do
        check()
        val readable=try {DiffusionImageInput.readRgb(p);true} catch case NonFatal(_) => false
        if !readable then failed+=p
        else
          options("image_path")=p.toString
          rows+=p->VulkanClip.execute(options,engine,device,check,(_,_,_)=>())("prompt").str
        progress("clip_interrogator",i+1,todo.size)
    finally if !raw.obj.get("keep_loaded").exists(_.bool) then engine.unloadClipVision()
    def csv(s:String):String=if s.exists(c=>c==',' || c=='"' || c=='\r' || c=='\n') then "\""+s.replace("\"","\"\"")+"\"" else s
    val outputs:Vector[(Path,String)]=
      if separate then rows.toVector.map((path,label)=>individual(path)->label)
      else if format==formats(1) then Vector(aggregate->rows.map(_._2+System.lineSeparator()).mkString)
      else Vector(aggregate->("filename,prompt\r\n"+rows.map((p,t)=>csv(p.getFileName.toString)+","+csv(t)+"\r\n").mkString))
    val staged=mutable.ArrayBuffer.empty[(Path,Path)]
    try
      outputs.foreach { (target,body)=>
        check()
        val temp=Files.createTempFile(dir,".clip-batch-",".tmp")
        staged+=temp->target
        Files.writeString(temp,body,UTF_8)
      }
      check();publish()
      staged.foreach { (temp,target)=>
        if overwrite then Files.move(temp,target,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
        else Files.createLink(target,temp)
      }
    finally staged.foreach((p,_)=>Files.deleteIfExists(p))
    def names(xs:Iterable[Path]):Value=Arr.from(xs.iterator.map(p=>ujson.Str(p.getFileName.toString)).toVector)
    Obj("processed"->rows.size,"total"->files.size,"failed"->names(failed),"skipped_existing"->names(skipped),
      "outputs"->Arr.from(outputs.take(200).map((p,_)=>ujson.Str(p.toString))),"mode"->mode,"output_mode"->format,"folder"->dir.toString)
