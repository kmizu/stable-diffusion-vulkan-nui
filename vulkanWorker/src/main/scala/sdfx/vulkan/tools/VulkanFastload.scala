package sdfx.vulkan.tools

import java.io.{ByteArrayInputStream,ByteArrayOutputStream,DataInputStream,InputStream}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files,Path}
import java.util.zip.{GZIPInputStream,GZIPOutputStream,InflaterInputStream}
import javax.imageio.ImageIO
import javax.imageio.stream.FileImageInputStream
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import ujson.{Arr,Obj,Value}

private[tools] object VulkanFastload:
  import FastloadPickle.*
  private val begin="###START_OF_CONTROLNET_FASTLOAD###".getBytes(StandardCharsets.US_ASCII)
  private val end="###END_OF_CONTROLNET_FASTLOAD###".getBytes(StandardCharsets.US_ASCII)
  private val compressedLimit=64*1024*1024
  private val inputLimit=Limit+compressedLimit+1024
  private def readBytes(path:Path):Array[Byte] =
    require(Files.isRegularFile(path) && Files.size(path)<inputLimit,"Fastload input unavailable or oversized")
    Using.resource(Files.newInputStream(path)) { in =>
      val bs=in.readNBytes(inputLimit);require(bs.length<inputLimit);bs
    }
  private def index(bs:Array[Byte],s:Array[Byte],start:Int=0):Int =
    var i=start
    while i<=bs.length-s.length do
      if bs(i)==s(0) && java.util.Arrays.equals(bs,i,i+s.length,s,0,s.length) then return i
      i+=1
    -1
  private def decode(bs:Array[Byte]):(Boolean,Value) =
    val a=index(bs,begin);val b=index(bs,end)
    require(a>=0 && b>=a+begin.length && b+end.length==bs.length && index(bs,begin,a+begin.length)<0,"Missing or ambiguous Fastload trailer")
    val n=b-a-begin.length;require(n<=compressedLimit)
    val pickle=Using.resource(new GZIPInputStream(new ByteArrayInputStream(bs,a+begin.length,n)))(_.readNBytes(Limit+1))
    require(pickle.length<=Limit);(a>0,read(pickle))
  private def encode(image:Array[Byte],units:Value):Array[Byte] =
    require(image.length<=Limit && index(image,begin)<0 && index(image,end)<0,"Invalid Fastload image prefix")
    val out=new ByteArrayOutputStream
    Using.resource(new GZIPOutputStream(out))(_.write(write(units)))
    require(out.size<=compressedLimit)
    image++begin++out.toByteArray++end
  private def checked(x:Value):Value = {write(x);x}
  private def summary(units:Value):Value = Arr.from(items(units).map { unit =>
    try
      val m=FastloadUnits.mapUnit(unit);val p=m.plainParams
      val r=Obj("enabled"->p("enabled"),"model"->p("model"),"preprocessor"->p("preprocessor"),"unapplied"->Arr.from(m.unapplied.keys.toVector.sorted.map(ujson.Str(_))))
      if !p("source_compatible_resize").bool then r("warning")="This unit uses NUI resize interpolation. Other Fastload apps may use different interpolation."
      r
    catch case NonFatal(_) => Obj("error"->"Some values cannot be applied to the current ControlNet form")
  })
  private def path(raw:Value):Path =
    val s=raw("path").str;require(s.nonEmpty);Path.of(s).toAbsolutePath.normalize()
  private def suffix(p:Path):String = p.getFileName.toString.toLowerCase(java.util.Locale.ROOT).split("\\.").last
  private def enabled(x:Value):Boolean=fields(x).get("enabled").exists { v=>require(kind(v)=="bool");v("value").bool }
  def execute(raw:Value,ctx:NativeWorkContext):Value =
    try run(raw,ctx)
    catch
      case e: java.util.concurrent.CancellationException => throw e
      case e: IllegalArgumentException => throw e
      case e: java.nio.file.FileAlreadyExistsException => throw IllegalArgumentException("Fastload output already exists",e)
      case NonFatal(e) => throw IllegalArgumentException("Invalid or unreadable Fastload data",e)

  private def run(raw:Value,ctx:NativeWorkContext):Value =
    ctx.checkCancelled()
    raw("operation").str match
      case "version" => Obj("version"->1,"upstream_version"->1.1,"legacy_subset_only"->true,"protected_write"->false)
      case "gallery" => gallery(raw,ctx)
      case "gallery_source" =>
        val p=path(raw);require(!Files.isSymbolicLink(p));val bs=readBytes(p)
        if (index(bs,begin)>=0 || index(bs,end)>=0) && items(decode(bs)._2).nonEmpty then Obj("path"->p.toString,"kind"->"embedded")
        else
          val side=p.resolveSibling(p.getFileName.toString.replaceFirst("\\.[^.]*$","")+".cni");require(!Files.isSymbolicLink(side))
          if Files.isRegularFile(side) && items(decode(readBytes(side))._2).nonEmpty then Obj("path"->side.toString,"kind"->"sidecar") else Obj("path"->ujson.Null,"kind"->"none")
      case "inspect" =>
        val (image,units)=decode(readBytes(path(raw)));ctx.checkCancelled();Obj("units"->units,"has_image"->image,"summary"->summary(units))
      case "map" => Obj("mapped"->seq("list",items(checked(raw("units"))).map(FastloadUnits.mapUnit(_).wire)))
      case "from_native" =>
        val units=seq("list",items(checked(raw("items"))).map(FastloadUnits.fromNative));Obj("units"->units,"summary"->summary(units))
      case op @ ("preview" | "native_form") =>
        val us=items(checked(raw("units")));val idx=raw("index").num;require(idx.isWhole && idx>=0 && idx<us.size)
        FastloadUnits.preview(FastloadUnits.mapUnit(us(idx.toInt)),op=="native_form")
      case op @ ("resolve" | "prepare_generation") =>
        val mode=raw("mode").str;val priority=raw("priority").str
        require(Set("load_only","save_only","load_save")(mode) && Set("current_first","loaded_first")(priority))
        val current=checked(raw("current"));val active=items(current).map(enabled).exists(identity)
        val provided=if op=="resolve" then raw.obj.get("loaded").exists(_!=ujson.Null) else raw.obj.get("path").exists(_.str.nonEmpty)
        val skip=mode=="save_only" || !provided || priority=="current_first" && active
        val chosen=if skip then current else if op=="resolve" then checked(raw("loaded")) else decode(readBytes(path(raw)))._2
        val flags=items(chosen).map(enabled)
        val result=Obj("units"->chosen,"save"->(mode!="load_only"),"load_skipped"->(skip && mode!="save_only"))
        if op=="prepare_generation" && raw.obj.get("native_forms").exists(_.bool) then
          require(flags.size<=3,"Native generation supports at most three ControlNet units")
          result("forms")=Arr.from(items(chosen).zip(flags).map { (unit,on) =>
            if on then FastloadUnits.preview(FastloadUnits.mapUnit(unit),true) else Obj("params"->Obj("enabled"->false),"image"->ujson.Null,"mask"->ujson.Null,"unapplied"->Arr())
          })
        result
      case "encode" =>
        val bs=encode(Array.emptyByteArray,raw("units"));ctx.checkCancelled();val r=Obj("base64"->base.encodeToString(bs));ctx.beginPublication();r
      case "save" =>
        val p=path(raw);val ext=suffix(p);require(Set("png","jpg","jpeg","webp","cni")(ext))
        val image=binary(raw.obj.get("image_base64").fold("")(_.str));require((ext=="cni")==image.isEmpty)
        val bs=encode(image,raw("units"));ctx.checkCancelled()
        val stage=Files.createTempFile(p.getParent,".fastload-stage-",".tmp")
        try
          Using.resource(java.nio.channels.FileChannel.open(stage,java.nio.file.StandardOpenOption.WRITE)) { ch =>
            val b=java.nio.ByteBuffer.wrap(bs);while b.hasRemaining do ch.write(b)
            ch.force(true)
          }
          ctx.checkCancelled();ctx.beginPublication();Files.createLink(p,stage);Obj("path"->p.toString)
        finally Files.deleteIfExists(stage)
      case _ => throw IllegalArgumentException("Unknown Fastload operation")

  private def metadata(path:Path):String =
    // Read only bounded PNG text chunks. Non-PNG files still appear in gallery pages.
    Using.resource(new DataInputStream(Files.newInputStream(path))) { in =>
      if in.readLong()!=0x89504e470d0a1a0aL then "" else
        var stop=false;var result="";var chunks=0
        while !stop do
          val n=Integer.toUnsignedLong(in.readInt());val typ=in.readInt();chunks+=1
          require(chunks<=100000,"PNG chunk limit")
          if Set(0x74455874,0x7a545874,0x69545874)(typ) && n<=4000000 then
            val a=in.readNBytes(n.toInt);require(a.length==n)
            if a.length>11 && new String(a,0,11,StandardCharsets.ISO_8859_1)=="parameters\u0000" then
              def unzipped(offset:Int)=Using.resource(new InflaterInputStream(new ByteArrayInputStream(a,offset,a.length-offset)))(_.readNBytes(2000001))
              val content=typ match
                case 0x74455874 => a.drop(11)
                case 0x7a545874 => require(a(11)==0);unzipped(12)
                case _ =>
                  val lang=a.indexOf(0.toByte,13);require(lang>=13);val tr=a.indexOf(0.toByte,lang+1);require(tr>=0)
                  if a(11)==0 then a.drop(tr+1) else {require(a(11)==1 && a(12)==0);unzipped(tr+1)}
              require(content.length<=2000000,"Gallery metadata limit");result=new String(content,if typ==0x69545874 then StandardCharsets.UTF_8 else StandardCharsets.ISO_8859_1)
          else in.skipNBytes(n)
          in.skipNBytes(4);stop=typ==0x49454e44
        result
    }
  private def pairs(text:String):Vector[(String,String)] =
    val out=Vector.newBuilder[(String,String)]
    var pos=0
    while pos<text.length do
      val s=text.indexOf("ControlNet",pos);val a=if s<0 then -1 else text.indexOf('"',s+10);val b=if a<0 then -1 else text.indexOf('"',a+1)
      if b<0 then pos=text.length else
        if a!=s+10 then
          val g=text.substring(a+1,b);var i=0
          while i<g.length do
            val start=i;while i<g.length && g(i)!=':' && g(i)!=',' do i+=1
            if i<g.length && g(i)==':' then
              val k=g.substring(start,i).stripLeading();i+=1;while i<g.length && g(i).isWhitespace do i+=1
              val value=i;val close=if i<g.length && g(i)=='(' then g.indexOf(')',i) else -1
              if close>=0 && (close+1==g.length || g(close+1)==',') then i=close+1 else while i<g.length && g(i)!=',' do i+=1
              if k.nonEmpty && i>value then out+=k->g.substring(value,i)
            i+=1
        pos=b+1
    out.result()
  private def gallery(raw:Value,ctx:NativeWorkContext):Value =
    val root=path(raw).toRealPath();require(Files.isDirectory(root))
    val filters=raw.obj.get("filters").toVector.flatMap(_.arr.map { x=> require(x.arr.size==2);x(0).str->x(1).str }).toSet
    val choices=mutable.Map.empty[String,mutable.Set[String]];val paths=Vector.newBuilder[String]
    var scanned=0;var links=0;var oversize=0;var total=0L;var pairCount=0
    Using.resource(Files.walk(root)) { stream=>stream.iterator.asScala.foreach { p=>
      ctx.checkCancelled()
      if Files.isSymbolicLink(p) then links+=1
      else if Files.isRegularFile(p) then
        scanned+=1;require(scanned<=100000,"Gallery file count limit")
        val size=try Using.resource(new FileImageInputStream(p.toFile)){ input=>
          val readers=ImageIO.getImageReaders(input);if !readers.hasNext then None else
            val r=readers.next();try {r.setInput(input,true,true);Some(r.getWidth(0).toLong*r.getHeight(0))} finally r.dispose()
        } catch case NonFatal(_) => None
        size.foreach { pixels=>
          if pixels>32000000 then oversize+=1 else
            val info=try metadata(p) catch case _:java.io.IOException=> ""
            total+=info.length;require(total<=32000000,"Gallery metadata limit")
            val ps=pairs(info);pairCount+=ps.size;require(pairCount<=100000,"Gallery filter count limit")
            ps.foreach((k,v)=>choices.getOrElseUpdate(k,mutable.Set.empty)+=v)
            if filters.subsetOf(ps.toSet) then paths+=p.toString
        }
    }}
    val selected=paths.result().sorted;val pages=(selected.size+35)/36;val req=raw.obj.get("page").fold(1.0)(_.num);require(req.isWhole)
    val page=if pages==0 then 0 else if req<1 || req>pages then 1 else req.toInt
    Obj("paths"->Arr.from(selected.slice(((page-1)*36).max(0),page*36).map(ujson.Str(_))),"page"->page,"pages"->pages,
      "choices"->Obj.from(choices.toVector.sortBy(_._1).map((k,vs)=>k->Arr.from(vs.toVector.sorted.map(ujson.Str(_))))),"skipped_links"->links,"skipped_oversize"->oversize)
