package sdfx.vulkan.tools

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{FileAlreadyExistsException, Files, Path}
import java.security.MessageDigest
import java.time.Duration
import scala.collection.mutable
import scala.io.Source
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.vulkan.{VulkanInference,VulkanProtocol}
import ujson.{Arr,Obj,Str,Value}

private[vulkan] object VulkanClip:
  val model = "ViT-H-14/laion2b_s32b_b79k"
  private val tablePrefix = "ViT-H-14_laion2b_s32b_b79k_"
  private val tableRepo = "https://huggingface.co/pharma/ci-preprocess/resolve/8304b4f0f2c02396589782b0beb71011975e4265/"
  private val visionUrl = "https://huggingface.co/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/model.safetensors"
  private val visionHash = "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030"
  private val categories = Vector(
    ("medium","mediums","3ed7eac346116bbf537a769336d959b2e9642a59b58507f76eff097da0391571"),
    ("artist","artists","6615dcfee70db0ae48256a34daa1e45109b6704e7eec296b5e1c154226a2fbc5"),
    ("movement","movements","61462de8a77f2ec8ad25cbab34c9bb52972423c26b9c8c84768baa63f61091bb"),
    ("trending","trendings","7d3b652df449e558dc1df6544145322506b7b3e8bdc786392cebc22ca1a262ea"),
    ("flavor","flavors","883b62589ba7e97b09096f5fbe89de5b70acf8c95ba342a9c773787a65a19df9"))
  private lazy val http = HttpClient.newBuilder.followRedirects(HttpClient.Redirect.NORMAL).connectTimeout(Duration.ofSeconds(15)).build()
  private val encoder = java.util.HexFormat.of()
  private def digest(bytes: Array[Byte]): String = encoder.formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))

  private[tools] def asset(dir: Path, name: String, url: String, sha: String, check: () => Unit, progress: (String,Int,Int) => Unit): Path =
    val dest = dir.resolve(name)
    if Files.isRegularFile(dest) then
      require(sdfx.core.imagebrowser.BrowserFiles.sha256(dest) == sha,s"Incorrect SHA-256 for $dest")
      return dest
    if Seq("SDFX_VULKAN_OFFLINE","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE").exists(k => sys.env.get(k).exists(v => Set("1","true","yes","on")(v.toLowerCase))) then
      VulkanProtocol.unsupported(s"downloading $name in offline mode; install the pinned file at $dest")
    progress("clip_interrogator_download",0,1)
    val temporary = Files.createTempFile(dir, ".clip-download-", ".part")
    try
      val req = HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofMinutes(30)).GET().build()
      val res = http.send(req,HttpResponse.BodyHandlers.ofInputStream())
      val hash = MessageDigest.getInstance("SHA-256")
      Using.resource(res.body()) { in =>
        require(res.statusCode() == 200, s"Cannot download $name (HTTP ${res.statusCode()})")
        Using.resource(Files.newOutputStream(temporary)) { out =>
          val buffer = new Array[Byte](1024*1024)
          var n = in.read(buffer)
          while n >= 0 do
            check()
            hash.update(buffer,0,n)
            out.write(buffer,0,n)
            n = in.read(buffer)
        }
      }
      require(encoder.formatHex(hash.digest()) == sha,s"Downloaded file digest mismatch for $name")
      check()
      try Files.move(temporary,dest)
      catch case _: FileAlreadyExistsException => require(sdfx.core.imagebrowser.BrowserFiles.sha256(dest) == sha,s"Conflicting file at $dest")
      dest
    finally Files.deleteIfExists(temporary)

  private def labels(kind: String): Vector[String] =
    if kind == "trendings" then
      // The pinned label cache predates the upstream spelling correction from "dribble" to "dribbble".
      val sites = Vector("Artstation","behance","cg society","cgsociety","deviantart","dribble","flickr","instagram",
        "pexels","pinterest","pixabay","pixiv","polycount","reddit","shutterstock","tumblr","unsplash","zbrush central")
      sites ++ sites.map("trending on " + _) ++ sites.map("featured on " + _) ++ sites.map(_ + " contest winner")
    else
      val values = Using.resource(Source.fromResource(s"clip-interrogator/$kind.txt")(scala.io.Codec.UTF8))(_.getLines().map(_.strip()).toVector)
      if kind == "artists" then values.map("by " + _) ++ values.map("inspired by " + _) else values

  private def rank(path: Path, names: Vector[String], image: Array[Float], check: () => Unit, count:Int=5, reverse:Boolean=false): Vector[(String,Double)] =
    Using.resource(CheckpointIO.open(path)) { file =>
      require(new String(file.raw(file.tensors("hash")),UTF_8) == digest(names.mkString(",").getBytes(UTF_8)),"CLIP label ordering does not match the cache")
      val tensor = file.tensors("embeds")
      require(tensor.shape == Vector(names.size.toLong,image.length.toLong) && tensor.dtype == "F16","Invalid CLIP text table")
      val table = new EmbeddingTables.Table(file,tensor)
      val best = mutable.PriorityQueue.empty[(Double,Int)](Ordering.by[(Double,Int),(Double,Int)](e => (-e._1,e._2)))
      for row <- names.indices do
        if (row & 127) == 0 then check()
        val values = table.row(row)
        var dot = 0.0
        var norm = 0.0
        var i = 0
        while i < values.length do
          dot += values(i)*image(i)
          norm += values(i).toDouble*values(i)
          i += 1
        val score = (if reverse then -dot else dot) / math.sqrt(norm)
        if score.isFinite && (best.size < count || score > best.head._1) then
          best.enqueue((score,row))
          if best.size > count then best.dequeue()
      best.dequeueAll.reverseIterator.map((score,row) => names(row) -> score).toVector
    }

  private lazy val tokenizer=ClipFastTokenizer.loadBundled()
  def modes:Vector[String]=
    val base=if VulkanBlip.available then Vector("classic","fast","caption","label") else Vector("label")
    base++Option.when(VulkanClipText.available && VulkanBlip.available)("best")++Option.when(VulkanClipText.available)("negative")
  private def truncate(prompt:String):String =
    val parts=prompt.split(", ",-1)
    var result=parts.head
    var i=1
    while i<parts.length && tokenizer.encode(result+parts(i)).length<77 do
      result+=", "+parts(i);i+=1
    result

  def execute(raw: Value, engine: VulkanInference, device: String, check: () => Unit, progress: (String,Int,Int) => Unit, publish: () => Unit = () => ()): Value =
    val op = raw.obj.get("operation").map(_.str).getOrElse("prompt")
    if op == "models" then return Obj("clip_models" -> Arr(model),"default_clip_model" -> model,
      "caption_models" -> Arr.from(VulkanBlip.known.map(ujson.Str(_))),"default_caption_model" -> VulkanBlip.known.headOption.getOrElse("none"),"modes" -> Arr.from(modes.map(ujson.Str(_))),
      "batch_output_modes" -> Arr.from(VulkanClipBatch.formats.map(ujson.Str(_))),"openclip_status" -> Obj("state" -> "native_vulkan"))
    if op == "batch" then return VulkanClipBatch.execute(raw,engine,device,check,publish,progress)
    if op == "unload" then return Obj("unloaded" -> engine.unloadClipVision())
    if op != "analyze" && op != "prompt" then VulkanProtocol.unsupported(s"CLIP operation $op")
    if raw.obj.get("clip_model").exists(_.str != model) then VulkanProtocol.unsupported("this CLIP vision architecture")
    val mode=raw.obj.get("mode").fold(modes.head)(_.str)
    if op=="prompt" && !modes.contains(mode) then VulkanProtocol.unsupported(s"CLIP mode $mode")
    val useCaption=op=="prompt" && mode!="label" && mode!="negative"
    val captionModel=raw.obj.get("caption_model").fold(if useCaption then VulkanBlip.known.headOption.getOrElse("none") else "none")(_.str)
    if useCaption && !VulkanBlip.known.contains(captionModel) then VulkanProtocol.unsupported(s"caption model $captionModel")
    val path=Path.of(raw("image_path").str)
    val caption=if useCaption then
      val adapter=engine.devices().asScala.find(_.id()==device).getOrElse(throw IllegalArgumentException("Unknown Vulkan adapter"))
      VulkanBlip.caption(path,adapter.name(),check,raw.obj.get("caption_max_length").fold(32)(_.num.toInt),captionModel)
    else ""
    if op=="prompt" && mode=="caption" then return Obj("prompt"->caption,"clip_model"->model,"caption_model"->captionModel,"mode"->mode)
    val count=if op=="prompt" && Set("fast","classic","best","negative")(mode) then raw.obj.get("max_flavors").fold(if mode=="classic" then 3 else 32)(_.num.toInt) else 5
    require(count>=1 && count<=64,"max_flavors must be between 1 and 64")
    val image = DiffusionImageInput.readRgb(Path.of(raw("image_path").str))
    val cache = Files.createDirectories(Path.of(sys.env.get("SDFX_VULKAN_CLIP_DIR")
      .orElse(raw.obj.get("cache_dir").map(_.str)).getOrElse("config/clip-interrogator")).toAbsolutePath.normalize())
    val weights = asset(cache,"clip_vision_h.safetensors",visionUrl,visionHash,check,progress)
    check()
    def quote(value: String): String = ujson.write(Str(value))
    val graph = s"""fn main(input) {
      vision = load_vision_encoder(${quote(weights.toString)}, backend = ${quote(device)}, params_backend = ${quote(device)})
      return {embedding = encode_vision(vision, input, pooled = true)}
    }"""
    val embedding = engine.clipVision(graph,image,raw.obj.get("keep_loaded").exists(_.bool))
    val norm = math.sqrt(embedding.iterator.map(x => x.toDouble*x).sum)
    require(norm > 0 && norm.isFinite,"Invalid CLIP vision projection")
    val unit = embedding.map(x => (x/norm).toFloat)
    val chained=op=="prompt" && Set("best","negative")(mode)
    val intermediate=if chained && engine.memory(device).total()<=12L*1024*1024*1024 then 1024 else 2048
    val tables=if op=="prompt" && mode=="negative" then categories.filter(_._1=="flavor") else categories
    val results = tables.zipWithIndex.map { case ((key,kind,sha),i) =>
      progress("clip_interrogator_rank",i,categories.size)
      val name = tablePrefix + kind + ".safetensors"
      key -> rank(asset(cache,name,tableRepo+name,sha,check,progress),labels(kind),unit,check,if chained then intermediate else count,op=="prompt" && mode=="negative")
    }
    progress("clip_interrogator_rank",categories.size,categories.size)
    if op == "analyze" then Obj.from(results.map((k,rows) => k -> Obj.from(rows.map((label,score) => label -> ujson.Num(score)))))
    else if Set("classic","fast","best","negative")(mode) then
      val top=results.toMap
      val all=results.flatMap(_._2).sortBy(-_._2)
      def fast=truncate(caption+", "+all.take(count).map(_._1).mkString(", "))
      def classic=
        val medium=top("medium").head._1
        val prefix=if caption.startsWith(medium) then caption else caption+", "+medium
        truncate(prefix+" "+top("artist").head._1+", "+top("trending").head._1+", "+top("movement").head._1+", "+top("flavor").take(count).map(_._1).mkString(", "))
      val prompt=mode match
        case "fast"=>fast
        case "classic"=>classic
        case _=>
          val negative=mode=="negative"
          val phrases=if negative then top("flavor").map(_._1)++labels("negative") else all.take(intermediate).map(_._1)
          val adapter=engine.devices().asScala.find(_.id()==device).getOrElse(throw IllegalArgumentException("Unknown Vulkan adapter"))
          VulkanClipText.withEncoder(cache,adapter.name(),check,progress) { encoder =>
            val minimum=raw.obj.get("min_flavors").fold(8)(_.num.toInt)
            require(minimum>=0 && minimum<=64,"min_flavors must be between 0 and 64")
            def sim(text:String)=encoder.score(text,unit)
            val candidates=mutable.TreeSet.from(phrases)
            def pick(prefix:String,reverse:Boolean):(String,Double) =
              var selected="";var score=Double.NegativeInfinity
              for term<-candidates do
                val text=if prefix.isEmpty then term else prefix+", "+term
                val value=sim(text)*(if reverse then -1 else 1)
                if value>score then {selected=term;score=value}
              selected->score
            val (initial,startScore)=if negative then
              val (chosen,negated)=pick("",true)
              candidates-=chosen
              chosen -> -negated
            else caption->sim(caption)
            var best=initial;var bestScore=startScore
            var current=initial;var currentScore=startScore
            var i=0;var more=true
            while i<count && more && candidates.nonEmpty do
              check();progress("clip_interrogator_chain",i,count)
              val (term,score)=pick(current,negative)
              val next=current+", "+term
              if score>bestScore then {best=next;bestScore=score}
              if score>currentScore || i<minimum then {current=next;currentScore=score} else more=false
              if tokenizer.encode(current).length>=77 then more=false
              candidates-=term;i+=1
            if negative then best else Vector(caption,classic,fast,best).maxBy(sim)
          }
      Obj("prompt"->prompt,"clip_model"->model,"caption_model"->(if useCaption then captionModel else "none"),"mode"->mode)
    else
      val (label,score) = results.flatMap(_._2).maxBy(_._2)
      Obj("prompt" -> label,"similarity" -> score,"clip_model" -> model,"caption_model" -> "none","mode" -> "label")
