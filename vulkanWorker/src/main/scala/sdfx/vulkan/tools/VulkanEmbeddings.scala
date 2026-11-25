package sdfx.vulkan.tools

import java.nio.file.{Files, Path}
import java.util.{Locale, UUID}
import scala.collection.mutable.ArrayBuffer
import scala.util.Using
import sdfx.core.protocol.LoadedModel
import NativeJson.*

private[vulkan] object VulkanEmbeddings:
  type Params = Map[String,NativeJson]
  private def string(p: Params, key: String, default: String = ""): String =
    p.get(key).fold(default)(_.asInstanceOf[Str].value)
  private def bool(p: Params, key: String, default: Boolean = false): Boolean =
    p.get(key).fold(default)(_.asInstanceOf[Bool].value)
  private def fail(what: String): Nothing = throw NativeBackendError("unsupported", what)
  private def staging(ctx: NativeWorkContext) = new NativeWorkContext(ctx.cancelled, () => ctx.checkCancelled(), ctx.event)
  private def plans(rows: Map[String,Vector[Array[Float]]]): Vector[(String,CheckpointIO.Plan)] =
    rows.toVector.sortBy(_._1).map { (name, vs) =>
      require(vs.nonEmpty && vs.size <= 4096 && vs.head.nonEmpty && vs.forall(r => r.length == vs.head.length && r.forall(_.isFinite)),
        "invalid embedding tensor")
      val key = if rows.size == 1 then "emb_params" else name
      key -> CheckpointIO.Plan("F32", Vector(vs.size,vs.head.length), vs.size.toLong * vs.head.length * 4,
        () => CheckpointIO.encode(vs.flatten.toArray, "F32"))
    }

  def execute(kind: String, raw: Params, model: LoadedModel, ctx: NativeWorkContext, opened: Seq[String] = Nil): Params =
    EmbeddingTables.withModel(model) { streams =>
      val requested = raw.get("embedding_paths").toVector.flatMap(_.asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value))
      val catalog = (opened ++ requested).distinct
      val p = raw.updated("embedding_paths",Arr(catalog.map(Str(_)).toVector))
      kind match
        case "embedding_inspector" => string(p,"operation") match
          case "list" =>
            val info = opened.map { file =>
              val data = EmbeddingTables.readVectors(Path.of(file),streams)
              NativeEmbeddingInspector.LoadedEmbedding(sdfx.vulkan.VulkanConditioning.stem(file),Path.of(file),data("clip_l").size,
                data("clip_l").head.length,data.get("clip_g").map(_.head.length))
            }
            NativeEmbeddingInspector.list(p,info.toVector,ctx)
          case "inspect" => NativeEmbeddingInspector.inspect(p,streams,ctx)
          case "export" => NativeEmbeddingInspector.exportVectors(p,streams,ctx)
          case "save" =>
            val output = Path.of(string(p,"output")).toAbsolutePath.normalize()
            if !output.toString.toLowerCase(Locale.ROOT).endsWith(".safetensors") then fail("Mixer export requires .safetensors")
            val mixed = NativeEmbeddingInspector.prepareMix(p,streams,ctx).map((k,v) => k -> v.map(_.toArray))
            require(!Files.exists(output) || !Files.isSameFile(output, Path.of(model.path)),
              "cannot overwrite the active checkpoint")
            CheckpointIO.write(output,plans(mixed),Map.empty,bool(p,"overwrite"),ctx,"embedding_save")
            Map("path" -> Str(output.toString), "paths" -> Arr(Vector(Str(output.toString))))
          case _ => NativeEmbeddingInspector.execute(p,streams.map((key,t,_) => key -> t),ctx)
        case "embedding_editor" =>
          val selected = catalog.filter(path => sdfx.vulkan.VulkanConditioning.stem(path).equalsIgnoreCase(string(p,"name")))
          val resolved = if string(p,"operation") == "open" && !p.contains("path") && selected.size == 1 then p.updated("path",Str(selected.head)) else p
          editor(resolved,model,streams,ctx)
        case "embedding_merge" => merge(p,streams,ctx)
        case _ => fail(kind)
    }

  private def editor(p: Params, model: LoadedModel, streams: EmbeddingTables.Streams, ctx: NativeWorkContext): Params =
    val op = string(p,"operation")
    val name = string(p,"name")
    val draft = p.get("draft").collect { case Obj(fields) => fields }.getOrElse(Map.empty)
    val rawPath = string(p,"path",string(draft,"path"))
    require(rawPath.nonEmpty && name.nonEmpty, "Select an embedding file to edit")
    val path = Path.of(rawPath).toRealPath()
    require(path.getFileName.toString.replaceFirst("\\.[^.]+$","").equalsIgnoreCase(name), "embedding selection changed")
    val result = NativeEmbeddingEditor.open(name,path,model.modelToken.get,string(p,"guidance"),streams,ctx)
    if op == "open" then return result
    require(Seq("name","model_token","path","file_identity").forall(k => result.get(k) == draft.get(k)),
      "embedding or checkpoint changed; reopen the embedding")
    if op == "guidance" then return result.filter((k,_) => Set("ranges","guidance")(k))
    if op != "save" then fail(s"Unsupported editor operation: $op")
    require(bool(p,"overwrite"), "Explicit overwrite consent is required")
    val expected = result("vectors").asInstanceOf[Obj].fields
    val edited = draft("vectors").asInstanceOf[Obj].fields
    require(edited.keySet == expected.keySet, "Editor encoder set changed")
    val values = edited.map { (key, value) =>
      val original = expected(key).asInstanceOf[Arr].values
      val rows = value.asInstanceOf[Arr].values
      require(rows.size == original.size, "Editor row count changed")
      key -> rows.zip(original).map { (row, before) =>
        val floats = row.asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
        require(floats.length == before.asInstanceOf[Arr].values.size && floats.forall(_.isFinite), "invalid editor values")
        floats
      }
    }
    val suffix = UUID.randomUUID().toString
    val stage = path.resolveSibling(s".${path.getFileName}.editing-$suffix")
    val backup = path.resolveSibling(s"${path.getFileName}.before-edit-$suffix.backup")
    val metadata = Using.resource(Files.newInputStream(path)) { input =>
      NativeMetadataUtils.parseHeader(input,Files.size(path))._1.get("__metadata__").collect {
        case Obj(fields) => fields.collect { case (k,Str(v)) => k -> v }
      }.getOrElse(Map.empty)
    }
    val identity = result("file_identity").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
    try
      Using.resource(CheckpointIO.open(path)) { source =>
        val updated = source.tensors.toVector.map { (key,t) =>
          val stream = if source.tensors.size == 1 && values.size == 1 then "clip_l" else key
          key -> values.get(stream).fold(CheckpointIO.copy(source,t)) { xs =>
            require(Set("F16","BF16","F32")(t.dtype), s"Unsupported editor storage: ${t.dtype}")
            CheckpointIO.converted(t,t.dtype)(xs.flatten.toArray)
          }
        }
        CheckpointIO.write(stage,updated,metadata,false,staging(ctx),"embedding_stage")
      }
      require(NativeEmbeddingEditor.fingerprint(path,ctx) == identity, "embedding changed before save")
      ctx.beginPublication()
      Files.move(path,backup)
      try
        require(NativeEmbeddingEditor.fingerprint(backup,ctx).last == identity.last,"embedding changed before publication")
        Files.createLink(path,stage)
      catch case e: Throwable =>
        try Files.createLink(path,backup) catch case restore: Throwable => e.addSuppressed(restore)
        throw e
      Map("path" -> Str(path.toString),"backup" -> Str(backup.toString),"reloaded" -> Bool(false))
    finally Files.deleteIfExists(stage)

  private def merge(p: Params, streams: EmbeddingTables.Streams, ctx: NativeWorkContext): Params =
    val op = string(p,"operation","inspect")
    require(Set("inspect","save","restore")(op), "Unknown merge operation")
    val text = string(p,"text")
    val restored = """<(\s*'[^']*'\s*)=\s*([^>]+)>""".r.replaceAllIn(text, m =>
      java.util.regex.Matcher.quoteReplacement("<" + m.group(1).trim + ">"))
    if op == "restore" || restored != text then
      return Map("text" -> Str(restored),"streams" -> Arr(Vector.empty),"paths" -> Arr(Vector.empty),"active" -> Bool(false))
    val names = streams.map(_._1)
    val pathsIn = p.get("embedding_paths").toVector.flatMap(_.asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value))
    val group = string(p,"grouping","parts")
    require(Set("none","comma","parts","words","tokens","vectors")(group), "Unknown grouping")
    case class Part(text: String, vectors: EmbeddingMerge.StreamVectors, ids: Vector[Vector[NativeJson]], comma: Int, derived: Boolean)
    val parts = Vector.newBuilder[Part]
    var comma = 0
    def append(fragment: String, derived: Boolean): Unit =
      if fragment.trim.nonEmpty then
        val value = if derived && !text.trim.startsWith("'") then fragment.substring(1,fragment.length-1) else fragment
        val vectors = resolve(value,streams,pathsIn,ctx)
        val external = pathsIn.exists(path => sdfx.vulkan.VulkanConditioning.stem(path).equalsIgnoreCase(value.trim))
        val ids = streams.zip(vectors).map { case ((_,tok,_),vs) =>
          if derived || external then vs.indices.map(i => Str(s"*_${i+1}")).toVector
          else tok.encode(value,false).map(id=>Number(BigDecimal(id))).toVector
        }
        if vectors.exists(_.nonEmpty) then parts += Part(fragment,vectors,ids,comma,derived || external)
    def plain(fragment: String): Unit =
      val chunks = if group == "comma" then fragment.split(",",-1).toVector else Vector(fragment)
      chunks.zipWithIndex.foreach { (chunk,i) =>
        if i > 0 then comma += 1
        if group == "words" then chunk.split("\\s+").foreach(append(_,false)) else append(chunk,false)
      }
    if text.trim.startsWith("'") then append(text,true)
    else
      var cursor = 0
      EmbeddingMerge.extractSpans(text).foreach { expression =>
        plain(text.substring(cursor,expression.start))
        append(expression.original,true)
        cursor = expression.end
      }
      plain(text.substring(cursor))
    val parsed = parts.result()
    val rows = names.indices.map(i => parsed.flatMap(_.vectors(i))).toVector
    require(rows.forall(_.size <= EmbeddingMerge.MaxRows), "Embedding row limit exceeded")
    def stats(vs: Vector[Array[Float]]): NativeJson =
      val count = vs.iterator.map(_.length.toLong).sum
      val sum = vs.iterator.flatMap(_.iterator).map(_.toDouble).sum
      val mean = if count > 0 then sum/count else 0.0
      val dev = vs.iterator.flatMap(_.iterator).map(x => math.pow(x-mean,2)).sum
      val norm = if vs.isEmpty then 0.0 else if vs.size == 1 then
        math.sqrt(vs.head.iterator.map(x => x.toDouble * x).sum)
      else
        ctx.checkCancelled()
        val matrix = new org.apache.commons.math3.linear.Array2DRowRealMatrix(vs.map(_.map(_.toDouble)).toArray, false)
        new org.apache.commons.math3.linear.SingularValueDecomposition(matrix).getNorm
      Obj(Map("min" -> Number(vs.iterator.map(_.min.toDouble).minOption.getOrElse(0.0)),
        "max" -> Number(vs.iterator.map(_.max.toDouble).maxOption.getOrElse(0.0)), "sum" -> Number(sum),
        "std" -> (if count > 1 then Number(math.sqrt(dev/(count-1))) else Null),
        "abs_sum" -> Number(vs.iterator.flatMap(_.iterator).map(x=>math.abs(x.toDouble)).sum), "norm" -> Number(norm)))
    val reports = names.zip(rows).zipWithIndex.map { case ((key,vs),stream) =>
      val clusters = group match
        case "none" => Vector(parsed).filter(_.nonEmpty)
        case "comma" => parsed.groupBy(_.comma).toVector.sortBy(_._1).map(_._2)
        case _ => parsed.map(Vector(_))
      var position = 1
      def row(vectors: Vector[Array[Float]], description: String, ids: Vector[NativeJson]): NativeJson =
        val start = position
        position += vectors.size
        Obj(Map("start" -> Number(start),"end" -> Number(position-1),"count" -> Number(vectors.size),
          "text" -> Str(description),"tokens" -> Arr(ids),"statistics" -> stats(vectors)))
      val items = clusters.flatMap { parts =>
        val vectors = parts.flatMap(_.vectors(stream))
        val tokens = parts.flatMap(_.ids(stream))
        val desc = parts.map(_.text).mkString
        if text.trim.startsWith("'") || group == "vectors" || (group == "tokens" && !parts.head.derived) then
          vectors.zip(tokens).map((v,id) => row(Vector(v),desc,Vector(id)))
        else Vector(row(vectors,desc,tokens))
      }
      Obj(Map("name" -> Str(key),"count" -> Number(vs.size),"rows" -> Arr(items),"total" -> stats(vs)))
    }
    val paths = if op != "save" then Vector.empty else
      if !bool(p,"safetensors",true) then fail("Embedding merge export requires safetensors")
      val root = string(p,"output_dir")
      val name = string(p,"name").stripSuffix(".safetensors")
      require(root.nonEmpty && name.nonEmpty && name.length <= 200 && name.forall(c => c.isLetterOrDigit || "_.- ".contains(c)) &&
        !name.startsWith(".") && !name.endsWith(" ") && !name.endsWith("."), "Invalid embedding output name")
      val folder = Files.createDirectories(Path.of(root).toAbsolutePath.normalize())
      val output = folder.resolve(s"$name.safetensors")
      val source = names.zip(rows).toMap
      val left = source("clip_l")
      val alternate = Option.when(left.exists(_.exists(_ != 0f))) {
        val dual = source.contains("clip_g")
        val sub = Files.createDirectories(folder.resolve(if dual then "sd1" else "sdxl")).toRealPath()
        require(sub.startsWith(folder.toRealPath()),"Conversion directory escapes the embedding folder")
        val converted = if dual then Map("clip_l" -> left) else Map("clip_l" -> left,"clip_g" -> left.map(_ => new Array[Float](1280)))
        sub.resolve(output.getFileName) -> converted
      }
      val writes = Vector(output -> source) ++ alternate
      writes.foreach((path,_) => require(!Files.exists(path),s"Embedding already exists: $path"))
      val published = ArrayBuffer.empty[Path]
      try
        writes.foreach { (path,data) =>
          CheckpointIO.write(path,plans(data),Map.empty,false,ctx,"embedding_save")
          published += path
        }
      catch case e: Throwable =>
        published.foreach(Files.deleteIfExists)
        throw e
      published.map(p => Str(p.toString)).toVector
    Map("text" -> Str(text),"streams" -> Arr(reports),"paths" -> Arr(paths),"active" -> Bool(false))

  private def resolve(text: String, streams: EmbeddingTables.Streams, embeddingPaths: Seq[String], ctx: NativeWorkContext): EmbeddingMerge.StreamVectors =
    val names = streams.map(_._1)
    val widths = streams.map(_._3.width)
    def byText(s: String): EmbeddingMerge.StreamVectors =
      val matches = embeddingPaths.filter(path => sdfx.vulkan.VulkanConditioning.stem(path).equalsIgnoreCase(s.trim))
      require(matches.size <= 1, "Ambiguous embedding name")
      matches.headOption match
        case Some(file) =>
          val vectors = EmbeddingTables.readVectors(Path.of(file),streams)
          names.map(vectors)
        case None => streams.map { (_,tok,t) =>
          val ids = tok.encode(s,false)
          require(ids.length <= EmbeddingMerge.MaxRows,"Embedding row limit exceeded")
          ids.toVector.map(id => t.row(Math.toIntExact(id)))
        }
    def byId(spec: Either[Int,String]): EmbeddingMerge.StreamVectors = streams.map { (_,tok,t) =>
      val id = spec.fold(_.toLong, {
        case "start" => tok.bosId
        case "end" => tok.eosId
        case "pad" => tok.padId
        case other => throw IllegalArgumentException(s"Unknown token: $other")
      })
      Vector(t.row(Math.toIntExact(id)))
    }
    def expression(s: String) =
      ctx.checkCancelled()
      EmbeddingMerge.evaluate(EmbeddingMerge.parseExpression(s),names,widths,byText,byId)
    val rows = if text.trim.startsWith("'") then expression(text) else
      val buffers = names.map(_ => Vector.newBuilder[Array[Float]])
      def add(values: EmbeddingMerge.StreamVectors): Unit = buffers.zip(values).foreach((b,vs) => b ++= vs)
      var pos = 0
      EmbeddingMerge.extractSpans(text).foreach { span =>
        add(byText(text.substring(pos,span.start)))
        add(expression(span.expression))
        pos = span.end
      }
      add(byText(text.substring(pos)))
      buffers.map(_.result())
    require(rows.forall(_.size <= EmbeddingMerge.MaxRows), "Embedding row limit exceeded")
    rows

  /** Publish draft/expression vectors only in this request's private directory. */
  def withGeneration[A](raw: ujson.Value, model: => LoadedModel, cancelled: () => Boolean)(run: ujson.Value => A): A =
    val draft = raw.obj.get("embedding_editor_preview")
    val params = raw.obj.get("params")
    def promptFields(p: ujson.Value): Vector[(ujson.Value,String)] =
      (Vector(p) ++ p.objOpt.flatMap(_.get("hires")).filter(_.objOpt.nonEmpty)).flatMap { o =>
        Vector("prompt","negative_prompt").filter(o.obj.contains).map(o -> _)
      }
    if draft.isEmpty && !params.exists(p => p.objOpt.nonEmpty && promptFields(p).exists((o,k) => o(k).strOpt.exists(EmbeddingMerge.hasExpressions))) then
      return run(raw)
    val request = ujson.read(ujson.write(raw))
    request.obj.remove("embedding_editor_preview")
    val baseline = model
    val ctx = new NativeWorkContext(cancelled, () => (), (_,_) => ())
    val dir = Files.createTempDirectory("sdv-embedding-")
    val generated = ArrayBuffer.empty[Path]
    try
      var paths = request.obj.get("embedding_paths").toVector.flatMap(_.arr.map(_.str))
      EmbeddingTables.withModel(baseline) { streams =>
        def add(token: String, data: Map[String,Vector[Array[Float]]]): Unit =
          val file = dir.resolve(token + ".safetensors").normalize()
          require(file.getParent == dir,"invalid temporary embedding name")
          CheckpointIO.write(file,plans(data),Map.empty,false,ctx,"embedding_preview")
          generated += file
          paths :+= file.toString
        draft.foreach { d =>
          require(baseline.modelToken.contains(d("model_token").str),"Checkpoint changed; reopen the embedding")
          val source = Path.of(d("path").str).toRealPath()
          val token = d("name").str
          require(source.getFileName.toString.replaceFirst("\\.[^.]+$","").equalsIgnoreCase(token),"Invalid preview name")
          require(d("file_identity").arr.map(_.str).toVector == NativeEmbeddingEditor.fingerprint(source,ctx),"Embedding changed; reopen it")
          val original = EmbeddingTables.readVectors(source,streams)
          require(d("vectors").obj.keySet == original.keySet,"Invalid editor encoder set")
          val edited = original.map { (key, expected) =>
            val vs = d("vectors")(key).arr.toVector.map(_.arr.map(_.num.toFloat).toArray)
            require(vs.size == expected.size && vs.zip(expected).forall((a,b) => a.length == b.length && a.forall(_.isFinite)), "invalid editor vectors")
            key -> vs
          }
          paths = paths.filterNot(p => sdfx.vulkan.VulkanConditioning.stem(p).equalsIgnoreCase(token))
          add(token,edited)
        }
        val replacements = scala.collection.mutable.Map.empty[String,String]
        promptFields(request("params")).foreach { (obj,key) =>
          val original = obj(key).str
          val spans = EmbeddingMerge.extractSpans(original)
          if spans.nonEmpty then
            val result = new StringBuilder
            var cursor = 0
            spans.foreach { span =>
              ctx.checkCancelled()
              result.append(original.substring(cursor,span.start))
              val name = replacements.getOrElseUpdate(span.expression, {
                val token = "sdvmerge" + UUID.randomUUID().toString.replace("-","")
                val rows = resolve(span.expression,streams,paths,ctx)
                add(token,streams.map(_._1).zip(rows).toMap)
                token
              })
              result.append(" ").append(name).append(" ")
              cursor = span.end
            }
            obj(key) = result.append(original.substring(cursor)).toString
        }
      }
      request("embedding_paths") = ujson.Arr.from(paths.map(ujson.Str(_)))
      run(request)
    finally
      generated.foreach(Files.deleteIfExists)
      Files.deleteIfExists(dir)
