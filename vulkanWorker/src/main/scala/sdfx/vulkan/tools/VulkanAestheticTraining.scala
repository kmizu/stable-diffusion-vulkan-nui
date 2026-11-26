package sdfx.vulkan.tools

import java.io.{BufferedInputStream, BufferedOutputStream, DataOutputStream}
import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets.UTF_8
import java.security.MessageDigest
import java.util.HexFormat
import scala.collection.mutable.Builder
import scala.util.Using
import scala.util.boundary, boundary.break
import sdfx.core.model.LoraRef
import sdfx.vulkan.{VulkanInference, VulkanPromptAttention, VulkanProtocol, VulkanComposableLora}
import ujson.Value

private[vulkan] object VulkanAestheticTraining:
  val id = "aesthetic_gradients"
  private val hidden = 768
  private val tokenCount = 77
  private lazy val tokenizer = ClipFastTokenizer.loadBundled()

  final case class Settings(embedding: String, sha256: String, weight: Double, steps: Int, rate: Float,
      negative: Boolean, slerp: Boolean, angle: Double, style: String):
    def active: Boolean = weight != 0 && steps != 0 && rate != 0

  final case class Conditioning(settings: Settings, sdxl: Boolean, trainedEncoder: (String, Seq[LoraRef]) => String,
      components: String => (Boolean, Boolean)):
    def encode(label: String, text: String, caption: String, options: String, negative: Boolean,
        loras: Seq[LoraRef], lines: Builder[String, Vector[String]]): String =
      val blankNegative = negative && caption.trim.isEmpty
      val quoted = ujson.write(ujson.Str(if blankNegative then "" else caption))
      def encoded(encoder: String): String = s"encode_prompt($encoder, $quoted, $options)"
      if settings.negative != negative || blankNegative then
        lines += s"$label = ${encoded(text)}"
      else
        val base = s"${label}_base"
        val trained = s"${label}_trained"
        lines += s"$base = ${encoded(text)}"
        lines += s"$trained = ${encoded(trainedEncoder(caption, loras))}"
        if sdxl then
          val left = s"${label}_l"
          lines += s"$left = tensor::lerp(tensor::crop($base.crossattn, axes = [0], start = [0], size = [768]), tensor::crop($trained.crossattn, axes = [0], start = [0], size = [768]), ${settings.weight})"
          lines += s"$label = {crossattn = tensor::replace_slice($base.crossattn, $left, axis = 0, start = 0), vector = $base.vector}"
        else lines += s"$label = {crossattn = tensor::lerp($base.crossattn, $trained.crossattn, ${settings.weight})}"
      label

  def loraComponents(path: String): (Boolean, Boolean) =
    if path.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      Using.resource(CheckpointIO.open(Path.of(path)))(file => VulkanComposableLora.components(file.tensors.keys))
    else (true, true)

  def settings(value: Value): Settings =
    def number(key: String, default: Double): Double =
      val selected = value.obj.get(key).filterNot(_ == ujson.Null).fold(default)(v =>
        v.numOpt.orElse(v.strOpt.flatMap(_.toDoubleOption)).getOrElse(VulkanProtocol.invalid(s"Invalid aesthetic $key")))
      if !selected.isFinite then VulkanProtocol.invalid(s"Aesthetic $key must be finite")
      selected
    def string(key: String): String = value.obj.get(key).flatMap(_.strOpt).getOrElse("")
    def flag(key: String): Boolean = value.obj.get(key).exists(_.boolOpt.contains(true))
    val steps = number("steps", 20)
    val weight = number("weight", .9)
    val angle = number("slerp_angle", .1)
    val rate = number("lr", .0001).toFloat
    val embedding = string("embedding").trim
    val hash = string("embedding_sha256").trim.toLowerCase(java.util.Locale.ROOT)
    if !steps.isWhole || steps < 0 || steps > 200 || weight < 0 || weight > 1 || angle < 0 || angle > 1 ||
        rate < 0 || !rate.isFinite || (hash.nonEmpty && !hash.matches("[0-9a-f]{64}")) then
      VulkanProtocol.invalid("Invalid Aesthetic Gradients configuration")
    val out = Settings(embedding, hash, weight, steps.toInt, rate, flag("negative"), flag("slerp"), angle, string("text"))
    if out.active && embedding.isEmpty then VulkanProtocol.invalid("Aesthetic Gradients requires an embedding")
    out

  private case class Parameter(name: String, shape: Vector[Long], reference: Boolean = false):
    def bytes: Int = (shape.product * 4).toInt

  private val parameters = Vector(
    Parameter("text_model.embeddings.token_embedding.weight", Vector(49408, 768)),
    Parameter("text_model.embeddings.position_embedding.weight", Vector(77, 768))) ++
    (0 until 12).flatMap { index =>
      val groups = Vector("layer_norm1" -> Vector(768L)) ++
        Vector("q_proj", "k_proj", "v_proj", "out_proj").map(name => s"self_attn.$name" -> Vector(768L, 768L)) ++
        Vector("layer_norm2" -> Vector(768L), "mlp.fc1" -> Vector(3072L, 768L), "mlp.fc2" -> Vector(768L, 3072L))
      groups.flatMap { (name, shape) =>
        val prefix = s"text_model.encoder.layers.$index.$name"
        Vector(Parameter(s"$prefix.weight", shape, index == 11), Parameter(s"$prefix.bias", shape.take(1), index == 11))
      }
    } ++ Vector("weight", "bias").map(name => Parameter(s"text_model.final_layer_norm.$name", Vector(768), true))

  private val projection = Parameter("text_projection.weight", Vector(768, 768), true)

  private def embedding(config: Settings, folder: Path): (Array[Float], String) =
    val raw = Path.of(config.embedding)
    val paths = Vector(raw) ++ Option.when(!raw.isAbsolute)(VulkanAesthetic.folders.flatMap(root =>
      Vector("", ".pt", ".safetensors").map(extension => root.resolve(config.embedding + extension)))).toVector.flatten
    val source = paths.find(Files.isRegularFile(_)).getOrElse(VulkanProtocol.invalid(s"Aesthetic embedding not found: $raw"))
    val bytes = Using.resource(Files.newInputStream(source))(_.readNBytes(64 * 1024 * 1024 + 1))
    if bytes.length > 64 * 1024 * 1024 then VulkanProtocol.invalid("Aesthetic embedding exceeds 64 MiB")
    val digest = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
    if config.sha256.nonEmpty && config.sha256 != digest then VulkanProtocol.invalid("Aesthetic embedding fingerprint mismatch")
    val safeTensor = source.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")
    val snapshot = Files.createTempFile(folder, "embedding-", if safeTensor then ".safetensors" else ".pt")
    Files.write(snapshot, bytes)
    val values =
      if safeTensor then Using.resource(CheckpointIO.open(snapshot)) { file =>
        val tensor = file.tensors.get("embedding").orElse(file.tensors.get("aesthetic_embedding"))
          .orElse(Option.when(file.tensors.size == 1)(file.tensors.values.head))
          .getOrElse(VulkanProtocol.invalid("No aesthetic embedding tensor found"))
        if tensor.elements != hidden then VulkanProtocol.invalid("Aesthetic embedding requires 768 values")
        file.floats(tensor)
      }
      else VulkanClipTorchFile.readAesthetic(snapshot).data
    val norm = math.sqrt(values.iterator.map(x => x.toDouble * x).sum)
    if values.length != hidden || !norm.isFinite || norm == 0 then
      VulkanProtocol.invalid("Expected a finite, nonzero 768-dimensional aesthetic embedding")
    values.map(x => (x / norm).toFloat) -> digest

  def chunks(prompt: String): Vector[Vector[Int]] = promptChunks(prompt, firstOnly = false)

  private def promptChunks(prompt: String, firstOnly: Boolean): Vector[Vector[Int]] = boundary:
    val comma = tokenizer.encode(",", addSpecialTokens = false).headOption.map(_.toInt)
    val result = scala.collection.mutable.ArrayBuffer.empty[Vector[Int]]
    var pending = Vector.empty[Int]
    var lastComma = -1
    def flush(): Unit =
      result += ((tokenizer.bosId.toInt +: pending).padTo(tokenCount - 1, tokenizer.eosId.toInt) :+ tokenizer.eosId.toInt)
      if firstOnly then break(result.toVector)
      if result.size > 16 then VulkanProtocol.unsupported("more than 16 CLIP chunks in an aesthetic training prompt")
      pending = Vector.empty
      lastComma = -1
    for span <- VulkanPromptAttention.parse(prompt) do
      if span.text == "BREAK" && span.weight == -1 then flush()
      else for id <- tokenizer.encode(span.text, addSpecialTokens = false).iterator.map(_.toInt) do
        if comma.contains(id) then lastComma = pending.size
        else if pending.size == 75 && lastComma >= 0 && 75 - lastComma <= 20 then
          val (keep, relocate) = pending.splitAt(lastComma + 1)
          pending = keep
          flush()
          pending = relocate
        if pending.size == 75 then flush()
        pending :+= id
    if pending.nonEmpty || result.isEmpty then flush()
    result.toVector

  def styleChunk(config: Settings): Option[Vector[Int]] =
    Option.when(config.slerp && config.angle.toFloat > 0 && config.style.trim.nonEmpty)(
      promptChunks(config.style, firstOnly = true).head)

  def train(engine: VulkanInference, device: String, checkpoint: Path, sdxl: Boolean, prompt: String,
      config: Settings, folder: Path, loras: Seq[(String, Float)], cancelled: () => Boolean,
      progress: (Int, Float) => Unit): (Path, String) =
    val safeTensor = checkpoint.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")
    if !safeTensor && !engine.supportsAestheticCheckpointExtraction() then
      VulkanProtocol.unsupported("Aesthetic Gradients with a checkpoint outside SafeTensors")
    val style = styleChunk(config)
    if loras.nonEmpty && !engine.supportsAestheticLora() then
      VulkanProtocol.unsupported("Aesthetic Gradients with LoRA in this Vulkan runtime")
    if loras.size > 64 then VulkanProtocol.unsupported("Aesthetic Gradients with more than 64 LoRAs")
    if style.nonEmpty && !engine.supportsAestheticSlerp() then
      VulkanProtocol.unsupported("Aesthetic Gradients slerp with this Vulkan runtime")
    val context = new NativeWorkContext(cancelled, () => (), (_, _) => ())
    val (target, digest) = embedding(config, folder)
    val reference = VulkanAesthetic.referenceWeights(VulkanAesthetic.cache, () => context.checkCancelled(), (_, _, _) => ())
    val input = Files.createTempFile(folder, "clip-training-", ".bin")
    val trained = Files.createTempFile(folder, "clip-trained-", ".bin")
    val prefix = if sdxl then "conditioner.embedders.0.transformer." else "cond_stage_model.transformer."
    val ids = chunks(prompt)
    Using.Manager { use =>
      val source = Option.when(safeTensor)(use(CheckpointIO.open(checkpoint)))
      val requested = parameters.filterNot(parameter => sdxl && parameter.reference)
      val extract = engine.supportsAestheticCheckpointExtraction() &&
        source.forall(file => requested.exists(p => !file.tensors.contains(prefix + p.name)))
      val raw = Option.when(extract) {
        val names = requested.map(p => s"cond_stage_model.transformer.${p.name}").toArray
        val dimensions = requested.flatMap(_.shape.reverse.padTo(2, 1L)).toArray
        val path = Files.createTempFile(folder, "clip-checkpoint-", ".bin")
        engine.extractAestheticClip(checkpoint, path, names, dimensions, (_, _) => !cancelled())
        use(new BufferedInputStream(Files.newInputStream(path)))
      }
      val stock = use(CheckpointIO.open(reference))
      val stream = use(new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(input))))
      def writeInt(value: Int): Unit = stream.writeInt(Integer.reverseBytes(value))
      def writeTensor(file: CheckpointIO.File, key: String, parameter: Parameter): Unit =
        context.checkCancelled()
        val value = file.tensors.getOrElse(key, VulkanProtocol.invalid(s"Missing CLIP-L parameter $key"))
        if value.shape != parameter.shape || !CheckpointIO.floatTypes(value.dtype) then
          VulkanProtocol.invalid(s"Invalid CLIP-L parameter $key")
        stream.write(CheckpointIO.cast(file, value, "F32").read())
      writeInt(ids.size)
      ids.flatten.foreach(writeInt)
      ids.zipWithIndex.foreach((chunk, i) => writeInt(i * tokenCount + chunk.indexOf(tokenizer.eosId.toInt)))
      stream.write(CheckpointIO.encode(target, "F32"))
      parameters.foreach { parameter =>
        if sdxl && parameter.reference then writeTensor(stock, parameter.name, parameter)
        else raw match
          case Some(reader) =>
            context.checkCancelled()
            val bytes = reader.readNBytes(parameter.bytes)
            if bytes.length != parameter.bytes then VulkanProtocol.invalid(s"Incomplete CLIP-L parameter ${parameter.name}")
            stream.write(bytes)
          case None => writeTensor(source.get, prefix + parameter.name, parameter)
      }
      writeTensor(stock, projection.name, projection)
      style.foreach { tokens =>
        writeInt(java.lang.Float.floatToRawIntBits(config.angle.toFloat))
        tokens.foreach(writeInt)
        writeInt(tokens.indexOf(tokenizer.eosId.toInt))
      }
      if loras.nonEmpty then
        if style.isEmpty then writeInt(0)
        writeInt(if sdxl then 1 else 0)
        writeInt(loras.size)
        loras.foreach { (path, weight) =>
          val bytes = path.getBytes(UTF_8)
          writeInt(java.lang.Float.floatToRawIntBits(weight))
          writeInt(bytes.length)
          stream.write(bytes)
        }
    }.get
    engine.trainAesthetic(device, input, trained, config.steps, config.rate, (step, loss) =>
      if step > 0 && !cancelled() then progress(step, loss)
      !cancelled())
    val output = Files.createTempFile(folder, "trained-clip-l-", ".safetensors")
    Using.resource(new BufferedInputStream(Files.newInputStream(trained))) { input =>
      val plans = parameters.map { p =>
        p.name -> CheckpointIO.Plan("F32", p.shape, p.bytes.toLong, () => input.readNBytes(p.bytes))
      }
      CheckpointIO.write(output, plans, Map.empty, overwrite = true, context, id)
    }
    output -> digest
