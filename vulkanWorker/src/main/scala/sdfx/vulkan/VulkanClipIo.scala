package sdfx.vulkan

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.{US_ASCII, UTF_8}
import java.nio.file.{FileAlreadyExistsException, Files, Path, StandardCopyOption}
import java.util.Locale
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.model.Csv
import sdfx.vulkan.tools.VulkanClipTorchFile
import ujson.Value

private[vulkan] object VulkanClipIo:
  val id = "clip_io"
  private val Limit = 256L * 1024 * 1024
  private val MaxTokens = 64 * 77
  private def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"Clip I/O: $detail")
  private def quoted(s: String): String = ujson.write(ujson.Str(s))

  private def directory: Path =
    sys.env.get("SDFX_CLIP_IO_DIR").filter(_.nonEmpty).map(Path.of(_)).getOrElse {
      val cache = Path.of(sys.props("user.home"), ".cache")
      val current = cache.resolve("stable-diffusion-nui/clip_io/conditioning")
      val legacy = cache.resolve("stablediffusion-nui/clip_io/conditioning")
      if !Files.exists(current) && Files.exists(legacy) then legacy else current
    }

  private case class Matrix(tokens: Int, width: Int, values: Array[Float])
  private val NpyMagic = Array(0x93.toByte, 'N'.toByte, 'U'.toByte, 'M'.toByte, 'P'.toByte, 'Y'.toByte)
  private val Shape = """'shape'\s*:\s*\(([^)]*)\)""".r
  private def writeNpy(file: Path, matrix: Matrix): Unit =
    val dict = s"{'descr': '<f4', 'fortran_order': False, 'shape': (${matrix.tokens}, ${matrix.width}), }"
    val pad = (64 - (10 + dict.length + 1) % 64) % 64
    val header = (dict + " " * pad + "\n").getBytes(US_ASCII)
    val data = ByteBuffer.allocate(10 + header.length + matrix.values.length * 4).order(ByteOrder.LITTLE_ENDIAN)
    data.put(NpyMagic).put(1.toByte).put(0.toByte).putShort(header.length.toShort).put(header)
    data.asFloatBuffer().put(matrix.values)
    Files.write(file, data.array())

  private def readNpy(file: Path, width: Int): Matrix =
    val data = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN)
    val magic = new Array[Byte](6)
    data.get(magic)
    if !magic.sameElements(NpyMagic) then invalid("unexpected native tensor file")
    val version = data.get()
    data.get()
    val headerSize = if version == 1 then java.lang.Short.toUnsignedInt(data.getShort()) else data.getInt()
    if headerSize < 1 || headerSize > data.remaining() then invalid("invalid native tensor header")
    val headerBytes = new Array[Byte](headerSize)
    data.get(headerBytes)
    val header = new String(headerBytes, US_ASCII)
    val dims = Shape.findFirstMatchIn(header).toVector.flatMap(_.group(1).split(",").toVector.map(_.trim).filter(_.nonEmpty).map(_.toInt))
    if !header.contains("'<f4'") || dims.lastOption != Some(width) || dims.size < 2 ||
        dims.dropRight(2).exists(_ != 1) || dims(dims.size - 2) > MaxTokens ||
        dims(dims.size - 2) < 1 || data.remaining() != 4L * dims(dims.size - 2) * width then
      invalid("unexpected conditioning tensor shape or type")
    val tokens = dims(dims.size - 2)
    val values = new Array[Float](tokens * width)
    data.asFloatBuffer().get(values)
    Matrix(tokens, width, values)

  private def readCsv(path: Path, width: Int): Matrix =
    if Files.size(path) > Limit then invalid(s"${path.getFileName} exceeds 256 MiB")
    val rows = Csv.parse(Files.readString(path, UTF_8).stripPrefix("\uFEFF"))
      .map(_.map(_.trim)).filter(_.exists(_.nonEmpty))
    if rows.isEmpty then invalid(s"${path.getFileName} is empty")
    val transposedHeader = rows.size > 1 && rows.head.forall(_.toDoubleOption.isEmpty)
    val body = if transposedHeader then rows.tail else rows
    val cells = if body.forall(_.headOption.exists(_.toDoubleOption.isEmpty)) then body.map(_.tail) else body
    val cols = cells.head.size
    if cols == 0 || cells.exists(_.size != cols) then invalid(s"${path.getFileName} has uneven rows")
    val transposed = cells.size == width && (cols != width || transposedHeader)
    if !transposed && cols != width then invalid(s"${path.getFileName} does not fit conditioning width $width")
    val tokens = if transposed then cols else cells.size
    if tokens > MaxTokens then invalid(s"${path.getFileName} exceeds $MaxTokens tokens")
    val values = Array.tabulate(tokens * width) { index =>
      val token = index / width
      val feature = index % width
      val cell = if transposed then cells(feature)(token) else cells(token)(feature)
      cell.toFloatOption.filter(_.isFinite).getOrElse(invalid(s"${path.getFileName} contains a non-finite or invalid value"))
    }
    Matrix(tokens,width,values)

  private def readTorch(path: Path, width: Int): (Matrix, Option[Array[Float]]) =
    if Files.size(path) > Limit then invalid(s"${path.getFileName} exceeds 256 MiB")
    val (tensor, pooled) = VulkanClipTorchFile.read(path)
    val shape = if tensor.shape.size == 3 && tensor.shape.head == 1 then tensor.shape.tail else tensor.shape
    val matrix = shape match
      case Vector(tokens, features) if features == width && tokens <= MaxTokens =>
        Matrix(tokens, width, tensor.data)
      case Vector(features, tokens) if features == width && tokens <= MaxTokens =>
        val rows = Array.tabulate(tokens * width)(i => tensor.data(i % width * tokens + i / width))
        Matrix(tokens, width, rows)
      case _ => invalid(s"${path.getFileName} does not fit conditioning width $width")
    matrix -> pooled.map { value =>
      if value.data.length != 1280 then invalid(s"${path.getFileName} has an invalid SDXL pooled tensor")
      value.data
    }

  private enum Piece:
    case File(name: String)
    case Prompt(text: String, clipSkip: Option[Int] = None)

  private def quotedEnd(value: String, start: Int): (Int, Int) =
    val quote = value(start).toString
    val delimiter = if value.startsWith(quote * 3, start) then quote * 3 else quote
    var end = start + delimiter.length
    while end < value.length && !value.startsWith(delimiter, end) do
      end += (if value(end) == '\\' then 2 else 1)
    if end >= value.length then invalid("unterminated quoted input")
    (end + delimiter.length, delimiter.length)

  private def splitArguments(value: String): Vector[String] =
    val parts = Vector.newBuilder[String]
    var start = 0
    var cursor = 0
    var nesting = 0
    while cursor < value.length do
      value(cursor) match
        case '"' | '\'' => cursor = quotedEnd(value, cursor)._1
        case '(' | '[' => nesting += 1; cursor += 1
        case ')' | ']' => nesting -= 1; cursor += 1
        case ',' if nesting == 0 =>
          parts += value.substring(start, cursor).trim
          cursor += 1
          start = cursor
        case _ => cursor += 1
    if nesting != 0 then invalid("unbalanced directive argument")
    parts += value.substring(start).trim
    parts.result()

  private val Argument = """([a-zA-Z_][\w]*)\s*=(.*)""".r
  private def promptDirective(args: String): Vector[Piece] =
    val keys = Vector("prompt", "clip_skip", "no_padding")
    val supplied = splitArguments(args).zipWithIndex.collect {
      case (Argument(name, value), _) if keys.contains(name) => name -> value.trim
      case (Argument(name, _), _) => invalid(s"unknown ?prompt argument $name")
      case (value, index) if value.nonEmpty =>
        if index >= keys.size then invalid("?prompt takes at most three arguments")
        keys(index) -> value
    }.toMap
    def values(raw: String): Vector[String] =
      if (raw.startsWith("[") && raw.endsWith("]")) || (raw.startsWith("(") && raw.endsWith(")")) then
        splitArguments(raw.substring(1, raw.length - 1)).filter(_.nonEmpty)
      else Vector(raw)
    val prompts = values(supplied.getOrElse("prompt", invalid("?prompt requires quoted text"))).map { value =>
      if value.isEmpty || !Set('"', '\'')(value.head) then invalid("?prompt requires quoted text")
      val (end, delimiter) = quotedEnd(value, 0)
      if end != value.length then invalid("?prompt requires a string literal")
      value.substring(delimiter, end - delimiter)
    }
    val skips = values(supplied.getOrElse("clip_skip","1")).map(_.toIntOption.filter(n => n >= 1 && n <= 12)
      .getOrElse(invalid("?prompt clip_skip must be in 1..12")))
    val padding = values(supplied.getOrElse("no_padding","False")).map {
      case "False" | "0" => false
      case "True" => true
      case value => value.toIntOption.map(_ != 0).getOrElse(invalid("?prompt no_padding must be a boolean or integer"))
    }
    if prompts.isEmpty || skips.isEmpty || padding.isEmpty then invalid("?prompt lists cannot be empty")
    prompts.zipWithIndex.map { (text, index) =>
      if padding(index.min(padding.size - 1)) then VulkanProtocol.unsupported("Clip I/O prompt encoding without padding")
      Piece.Prompt(text, Some(skips(index.min(skips.size - 1))))
    }

  private val Directive = """\?([a-zA-Z_]\w*)\(""".r
  private def pieces(prompt: String, directives: Boolean): Vector[Piece] =
    val out = Vector.newBuilder[Piece]
    var i = 0
    while i < prompt.length do
      val ch = prompt(i)
      if ch.isWhitespace || ch == ',' then i += 1
      else if ch == '"' || ch == '\'' then
        val (end, delimiter) = quotedEnd(prompt, i)
        val content = prompt.substring(i + delimiter, end - delimiter)
        out += (if delimiter == 3 then Piece.Prompt(content) else Piece.File(content))
        i = end
      else
        val directive = if ch == '?' && directives then Directive.findPrefixMatchOf(prompt.substring(i)) else None
        directive match
          case Some(call) =>
            val start = i + call.end
            var end = start
            var depth = 1
            while end < prompt.length && depth > 0 do
              prompt(end) match
                case '"' | '\'' => end = quotedEnd(prompt, end)._1
                case '(' => depth += 1; end += 1
                case ')' => depth -= 1; end += 1
                case _ => end += 1
            if depth != 0 then invalid("unbalanced directive")
            val name = call.group(1).toLowerCase(Locale.ROOT)
            if name == "prompt" then out ++= promptDirective(prompt.substring(start, end - 1))
            else if name == "eval" || name.matches("eval_\\d+") then VulkanProtocol.unsupported("Clip I/O eval directives")
            else invalid(s"unknown directive ?$name")
            i = end
          case None =>
            val start = i
            while i < prompt.length && !prompt(i).isWhitespace do i += 1
            out += Piece.File(prompt.substring(start, i))
    out.result()

  def withSession[A](config: Value, width: Int)(body: Session => A): A =
    val scratch = Files.createTempDirectory("sdv-clip-io-")
    try body(new Session(config, width, scratch))
    finally
      Using.resource(Files.list(scratch))(_.iterator().asScala.foreach(Files.deleteIfExists(_)))
      Files.deleteIfExists(scratch)

  final class Session private[VulkanClipIo](config: Value, width: Int, scratch: Path):
    private val exports = Vector.newBuilder[(Path, Option[Path], Path)]
    private val root = Files.createDirectories(directory).toRealPath()
    private var serial = 0
    private def flag(key: String): Boolean = config.obj.get(key).exists(_.bool)
    private def option(key: String, fallback: String = ""): String = config.obj.get(key).fold(fallback)(_.str.trim)
    private def mode(key: String, fallback: String): String = option(key, fallback).toLowerCase(Locale.ROOT) match
      case value @ ("off" | "simple" | "directive") => value
      case _ => invalid(s"unknown mode for $key")
    private def target(name: String): Path =
      val relative = Path.of(name.replace('\\', '/'))
      if relative.isAbsolute || relative.getRoot != null || name.isBlank || relative.iterator().asScala.exists(_.toString == "..") then
        invalid(s"'$name' must be inside the conditioning folder")
      val destination = root.resolve(relative).normalize()
      if !destination.startsWith(root) || destination == root then invalid(s"'$name' must be inside the conditioning folder")
      var ancestor = destination
      while !Files.exists(ancestor, java.nio.file.LinkOption.NOFOLLOW_LINKS) do ancestor = ancestor.getParent
      if !ancestor.toRealPath().startsWith(root) then invalid(s"'$name' leaves the conditioning folder")
      destination

    if flag("output_enabled") && option("output_name").isEmpty && option("negative_output_name").isEmpty then
      invalid("an output name is required")

    private def stage(matrix: Matrix): Path =
      serial += 1
      val path = scratch.resolve(s"input-$serial.npy")
      writeNpy(path, matrix)
      path

    def build(label: String, prompt: String, encoder: String, encoding: String,
        lines: scala.collection.mutable.Builder[String, Vector[String]]): Unit =
      val positive = label == "pos"
      val inputMode = if !flag("input_enabled") then "off"
        else mode(if positive then "positive_mode" else "negative_mode", if positive then "simple" else "off")
      val outputName = option(if positive then "output_name" else "negative_output_name")
      def encode(text: String, clipSkip: Option[Int] = None): String =
        val options = clipSkip.fold(encoding)(skip => encoding.replaceFirst("clip_skip = \\d+$", s"clip_skip = $skip"))
        s"encode_prompt($encoder, ${quoted(text)}, $options)"
      val condition = s"clip_${label}_base"
      lines += s"$condition = ${encode(if inputMode == "off" then prompt else "")}"
      if flag("output_enabled") && outputName.nonEmpty then
        val format = option("format", "csv").toLowerCase(Locale.ROOT)
        if !Set("csv", "pt")(format) then invalid("format must be csv or pt")
        val filename = if outputName.toLowerCase(Locale.ROOT).matches(".*\\.(csv|pt)$") then outputName else s"$outputName.$format"
        val destination = target(filename)
        if !Files.exists(destination) || flag("overwrite") then
          if Seq("no_special_tokens","ignore_emphasis","bypass_normalization","token_strings").exists(flag) then
            VulkanProtocol.unsupported("Clip I/O custom token or normalization export")
          val path = scratch.resolve(s"output-$label.npy")
          val original = if inputMode == "off" then condition else
            val variable = s"clip_${label}_original"
            lines += s"$variable = ${encode(prompt)}"
            variable
          lines += s"tensor::save(${quoted(path.toString)}, $original.crossattn)"
          val pooled = Option.when(width == 2048 && filename.toLowerCase(Locale.ROOT).endsWith(".pt")) {
            val pooledPath = scratch.resolve(s"pooled-$label.npy")
            lines += s"${label}_flat = tensor::reshape($original.vector,[tensor::numel($original.vector)])"
            lines += s"${label}_pool = tensor::crop(${label}_flat, axes = [0], start = [0], size = [1280])"
            lines += s"tensor::save(${quoted(pooledPath.toString)}, tensor::reshape(${label}_pool,[1280,1]))"
            pooledPath
          }
          exports += ((path, pooled, destination))
      if inputMode == "off" then lines += s"latent_$label = $condition"
      else
        val inputs = pieces(prompt, inputMode == "directive")
        if inputs.isEmpty || inputs.size > 64 then invalid("supply between 1 and 64 input pieces")
        var fileTokens = 0
        var pooled: Option[String] = None
        val tensors = inputs.zipWithIndex.map { (piece, index) =>
          val name = s"clip_${label}_$index"
          piece match
            case Piece.Prompt(text, skip) =>
              lines += s"${name}_encoded = ${encode(text, skip)}"
              val encoded = s"${name}_encoded.crossattn"
              lines += s"$name = tensor::reshape($encoded,[$width,tensor::shape($encoded)[1]])"
              if width == 2048 && pooled.isEmpty then
                lines += s"${name}_flat = tensor::reshape(${name}_encoded.vector,[tensor::numel(${name}_encoded.vector)])"
                lines += s"${name}_pool = tensor::crop(${name}_flat, axes = [0], start = [0], size = [1280])"
                pooled = Some(s"${name}_pool")
            case Piece.File(path) =>
              val file = Seq(path, s"$path.csv", s"$path.pt").iterator.map(target).find(Files.isRegularFile(_))
                .getOrElse(invalid(s"conditioning file '$path' was not found"))
              val (matrix, pool) = if file.toString.toLowerCase(Locale.ROOT).endsWith(".pt") then readTorch(file, width)
                else readCsv(file, width) -> None
              fileTokens += matrix.tokens
              if fileTokens > MaxTokens then invalid(s"conditioning exceeds $MaxTokens tokens")
              lines += s"$name = tensor::load(${quoted(stage(matrix).toString)})"
              if width == 2048 && pooled.isEmpty then pool.foreach { values =>
                val path = stage(Matrix(1,1280,values))
                lines += s"${name}_pool = tensor::reshape(tensor::load(${quoted(path.toString)}), [1280])"
                pooled = Some(s"${name}_pool")
              }
          name
        }
        val cross = if tensors.size == 1 then tensors.head else s"tensor::concat(${tensors.mkString("[", ",", "]")}, axis = 1)"
        val vector = if width != 2048 then "" else pooled.fold(s", vector = $condition.vector") { pool =>
          lines += s"${label}_vector_flat = tensor::reshape($condition.vector,[tensor::numel($condition.vector)])"
          lines += s"${label}_vector_replaced = tensor::replace_slice(${label}_vector_flat,$pool,axis = 0,start = 0)"
          s", vector = tensor::reshape(${label}_vector_replaced,tensor::shape($condition.vector))"
        }
        lines += s"latent_$label = {crossattn = $cross$vector}"

    def publish(): Unit =
      for (source, pooled, destination) <- exports.result() do
        val matrix = readNpy(source, width)
        val output = target(root.relativize(destination).toString)
        Files.createDirectories(output.getParent)
        val temporary = Files.createTempFile(output.getParent, ".sdv-clip-", ".tmp")
        try
          if destination.toString.toLowerCase(Locale.ROOT).endsWith(".pt") then
            val pool = pooled.map { path =>
              val data = readNpy(path, 1280)
              VulkanClipTorchFile.Tensor(Vector(data.tokens,data.width), data.values)
            }
            VulkanClipTorchFile.write(temporary, VulkanClipTorchFile.Tensor(Vector(matrix.tokens, matrix.width), matrix.values), pool)
          else Using.resource(Files.newBufferedWriter(temporary, UTF_8)) { writer =>
            val transpose = flag("transpose")
            val rowCount = if transpose then matrix.width else matrix.tokens
            val colCount = if transpose then matrix.tokens else matrix.width
            for row <- 0 until rowCount do
              for col <- 0 until colCount do
                if col > 0 then writer.write(",")
                val index = if transpose then col * matrix.width + row else row * matrix.width + col
                writer.write(java.lang.Float.toString(matrix.values(index)))
              writer.newLine()
          }
          try
            if flag("overwrite") then Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING)
            else Files.move(temporary, output)
          catch case _: FileAlreadyExistsException if !flag("overwrite") => ()
        finally Files.deleteIfExists(temporary)
