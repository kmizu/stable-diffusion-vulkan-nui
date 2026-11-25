package sdfx.vulkan.tools

import java.nio.file.{Files, Path}
import scala.util.Using
import sdfx.core.protocol.LoadedModel

private[tools] object EmbeddingTables:
  type Streams = Vector[(String, ClipFastTokenizer, Table)]
  private lazy val clipL = ClipFastTokenizer.loadBundled()
  private lazy val clipG = ClipFastTokenizer.loadBundled(padToken = "!")

  /** Bounded row cache for vocabulary scans and individual token lookups on large SDXL checkpoints. */
  final class Table(file: CheckpointIO.File, tensor: CheckpointIO.Tensor):
    require(tensor.shape.size == 2 && CheckpointIO.floatTypes(tensor.dtype), "invalid token embedding table")
    val count = Math.toIntExact(tensor.shape.head)
    val width = Math.toIntExact(tensor.shape.last)
    require(count > 0 && width > 0 && width <= 4096, "invalid token embedding dimensions")
    private val rowBytes = tensor.length / count
    private var start = -1
    private var block = Array.emptyFloatArray
    private def offset(row: Int): Int =
      require(row >= 0 && row < count, s"token ID outside vocabulary: $row")
      val begin = row / 128 * 128
      if start != begin then
        val rows = (count - begin).min(128)
        block = file.floats(tensor.copy(shape = Vector(rows, width), offset = tensor.offset + begin * rowBytes, length = rows * rowBytes))
        require(block.forall(_.isFinite), "nonfinite token embedding")
        start = begin
      (row - start) * width
    def row(id: Int): Array[Float] =
      val i = offset(id)
      java.util.Arrays.copyOfRange(block, i, i + width)
    def at(id: Int, column: Int): Float =
      val index = offset(id) + column
      block(index)

  def withModel[A](model: LoadedModel)(body: Streams => A): A =
    Using.resource(CheckpointIO.open(Path.of(model.path))) { file =>
      val tables = file.tensors.toVector.filter { (key, t) =>
        key.endsWith("token_embedding.weight") && t.shape.size == 2 && t.shape.head == 49408
      }
      def table(width: Int): Table =
        val candidates = tables.filter(_._2.shape.last == width)
        require(candidates.size == 1, s"Expected one CLIP vocabulary table of width $width in ${model.path}")
        new Table(file, candidates.head._2)
      val left = ("clip_l", clipL, table(768))
      val streams = model.kind match
        case "sd15" => Vector(left)
        case "sdxl" => Vector(left, ("clip_g", clipG, table(1280)))
        case other => throw NativeBackendError("unsupported", s"Unsupported CLIP model: $other")
      body(streams)
    }

  def readVectors(path: Path, streams: Streams): Map[String, Vector[Array[Float]]] =
    if !path.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      throw NativeBackendError("unsupported", "Embedding tools require safetensors")
    require(Files.isRegularFile(path) && Files.size(path) <= 64L * 1024 * 1024, "invalid embedding file")
    Using.resource(CheckpointIO.open(path)) { file =>
      streams.map { (key, _, table) =>
        val t = file.tensors.get(key).orElse(Option.when(streams.size == 1 && file.tensors.size == 1)(file.tensors.values.head))
          .getOrElse(throw IllegalArgumentException(s"Missing $key tensor in $path"))
        require(t.shape.nonEmpty && t.shape.size <= 2 && t.shape.last == table.width && t.elements > 0 &&
          t.elements / table.width <= 4096 && CheckpointIO.floatTypes(t.dtype), s"Invalid $key embedding dimensions")
        val xs = file.floats(t)
        require(xs.forall(_.isFinite), "embedding has nonfinite values")
        key -> xs.grouped(table.width).toVector
      }.toMap
    }
