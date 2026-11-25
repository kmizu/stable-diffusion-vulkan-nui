package sdfx.vulkan.tools

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{Files, Path, StandardCopyOption, StandardOpenOption}
import scala.util.Using
import NativeJson.*

/** Safetensors I/O for checkpoint tools. Only the tensor being transformed is resident. */
private[tools] object CheckpointIO:
  val floatTypes = Set("F16", "BF16", "F32", "F64")
  val precision = Map("fp16" -> "F16", "bf16" -> "BF16", "fp32" -> "F32")
  private val widths = Map("F16" -> 2, "BF16" -> 2, "F32" -> 4, "F64" -> 8, "I64" -> 8)
  private def little(b: ByteBuffer): ByteBuffer = b.order(ByteOrder.LITTLE_ENDIAN)
  private def buffer(n: Int): ByteBuffer = little(ByteBuffer.allocate(n))
  final case class Tensor(dtype: String, shape: Vector[Long], offset: Long, length: Long):
    def elements: Int = Math.toIntExact(shape.foldLeft(1L)(Math.multiplyExact))
  final case class File(path: Path, channel: FileChannel, tensors: Map[String, Tensor]) extends AutoCloseable:
    def close(): Unit = channel.close()
    def raw(t: Tensor): Array[Byte] =
      val b = buffer(Math.toIntExact(t.length))
      channel.position(t.offset)
      while b.hasRemaining do
        if channel.read(b) < 0 then throw java.io.EOFException(s"Truncated tensor in $path")
      b.array()
    def floats(t: Tensor): Array[Float] =
      val b = little(ByteBuffer.wrap(raw(t)))
      t.dtype match
        case "F16" => Array.fill(t.elements)(java.lang.Float.float16ToFloat(b.getShort()))
        case "BF16" => Array.fill(t.elements)(java.lang.Float.intBitsToFloat(b.getShort().toInt << 16))
        case "F32" => Array.fill(t.elements)(b.getFloat())
        case "F64" => Array.fill(t.elements)(b.getDouble().toFloat)
        case other => throw IllegalArgumentException(s"Cannot merge $other with floating-point weights")

  def open(path: Path): File =
    val channel = FileChannel.open(path)
    try
      val (header, _, _) = NativeMetadataUtils.parseHeader(Channels.newInputStream(channel), channel.size())
      val start = channel.position()
      val tensors = header.iterator.collect { case (key, Obj(d)) if key != "__metadata__" =>
        val shape = d("shape").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toLongExact)
        val offsets = d("data_offsets").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toLongExact)
        key -> Tensor(d("dtype").asInstanceOf[Str].value, shape, Math.addExact(start, offsets.head), offsets.last-offsets.head)
      }.toMap
      File(path, channel, tensors)
    catch case e: Throwable =>
      channel.close()
      throw e

  def encode(xs: Array[Float], dtype: String): Array[Byte] =
    val b = buffer(Math.multiplyExact(xs.length, widths(dtype)))
    dtype match
      case "F32" => b.asFloatBuffer().put(xs)
      case "F16" => xs.foreach(x => b.putShort(java.lang.Float.floatToFloat16(x)))
      case "BF16" => xs.foreach { x =>
        val bits = java.lang.Float.floatToIntBits(x)
        b.putShort(((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16).toShort)
      }
      case other => throw IllegalArgumentException(s"Unsupported output type $other")
    b.array()

  final case class Plan(dtype: String, shape: Vector[Long], bytes: Long, read: () => Array[Byte])
  def copy(file: File, t: Tensor): Plan = Plan(t.dtype, t.shape, t.length, () => file.raw(t))
  def cast(file: File, t: Tensor, dtype: String): Plan =
    if dtype == t.dtype then copy(file, t)
    else converted(t, dtype)(file.floats(t))
  def converted(t: Tensor, dtype: String)(values: => Array[Float]): Plan =
    Plan(dtype, t.shape, t.elements.toLong * widths(dtype), () => encode(values, dtype))
  def positions(t: Tensor): Plan =
    require(t.shape.nonEmpty && t.elements == t.shape.last, "invalid position_ids shape")
    Plan("I64", t.shape, t.elements.toLong * 8, () =>
      val b = buffer(Math.multiplyExact(t.elements, 8))
      for i <- 0 until t.elements do b.putLong(i.toLong)
      b.array())

  def write(output: Path, plans: Vector[(String, Plan)], meta: Map[String,String], overwrite: Boolean,
      ctx: NativeWorkContext, stage: String): Unit =
    var offset = 0L
    val header = ujson.Obj("__metadata__" -> ujson.Obj.from(meta.map((k,v) => k -> ujson.Str(v))))
    plans.foreach { (key, p) =>
      val end = Math.addExact(offset, p.bytes)
      require(end <= (1L << 53), "file size exceeds JSON integer precision")
      header(key) = ujson.Obj("dtype" -> p.dtype, "shape" -> ujson.Arr.from(p.shape.map(x => ujson.Num(x.toDouble))),
        "data_offsets" -> ujson.Arr(offset.toDouble, end.toDouble))
      offset = end
    }
    val json = ujson.write(header).getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val padding = (8 - json.length % 8) % 8
    val temp = Files.createTempFile(output.getParent, ".sdv-checkpoint-", ".tmp")
    try
      Using.resource(Files.newOutputStream(temp)) { out =>
        out.write(buffer(8).putLong(json.length.toLong + padding).array())
        out.write(json)
        out.write(Array.fill[Byte](padding)(32))
        plans.zipWithIndex.foreach { case ((key, p), index) =>
          ctx.checkCancelled()
          if index % 32 == 0 then ctx.event("progress", Map("step" -> Number(index), "total" -> Number(plans.size), "pass" -> Str(stage)))
          val bytes = p.read()
          require(bytes.length == p.bytes, s"Unexpected size for $key")
          out.write(bytes)
        }
      }
      ctx.beginPublication()
      if overwrite then Files.move(temp, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      else Files.createLink(output, temp)
    finally Files.deleteIfExists(temp)
