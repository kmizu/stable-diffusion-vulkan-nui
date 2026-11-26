package sdfx.vulkan.tools

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.{US_ASCII, UTF_8}
import java.nio.file.{Files, Path}
import java.util.zip.ZipFile
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.vulkan.VulkanProtocol
import TorchYoloPickle.Tensor

private[vulkan] object VulkanFooocusFiles:
  private val prefix = "diffusion_model."
  private val headShape = Vector(320L, 5L, 3L, 3L)
  private def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"Fooocus Inpaint: $detail")

  private final class Archive(zip: ZipFile):
    private val descriptors = zip.entries().asScala.filter(_.getName.endsWith("/data.pkl")).take(2).toList
    if descriptors.size != 1 then invalid("expected one PyTorch tensor descriptor")
    private val root = descriptors.head.getName.stripSuffix("data.pkl")

    private def bytes(name: String, limit: Int): Array[Byte] =
      val entry = Option(zip.getEntry(root + name)).getOrElse(invalid(s"missing archive entry $name"))
      if entry.getSize < 0 || entry.getSize > limit then invalid(s"archive entry $name exceeds the size limit")
      val result = Using.resource(zip.getInputStream(entry))(_.readNBytes(entry.getSize.toInt + 1))
      if result.length != entry.getSize then invalid(s"unexpected length for archive entry $name")
      result

    val order: ByteOrder = if zip.getEntry(root + "byteorder") == null then ByteOrder.LITTLE_ENDIAN
      else new String(bytes("byteorder", 8), UTF_8).trim match
        case "little" => ByteOrder.LITTLE_ENDIAN
        case "big" => ByteOrder.BIG_ENDIAN
        case _ => invalid("invalid archive byte order")

    def table = TorchYoloPickle.map(TorchYoloPickle.tensorPayload(bytes("data.pkl", 4 * 1024 * 1024))).asScala
    def storage(t: Tensor): Array[Byte] = bytes("data/" + t.storage(), 128 * 1024 * 1024)

  private def shape(t: Tensor): Vector[Long] = t.shape().asScala.map(_.longValue()).toVector

  private final class Layout(t: Tensor, capacity: Int):
    private val dims = t.shape().asScala.map(_.intValue()).toArray
    private val strides = t.stride().asScala.map(_.intValue()).toArray
    if dims.length > 4 || dims.exists(x => x <= 0 || x > 65536) || strides.length != dims.length ||
        strides.exists(_ < 0) || t.offset() < 0 then invalid("invalid tensor shape, stride or offset")
    private var size = 1L
    private var contiguous = true
    private var last = t.offset().toLong
    for axis <- dims.indices.reverse do
      if dims(axis) > 1 && size != strides(axis) then contiguous = false
      size *= dims(axis)
      if size > 32L * 1024 * 1024 then invalid("tensor exceeds the size limit")
      last += (dims(axis) - 1L) * strides(axis)
    if last >= capacity then invalid("tensor points beyond its storage")
    val count: Int = size.toInt

    def index(position: Int): Int =
      if contiguous then t.offset() + position
      else
        var offset = t.offset()
        var rest = position
        var axis = dims.length - 1
        while axis >= 0 do
          offset += (rest % dims(axis)) * strides(axis)
          rest /= dims(axis)
          axis -= 1
        offset

  private def floats(t: Tensor, bytes: Array[Byte], order: ByteOrder): Array[Float] =
    val (width, read) = t.dtype() match
      case "torch.FloatStorage" => (4, (b: ByteBuffer, i: Int) => b.getFloat(i))
      case "torch.HalfStorage" => (2, (b: ByteBuffer, i: Int) => java.lang.Float.float16ToFloat(b.getShort(i)))
      case "torch.BFloat16Storage" => (2, (b: ByteBuffer, i: Int) => java.lang.Float.intBitsToFloat(b.getShort(i).toInt << 16))
      case _ => invalid("expected floating-point tensor storage")
    val layout = new Layout(t, bytes.length / width)
    val data = ByteBuffer.wrap(bytes).order(order)
    Array.tabulate(layout.count)(i => read(data, layout.index(i) * width))

  private[vulkan] def dequantize(t: Tensor, bytes: Array[Byte], low: Tensor, lowBytes: Array[Byte],
      high: Tensor, highBytes: Array[Byte], order: ByteOrder): Array[Float] =
    val dims = shape(t)
    val limits = shape(low)
    if limits != shape(high) || limits.size > dims.size then invalid("incompatible quantization bounds")
    val padded = Vector.fill(dims.size - limits.size)(1L) ++ limits
    if padded.zip(dims).exists((bound, dim) => bound != 1 && bound != dim) then
      invalid("quantization bounds cannot broadcast to the weight")
    val lo = floats(low, lowBytes, order)
    val hi = floats(high, highBytes, order)
    for i <- lo.indices do
      if !lo(i).isFinite || !hi(i).isFinite || hi(i) < lo(i) || !(hi(i) - lo(i)).isFinite then
        invalid("invalid quantization bounds")
      hi(i) -= lo(i)

    val expanded = padded.indices.filter(axis => padded(axis) != dims(axis))
    val boundIndex: Int => Int = if lo.length == 1 then _ => 0
      else if expanded.size == 1 then
        val axis = expanded.head
        val inner = dims.drop(axis + 1).product.toInt
        val block = inner * dims(axis).toInt
        i => (i / block) * inner + i % inner
      else if expanded.isEmpty then identity
      else i =>
        var rest = i
        var result = 0
        var stride = 1
        var axis = dims.size - 1
        while axis >= 0 do
          if padded(axis) != 1 then result += (rest % dims(axis).toInt) * stride
          rest /= dims(axis).toInt
          stride *= padded(axis).toInt
          axis -= 1
        result
    val (values, read): (Int, Int => Float) = if t.dtype() == "torch.ByteStorage" then
      val layout = new Layout(t, bytes.length)
      (layout.count, (i: Int) => (bytes(layout.index(i)) & 255).toFloat)
    else
      val decoded = floats(t, bytes, order)
      (decoded.length, (i: Int) => decoded(i))
    Array.tabulate(values) { i =>
      val bound = boundIndex(i)
      read(i) / 255f * hi(bound) + lo(bound)
    }

  private case class Delta(shape: Vector[Long], read: () => Array[Float])

  private def unetName(key: String): Option[String] =
    Option(key.stripPrefix("model.")).filter(_.startsWith(prefix))

  private def withDeltas[A](patch: Path)(body: Map[String, Delta] => A): A =
    if patch.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      Using.resource(CheckpointIO.open(patch)) { file =>
        body(file.tensors.flatMap { (key, t) =>
          unetName(key).map(_ -> Delta(t.shape, () => file.floats(t)))
        })
      }
    else Using.resource(new ZipFile(patch.toFile)) { zip =>
      val archive = new Archive(zip)
      body(archive.table.iterator.flatMap { (key, value) =>
        unetName(key.toString).map { name =>
          val delta = value match
            case t: Tensor => Delta(shape(t), () => floats(t, archive.storage(t), archive.order))
            case tuple: java.util.List[?] if tuple.size() == 3 =>
              val fields: (Any, Any, Any) = (tuple.get(0), tuple.get(1), tuple.get(2))
              fields match
                case (t: Tensor, lo: Tensor, hi: Tensor) =>
                  Delta(shape(t), () => dequantize(t, archive.storage(t), lo, archive.storage(lo), hi, archive.storage(hi), archive.order))
                case _ => invalid(s"invalid quantized delta for $name")
            case _ => invalid(s"invalid weight delta for $name")
          name -> delta
        }
      }.toMap)
    }

  def mergeUnet(checkpoint: Path, patch: Path, output: Path, cancelled: () => Boolean): Unit =
    Using.resource(CheckpointIO.open(checkpoint)) { base =>
      withDeltas(patch) { deltas =>
        val weights = base.tensors.toVector.sortBy(_._1).flatMap { (key, t) =>
          unetName(key).map { name =>
            val delta = deltas.get(name).filter(_.shape == t.shape)
            ("model." + name, t, delta)
          }
        }
        if !weights.exists(_._3.nonEmpty) then invalid("patch has no weights matching this SDXL U-Net")
        val plans = weights.map { (key, t, delta) =>
          val plan = delta.fold(CheckpointIO.copy(base, t)) { d =>
            if !Set("F16", "BF16", "F32")(t.dtype) then invalid(s"unsupported checkpoint weight type for $key")
            CheckpointIO.converted(t, t.dtype) {
              val weights = base.floats(t)
              val changes = d.read()
              val rounded = ByteBuffer.wrap(CheckpointIO.encode(changes, t.dtype)).order(ByteOrder.LITTLE_ENDIAN)
              for i <- weights.indices do
                val change = t.dtype match
                  case "F16" => java.lang.Float.float16ToFloat(rounded.getShort())
                  case "BF16" => java.lang.Float.intBitsToFloat(rounded.getShort().toInt << 16)
                  case _ => rounded.getFloat()
                weights(i) += change
                if !weights(i).isFinite then invalid(s"non-finite patched weight for $key")
              weights
            }
          }
          key -> plan
        }
        val context = new NativeWorkContext(cancelled, () => (), (_, _) => ())
        CheckpointIO.write(output, plans, Map.empty, overwrite = true, context, "fooocus_inpaint")
      }
    }

  def writeHead(head: Path, output: Path): Unit =
    val values = Using.resource(new ZipFile(head.toFile)) { zip =>
      val archive = new Archive(zip)
      archive.table.get("head") match
        case Some(t: Tensor) if shape(t) == headShape => floats(t, archive.storage(t), archive.order)
        case _ => invalid("head must have shape [320, 5, 3, 3]")
    }
    if values.exists(!_.isFinite) then invalid("non-finite head weight")
    val dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (320, 5, 3, 3), }"
    val padding = (16 - (10 + dict.length + 1) % 16) % 16
    val header = (dict + " " * padding + "\n").getBytes(US_ASCII)
    val preamble = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN)
      .put(0x93.toByte).put("NUMPY".getBytes(US_ASCII)).put(1.toByte).put(0.toByte)
      .putShort(header.length.toShort).array()
    Using.resource(Files.newOutputStream(output)) { stream =>
      stream.write(preamble)
      stream.write(header)
      stream.write(CheckpointIO.encode(values, "F32"))
    }
