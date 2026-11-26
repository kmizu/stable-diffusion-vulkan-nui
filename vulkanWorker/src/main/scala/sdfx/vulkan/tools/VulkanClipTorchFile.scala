package sdfx.vulkan.tools

import java.io.{ByteArrayOutputStream, DataOutputStream}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.zip.{ZipEntry, ZipFile, ZipOutputStream}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.vulkan.VulkanProtocol

private[vulkan] object VulkanClipTorchFile:
  final case class Tensor(shape: Vector[Int], data: Array[Float])
  private def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"Clip I/O PT: $detail")

  def read(path: Path): (Tensor, Option[Tensor]) = read(path, aesthetic = false)
  def readAesthetic(path: Path): Tensor = read(path, aesthetic = true)._1

  private def read(path: Path, aesthetic: Boolean): (Tensor, Option[Tensor]) =
    Using.resource(new ZipFile(path.toFile)) { zip =>
      val descriptors = zip.entries().asScala.filter(_.getName.endsWith("/data.pkl")).take(2).toList
      if descriptors.size != 1 then invalid("expected one tensor descriptor")
      val prefix = descriptors.head.getName.stripSuffix("data.pkl")
      def bytes(name: String, max: Int): Array[Byte] =
        val entry = Option(zip.getEntry(prefix + name)).getOrElse(invalid(s"missing archive member $name"))
        if entry.getSize < 0 || entry.getSize > max then invalid(s"archive member $name exceeds the size limit")
        val data = Using.resource(zip.getInputStream(entry))(_.readNBytes(max + 1))
        if data.length != entry.getSize then invalid(s"unexpected length for $name")
        data
      val endian = if zip.getEntry(prefix + "byteorder") == null then ByteOrder.LITTLE_ENDIAN
        else new String(bytes("byteorder", 8), UTF_8).trim match
          case "little" => ByteOrder.LITTLE_ENDIAN
          case "big" => ByteOrder.BIG_ENDIAN
          case _ => invalid("invalid byte order")

      def tensor(value: Object): Tensor = value match
        case descriptor: TorchYoloPickle.Tensor =>
          val shape = descriptor.shape().asScala.map(_.intValue).toVector
          val strides = descriptor.stride().asScala.map(_.intValue).toVector
          if shape.isEmpty || shape.size > 3 || shape.size != strides.size || shape.exists(n => n < 1 || n > 4928) ||
              strides.exists(_ < 0) || descriptor.offset() < 0 then invalid("invalid tensor shape or stride")
          val count = shape.iterator.map(_.toLong).product
          if count > 4928L * 2048 then invalid("tensor exceeds the size limit")
          val (size, load) = descriptor.dtype() match
            case "torch.FloatStorage" => (4, (b: ByteBuffer, pos: Int) => b.getFloat(pos))
            case "torch.DoubleStorage" => (8, (b: ByteBuffer, pos: Int) => b.getDouble(pos).toFloat)
            case "torch.HalfStorage" => (2, (b: ByteBuffer, pos: Int) => java.lang.Float.float16ToFloat(b.getShort(pos)))
            case "torch.BFloat16Storage" => (2, (b: ByteBuffer, pos: Int) => java.lang.Float.intBitsToFloat(b.getShort(pos).toInt << 16))
            case _ => invalid("expected floating-point tensor storage")
          val data = ByteBuffer.wrap(bytes("data/" + descriptor.storage(), 128 * 1024 * 1024)).order(endian)
          val end = shape.zip(strides).foldLeft(descriptor.offset().toLong) { case (offset, (length, stride)) =>
            offset + (length - 1L) * stride
          }
          if end >= data.remaining().toLong / size then invalid("tensor points beyond its storage")
          val values = Array.tabulate(count.toInt) { index =>
            var rest = index
            var offset = descriptor.offset()
            for axis <- shape.indices.reverse do
              offset += rest % shape(axis) * strides(axis)
              rest /= shape(axis)
            val value = load(data, offset * size)
            if !value.isFinite then invalid("tensor contains non-finite values")
            value
          }
          Tensor(shape, values)
        case _ => invalid("expected a tensor")

      TorchYoloPickle.tensorPayload(bytes("data.pkl", 1024 * 1024)) match
        case table: java.util.Map[?, ?] if aesthetic =>
          val available = table.values().asScala.iterator.collect { case value: TorchYoloPickle.Tensor => value }.take(2).toVector
          val selected = Option(table.get("embedding")).orElse(Option(table.get("aesthetic_embedding")))
            .orElse(available match
              case Vector(only) => Some(only)
              case _ => None)
            .getOrElse(invalid("archive has no aesthetic embedding tensor"))
          tensor(selected) -> None
        case table: java.util.Map[?, ?] =>
          val cond = Seq("cond", "conditioning", "c").iterator.map(name => table.get(name))
            .find(_ != null).getOrElse(invalid("archive has no conditioning tensor"))
          tensor(cond) -> Option(table.get("pooled")).map(tensor)
        case single => tensor(single) -> None
    }

  def write(path: Path, cond: Tensor, pooled: Option[Tensor], tokens: Vector[String]): Unit =
    val payload = new ByteArrayOutputStream()
    val pickle = new DataOutputStream(payload)
    def ascii(text: String): Unit = pickle.writeBytes(text)
    def text(value: String): Unit =
      val bytes = value.getBytes(UTF_8)
      pickle.writeByte('X')
      pickle.writeInt(Integer.reverseBytes(bytes.length))
      pickle.write(bytes)
    def int(n: Int): Unit =
      pickle.writeByte('J')
      pickle.writeInt(Integer.reverseBytes(n))
    def tuple(values: Seq[Int]): Unit =
      ascii("(")
      values.foreach(int)
      ascii("t")
    def tensor(value: Tensor, key: String): Unit =
      ascii("ctorch._utils\n_rebuild_tensor_v2\n((")
      text("storage")
      ascii("ctorch\nFloatStorage\n")
      text(key)
      text("cpu")
      int(value.data.length)
      ascii("tQ")
      int(0)
      tuple(value.shape)
      tuple(value.shape.indices.map(i => value.shape.drop(i + 1).product))
      pickle.writeByte(0x89)
      ascii("ccollections\nOrderedDict\n)RtR")
    pickle.writeByte(0x80)
    pickle.writeByte(2)
    pooled match
      case None => tensor(cond, "0")
      case Some(value) =>
        ascii("}(")
        text("cond")
        tensor(cond, "0")
        text("pooled")
        tensor(value, "1")
        text("tokens")
        ascii("]")
        if tokens.nonEmpty then
          ascii("(")
          tokens.foreach(text)
          ascii("e")
        ascii("u")
    ascii(".")

    Using.resource(new ZipOutputStream(Files.newOutputStream(path))) { zip =>
      def entry(name: String, bytes: Array[Byte]): Unit =
        zip.putNextEntry(new ZipEntry(s"archive/$name"))
        zip.write(bytes)
        zip.closeEntry()
      def storage(tensor: Tensor, key: String): Unit =
        val buffer = ByteBuffer.allocate(tensor.data.length * 4).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(tensor.data)
        entry(s"data/$key", buffer.array())
      entry("data.pkl", payload.toByteArray)
      entry("byteorder", "little".getBytes(UTF_8))
      storage(cond, "0")
      pooled.foreach(storage(_, "1"))
      entry("version", "3\n".getBytes(UTF_8))
    }
