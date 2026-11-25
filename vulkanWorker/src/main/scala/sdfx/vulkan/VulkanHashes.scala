package sdfx.vulkan

import java.nio.file.{Files, Path}
import java.nio.file.attribute.BasicFileAttributes
import scala.collection.mutable
import scala.util.control.NonFatal
import sdfx.core.imagebrowser.BrowserFiles

private[vulkan] object VulkanHashes:
  final case class Stamp(size: Long, modified: java.nio.file.attribute.FileTime, key: Option[AnyRef])
  private def stamp(path: Path): Stamp =
    val attributes = Files.readAttributes(path, classOf[BasicFileAttributes])
    Stamp(attributes.size, attributes.lastModifiedTime, Option(attributes.fileKey))

  final class File(val path: Path):
    private val initial = stamp(path)
    lazy val digest: String =
      require(stamp(path) == initial, s"Resource changed since loading: ${path.getFileName}")
      val hash = BrowserFiles.sha256(path).take(10)
      require(stamp(path) == initial, s"Resource changed during hashing: ${path.getFileName}")
      hash

  final class Run:
    private val values = mutable.Map.empty[String,String]
    private val ambiguous = mutable.Set.empty[String]
    private val messages = mutable.LinkedHashSet.empty[String]

    def record(name: String, file: File): Unit =
      if !ambiguous(name) then
        try
          val hash = file.digest
          if values.get(name).exists(_ != hash) then
            values.remove(name)
            ambiguous += name
            messages += s"Ambiguous resource name $name"
          else values(name) = hash
        catch case NonFatal(e) => messages += s"Resource hash unavailable for $name: ${e.getMessage}"

    def hashes: Map[String,String] = values.toMap
    def warnings: Seq[String] = messages.toVector
