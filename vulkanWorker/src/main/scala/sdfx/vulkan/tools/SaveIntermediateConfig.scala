// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.UUID
import javax.imageio.{IIOImage, ImageIO, ImageTypeSpecifier}
import javax.imageio.metadata.IIOMetadataNode

/** Intermediate sampling frame recorder (`save_intermediate`) and Steps Animation (`steps_animation`),
  * plus DDIM inversion (`img2imgalt`) and CD Tuner (`cd_tuner`) helpers.
  */
final case class SaveIntermediateConfig(
    everyN: Int = 5,
    startAt: Int = 0,
    stopAt: Int = 0,
    imageType: String = "Denoised",
    mode: String = "Standard",
    finalSave: Boolean = false,
    format: String = "mp4",
    codec: String = "h264",
    fps: Double = 2.0,
    addFirstFrames: Int = 0,
    addLastFrames: Int = 0,
    smooth: Boolean = false,
    seconds: Double = 0.0,
    lores: Boolean = true,
    hires: Boolean = true,
    ffmpegBat: Boolean = false,
    batOnly: Boolean = false,
    outputDirOverride: Option[String] = None
):
  def isSelected(step: Int): Boolean =
    step > 0 && step % everyN == 0 && step >= startAt && (stopAt == 0 || step <= stopAt)

object SaveIntermediateConfig:
  val ValidImageTypes: Set[String] = Set("Denoised", "Noisy", "Preview")
  val ValidModes: Set[String] = Set("Standard", "Suffix", "Video")
  val ValidFormats: Set[String] = Set("gif", "webm", "mp4")
  val ValidCodecs: Set[String] = Set("h264", "h265/hevc", "av1")

  def validate(raw: SaveIntermediateConfig): SaveIntermediateConfig =
    require(raw.everyN >= 1, "save intermediate: every_n must be a positive integer")
    require(raw.startAt >= 0, "save intermediate: start_at must be a nonnegative integer")
    require(raw.stopAt >= 0, "save intermediate: stop_at must be a nonnegative integer")
    require(raw.addFirstFrames >= 0, "save intermediate: add_first_frames must be a nonnegative integer")
    require(raw.addLastFrames >= 0, "save intermediate: add_last_frames must be a nonnegative integer")
    require(raw.fps.isFinite && raw.fps >= 0.01, "save intermediate: invalid fps")
    require(raw.seconds.isFinite && raw.seconds >= 0.0, "save intermediate: invalid seconds")
    require(ValidImageTypes.contains(raw.imageType), "save intermediate: invalid image_type")
    require(ValidModes.contains(raw.mode), "save intermediate: invalid mode")
    require(ValidFormats.contains(raw.format), "save intermediate: invalid format")
    require(ValidCodecs.contains(raw.codec), "save intermediate: invalid codec")
    require(!raw.smooth || raw.fps > 1.0, "save intermediate: smoothing requires output fps greater than 1")
    require(raw.mode != "Video" || raw.lores || raw.hires, "save intermediate: select base or hires frames for video")
    raw.copy(
      startAt = (raw.startAt / raw.everyN) * raw.everyN,
      stopAt = (raw.stopAt / raw.everyN) * raw.everyN
    )

final case class StepsAnimationConfig(
    duration: Double = 5.0,
    fps: Int = 30,
    codec: String = "h264",
    interpolation: String = "blend",
    skipInitial: Int = 0,
    debug: Boolean = false,
    runIncomplete: Boolean = false,
    deleteIntermediate: Boolean = false,
    createOutput: Boolean = true,
    framesDir: String = "steps-animation/frames",
    outputDir: String = "steps-animation/videos"
)

object StepsAnimationConfig:
  val ValidCodecs: Set[String] = Set("h264", "h265", "vp9", "av1", "prores")
  val ValidInterpolations: Set[String] = Set("none", "blend", "mci")

  def validateRelativePath(key: String, raw: String): String =
    require(raw.nonEmpty && !raw.exists(_.toInt < 32), s"steps animation: invalid $key")
    val normalized = raw.replace('\\', '/')
    require(!normalized.startsWith("/") && !normalized.contains(":"), s"steps animation: invalid $key")
    val parts = normalized.split("/").filter(_.nonEmpty).toVector
    require(parts.nonEmpty && !parts.contains("..") && parts != Vector("."), s"steps animation: invalid $key")
    parts.mkString("/")

  def validate(raw: StepsAnimationConfig): StepsAnimationConfig =
    require(raw.duration.isFinite && raw.duration >= 0.01 && raw.duration <= 3600.0, "steps animation: invalid duration")
    require(raw.fps >= 1 && raw.fps <= 240, "steps animation: invalid fps")
    require(raw.skipInitial >= 0 && raw.skipInitial <= 10000, "steps animation: invalid skip_initial")
    require(ValidCodecs.contains(raw.codec), "steps animation: invalid codec")
    require(ValidInterpolations.contains(raw.interpolation), "steps animation: invalid interpolation")
    val cleanFrames = validateRelativePath("frames_dir", raw.framesDir)
    val cleanOutput = validateRelativePath("output_dir", raw.outputDir)
    raw.copy(framesDir = cleanFrames, outputDir = cleanOutput)

final case class IntermediateFrameRecord(
    path: String,
    seed: Long,
    passName: String,
    step: Option[Int],
    passStep: Option[Int],
    iteration: Int,
    batchIndex: Int,
    width: Int,
    height: Int,
    imageType: String
)

final case class IntermediateArtifact(
    kind: String,
    path: String,
    seed: Option[Long] = None,
    passName: Option[String] = None,
    step: Option[Int] = None,
    passStep: Option[Int] = None,
    iteration: Option[Int] = None,
    batchIndex: Option[Int] = None,
    width: Option[Int] = None,
    height: Option[Int] = None,
    imageType: Option[String] = None,
    frames: Option[Int] = None,
    format: Option[String] = None
)

object IntermediatePngWriter:
  def writePngWithMetadata(image: BufferedImage, path: Path, texts: Vector[(String, String)]): Unit =
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    val out = ImageIO.createImageOutputStream(path.toFile)
    try
      writer.setOutput(out)
      val metadata = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), writer.getDefaultWriteParam)
      if texts.nonEmpty then
        val root = new IIOMetadataNode("javax_imageio_png_1.0")
        val entries = new IIOMetadataNode("iTXt")
        texts.foreach { (key, value) =>
          val node = new IIOMetadataNode("iTXtEntry")
          node.setAttribute("keyword", key)
          node.setAttribute("compressionFlag", "FALSE")
          node.setAttribute("compressionMethod", "0")
          node.setAttribute("languageTag", "")
          node.setAttribute("translatedKeyword", "")
          node.setAttribute("text", value)
          entries.appendChild(node)
        }
        root.appendChild(entries)
        metadata.mergeTree("javax_imageio_png_1.0", root)
      writer.write(null, new IIOImage(image, null, metadata), writer.getDefaultWriteParam)
    finally
      writer.dispose()
      out.close()
