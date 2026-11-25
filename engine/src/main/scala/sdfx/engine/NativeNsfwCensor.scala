package sdfx.engine

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import sdfx.core.config.AppHome
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.{EngineError, Protocol}

import java.awt.geom.{Ellipse2D, Rectangle2D, RoundRectangle2D}
import java.awt.image.{BufferedImage, DataBufferByte, DataBufferInt}
import java.awt.{Color, RenderingHints}
import java.io.ByteArrayInputStream
import java.nio.FloatBuffer
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import java.util.{Base64, Collections, Locale, UUID}
import javax.imageio.{IIOImage, ImageIO, ImageWriteParam}
import javax.imageio.stream.MemoryCacheImageOutputStream
import scala.collection.mutable.ArrayBuffer
import scala.util.Using
import scala.util.control.NonFatal

/** Native JVM implementation of NudeNet NSFW Censor (`nsfw_checker` hook and `nsfw_checker` image tool),
  * running ONNX Runtime + Java2D with zero Python dependency.
  */
object NativeNsfwCensor extends sdfx.core.image.RegionCensor:
  private final case class FileStamp(path: Path, size: Long, mtimeNanos: Long)
  private final case class CachedSession(stamp: FileStamp, session: OrtSession, inputName: String, inputWidth: Int, inputHeight: Int)

  private val sessionLock = new Object
  @volatile private var verifiedStamp: Option[FileStamp] = None
  @volatile private var cachedSession: Option[CachedSession] = None

  // ---- model lookup & verification ---------------------------------------------------------------------------

  private lazy val jarInstallRoots: Seq[Path] =
    try
      val jar = Path.of(classOf[NativeNsfwCensorMarker].getProtectionDomain.getCodeSource.getLocation.toURI)
        .toAbsolutePath.normalize()
      val dir = if Files.isDirectory(jar) then jar else Option(jar.getParent).getOrElse(jar)
      Seq(dir, Option(dir.getParent).getOrElse(dir), Option(dir.getParent).flatMap(p => Option(p.getParent)).getOrElse(dir)).distinct
    catch case NonFatal(_) => Nil

  def resolveModelPath(home: AppHome, explicit: Option[Path] = None): Option[Path] =
    val rel = Path.of("models").resolve("nudenet").resolve("best.onnx")
    val candidates = explicit.toSeq ++ Seq(home.root.resolve(rel)) ++ jarInstallRoots.map(_.resolve(rel)) ++ Seq(rel)
    candidates.map(_.toAbsolutePath.normalize()).find(Files.isRegularFile(_))

  def available(home: AppHome): Boolean =
    resolveModelPath(home).exists { path =>
      try
        OrtEnvironment.getEnvironment != null && Files.isRegularFile(path)
      catch case NonFatal(_) => false
    }

  def verifyModel(path: Path): Path =
    val abs = path.toAbsolutePath.normalize()
    if !Files.isRegularFile(abs) then
      throw EngineError(Protocol.ErrorCodes.Unsupported,
        s"NudeNet NSFW Censor: NudeNet model not found: $abs (reinstall models/nudenet/best.onnx)")
    val size = Files.size(abs)
    val mtime = Files.getLastModifiedTime(abs).toInstant.getNano.toLong ^
      (Files.getLastModifiedTime(abs).toInstant.getEpochSecond * 1000000007L)
    val stamp = FileStamp(abs, size, mtime)
    if verifiedStamp.contains(stamp) then return abs
    if size != ModelSize then
      throw EngineError(Protocol.ErrorCodes.Unsupported,
        s"NudeNet NSFW Censor: model $abs has $size bytes, expected $ModelSize; refusing to load it")
    val digest = MessageDigest.getInstance("SHA-256")
    val buf = new Array[Byte](1 << 20)
    Using.resource(Files.newInputStream(abs)) { in =>
      var n = in.read(buf)
      while n >= 0 do
        if n > 0 then digest.update(buf, 0, n)
        n = in.read(buf)
    }
    val hex = digest.digest().map(b => f"${b & 0xff}%02x").mkString
    if hex != ModelSha256 then
      throw EngineError(Protocol.ErrorCodes.Unsupported,
        s"NudeNet NSFW Censor: model $abs SHA-256 $hex does not match pinned $ModelSha256; refusing to load it")
    verifiedStamp = Some(stamp)
    abs

  private def sessionFor(home: AppHome, explicitModelPath: Option[Path] = None): CachedSession =
    val resolved = resolveModelPath(home, explicitModelPath).getOrElse {
      val expected = home.root.resolve("models/nudenet/best.onnx").toAbsolutePath.normalize()
      throw EngineError(Protocol.ErrorCodes.Unsupported,
        s"NudeNet NSFW Censor: NudeNet model not found: $expected (reinstall models/nudenet/best.onnx)")
    }
    val abs = verifyModel(resolved)
    val stamp = verifiedStamp.get
    sessionLock.synchronized {
      cachedSession match
        case Some(c) if c.stamp == stamp => c
        case old =>
          old.foreach { prev =>
            try prev.session.close() catch case NonFatal(_) => ()
          }
          val env = OrtEnvironment.getEnvironment()
          val opts = new OrtSession.SessionOptions()
          try
            val threads = math.max(1, math.min(4, Runtime.getRuntime.availableProcessors()))
            opts.setIntraOpNumThreads(threads)
            val sess = env.createSession(abs.toString, opts)
            val inputInfo = sess.getInputInfo.entrySet().iterator().next()
            val inputName = inputInfo.getKey
            val dims = inputInfo.getValue.getInfo.asInstanceOf[ai.onnxruntime.TensorInfo].getShape
            val w = if dims.length > 2 && dims(2) > 0 then dims(2).toInt else InputSize
            val h = if dims.length > 3 && dims(3) > 0 then dims(3).toInt else InputSize
            val created = CachedSession(stamp, sess, inputName, w, h)
            cachedSession = Some(created)
            created
          finally opts.close()
    }

  def prepareSession(cfg: GenerationConfig, home: AppHome): Unit =
    if cfg.safetyChecker || cfg.nsfwjs then
      throw EngineError(Protocol.ErrorCodes.Unsupported,
        "NSFW Checker: SD safety checker and NSFWJS filter require the Python engine; native scala-torch mode supports NudeNet region censoring.")
    if cfg.nudenetActive(preview = false) || cfg.nudenetActive(preview = true) then
      sessionFor(home)

  override def detectMask(
      img: BufferedImage,
      home: AppHome,
      thresholds: Array[Float],
      expandH: Array[Float],
      expandV: Array[Float],
      nmsThreshold: Double,
      maskShape: String,
      roundRadius: Double,
      modelPath: Option[Path] = None,
      log: Option[String => Unit] = None
  ): Option[BufferedImage] =
    val sess = sessionFor(home, modelPath)
    val nchw = preprocess(img, sess.inputWidth, sess.inputHeight)
    val env = OrtEnvironment.getEnvironment()
    val rawChannels = Using.resource(
      OnnxTensor.createTensor(env, FloatBuffer.wrap(nchw), Array(1L, 3L, sess.inputHeight.toLong, sess.inputWidth.toLong))
    ) { tensor =>
      Using.resource(sess.session.run(Collections.singletonMap(sess.inputName, tensor))) { res =>
        val value = res.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
        value(0)
      }
    }
    val boxes = decodeDetections(
      rawChannels, img.getWidth, img.getHeight, thresholds, expandH, expandV,
      nmsThreshold, sess.inputWidth, sess.inputHeight
    )
    if boxes.isEmpty then None
    else
      log.foreach { emit =>
        boxes.foreach { b =>
          emit(s"${Friendly(b.classIndex)}: score ${b.score}, x1 ${b.x1} y1 ${b.y1}, w ${math.round(b.w)} h ${math.round(b.h)}, x2 ${b.x2} y2 ${b.y2}")
        }
      }
      Some(renderMask(img.getWidth, img.getHeight, boxes, maskShape, roundRadius))

private final class NativeNsfwCensorMarker
