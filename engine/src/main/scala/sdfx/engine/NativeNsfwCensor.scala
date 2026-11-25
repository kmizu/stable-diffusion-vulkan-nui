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
object NativeNsfwCensor:
  val Labels: Seq[String] = Seq(
    "Female_genitalia_covered", "Face_female", "Buttocks_exposed", "Female_breast_exposed",
    "Female_genitalia_exposed", "Male_breast_exposed", "Anus_exposed", "Feet_exposed", "Belly_covered",
    "Feet_covered", "Armpits_covered", "Armpits_exposed", "Face_male", "Belly_exposed",
    "Male_genitalia_exposed", "Anus_covered", "Female_breast_covered", "Buttocks_covered"
  )
  val Friendly: Seq[String] = Labels.map(_.replace('_', ' '))
  val DefaultSelected: Set[String] = Set(
    "Anus_exposed", "Female_breast_exposed", "Female_genitalia_exposed", "Male_genitalia_exposed"
  )
  val DefaultThresholds: Array[Float] =
    Labels.map(l => if DefaultSelected.contains(l) then 0.25f else 1.0f).toArray
  val DefaultExpand: Array[Float] = Array.fill(Labels.size)(1.0f)

  val Filters: Seq[String] = Seq("Variable blur", "Gaussian Blur", "Pixelate", "Fill color", "No censor", "Disable")
  val MaskShapes: Seq[String] = Seq("Ellipse", "Rectangle", "Rounded rectangle", "Entire image")
  val EntireImageAlways: String = "Entire image (always)"
  val ToolCheckers: Seq[String] = Seq("nudenet", "nsfwjs", "safety_checker")

  val ModelSha256: String = "9832f15515bdb06bcb5a77beb60bc8ea54439bd7ecbaac46dac3b760b3dd13cc"
  val ModelSize: Long = 12125773L
  val InputSize: Int = 320
  val AuxiliarySuffixes: (String, String) = ("-before-censor", "-censor-mask")

  final case class DetectionBox(
      x1: Double,
      y1: Double,
      x2: Double,
      y2: Double,
      w: Double,
      h: Double,
      classIndex: Int,
      score: Float
  )

  final case class GenerationConfig(
      genFilterType: String = "Gaussian Blur",
      livePreviewFilterType: String = "Gaussian Blur",
      saveBeforeCensor: Boolean = true,
      saveMask: Boolean = false,
      maskShape: String = "Ellipse",
      rectangleRoundRadius: Double = 0.5,
      blurRadius: Double = 10.0,
      maskBlendRadius: Double = 0.0,
      maskBlendRadiusVariableBlur: Double = 10.0,
      blurStrengthCurve: Double = 3.0,
      pixelationFactor: Double = 5.0,
      fillColor: String = "#000000",
      nmsThreshold: Double = 0.5,
      verboseDetection: Boolean = false,
      thresholds: Array[Float] = DefaultThresholds.clone(),
      expandHorizontal: Array[Float] = DefaultExpand.clone(),
      expandVertical: Array[Float] = DefaultExpand.clone(),
      nudenet: Boolean = true,
      safetyChecker: Boolean = false,
      safetyCheckerLivePreview: Boolean = false,
      nsfwjs: Boolean = false,
      nsfwjsSafetyLevel: String = "safe"
  ):
    def nudenetActive(preview: Boolean = false): Boolean =
      val f = if preview then livePreviewFilterType else genFilterType
      nudenet && f != "Disable"

  final case class CensorGenerationResult(
      image: BufferedImage,
      beforeCensor: Option[BufferedImage],
      mask: Option[BufferedImage],
      detected: Boolean
  )

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

  // ---- config parsing & validation ---------------------------------------------------------------------------

  private def invalid(msg: String): Nothing =
    throw EngineError(Protocol.ErrorCodes.InvalidParams, s"NudeNet NSFW Censor: $msg")

  private def coerceBool(v: Option[ujson.Value], default: Boolean): Boolean = v match
    case None | Some(ujson.Null) => default
    case Some(ujson.Bool(b))     => b
    case Some(ujson.Num(n))      => n != 0.0
    case Some(ujson.Str(s)) =>
      s.trim.toLowerCase(Locale.ROOT) match
        case "true" | "1" | "yes" | "on"  => true
        case "false" | "0" | "no" | "off" => false
        case _                            => default
    case _ => default

  private def coerceNum(v: ujson.Value, key: String, low: Option[Double] = None, high: Option[Double] = None): Double =
    val n = v match
      case ujson.Num(d) => d
      case ujson.Str(s) =>
        try s.trim.toDouble
        catch case _: NumberFormatException => invalid(s"$key must be a number")
      case _ => invalid(s"$key must be a number")
    if !java.lang.Double.isFinite(n) || low.exists(n < _) || high.exists(n > _) then
      val lo = low.fold("-inf")(_.toString)
      val hi = high.fold("inf")(_.toString)
      invalid(s"$key must be a finite number in $lo..$hi (got $n)")
    n

  private def coerceChoice(v: Option[ujson.Value], default: String, key: String, options: Seq[String]): String =
    val chosen = v match
      case None | Some(ujson.Null) => default
      case Some(ujson.Str(s))      => s
      case Some(other)             => invalid(s"$key must be a string (got $other)")
    if !options.contains(chosen) then invalid(s"$key must be one of ${options.mkString(", ")} (got '$chosen')")
    chosen

  private def coerceVector(v: Option[ujson.Value], key: String, default: Array[Float], low: Double = Double.NegativeInfinity): Array[Float] =
    v match
      case None | Some(ujson.Null) => default.clone()
      case Some(ujson.Arr(arr)) =>
        if arr.size != Labels.size then invalid(s"$key must list ${Labels.size} numbers in NudeNet label order")
        arr.zipWithIndex.map { (item, _) =>
          val d = coerceNum(item, key, Some(low), None)
          d.toFloat
        }.toArray
      case Some(ujson.Str(s)) =>
        val parts = s.replace(';', ',').split(',').map(_.trim).filter(_.nonEmpty)
        if parts.length != Labels.size then invalid(s"$key must list ${Labels.size} numbers in NudeNet label order")
        parts.map(p => coerceNum(ujson.Str(p), key, Some(low), None).toFloat)
      case _ => invalid(s"$key must list ${Labels.size} numbers in NudeNet label order")

  private[engine] def parseColor(value: Option[ujson.Value], default: String = "#000000"): (String, Color) =
    val raw = value match
      case None | Some(ujson.Null) => default
      case Some(ujson.Str(s))      => s.trim
      case Some(other)             => other.toString.trim
    val normalized = if raw.nonEmpty then raw else default
    val c =
      try
        if normalized.matches("#[0-9A-Fa-f]{6}") then Color.decode(normalized)
        else if normalized.matches("#[0-9A-Fa-f]{3}") then
          val r = Integer.parseInt(normalized.substring(1, 2) * 2, 16)
          val g = Integer.parseInt(normalized.substring(2, 3) * 2, 16)
          val b = Integer.parseInt(normalized.substring(3, 4) * 2, 16)
          new Color(r, g, b)
        else invalid(s"fill_color must be a colour such as #000000 (got '$raw')")
      catch case _: NumberFormatException => invalid(s"fill_color must be a colour such as #000000 (got '$raw')")
    (normalized, c)

  def parseGenerationConfig(raw: ujson.Obj): GenerationConfig =
    val m = raw.value
    def num(key: String, default: Double, low: Option[Double] = None, high: Option[Double] = None): Double =
      m.get(key).filter(_ != ujson.Null).fold(default)(coerceNum(_, key, low, high))
    val (fillHex, _) = parseColor(m.get("fill_color"), "#000000")
    GenerationConfig(
      genFilterType = coerceChoice(m.get("gen_filter_type"), "Gaussian Blur", "gen_filter_type", Filters),
      livePreviewFilterType = coerceChoice(m.get("live_preview_filter_type"), "Gaussian Blur", "live_preview_filter_type", Filters),
      saveBeforeCensor = coerceBool(m.get("save_before_censor"), default = true),
      saveMask = coerceBool(m.get("save_mask"), default = false),
      maskShape = coerceChoice(m.get("mask_shape"), "Ellipse", "mask_shape", MaskShapes),
      rectangleRoundRadius = num("rectangle_round_radius", 0.5),
      blurRadius = num("blur_radius", 10.0, Some(0.0)),
      maskBlendRadius = num("mask_blend_radius", 0.0, Some(0.0)),
      maskBlendRadiusVariableBlur = num("mask_blend_radius_variable_blur", 10.0, Some(0.0)),
      blurStrengthCurve = num("blur_strength_curve", 3.0),
      pixelationFactor = num("pixelation_factor", 5.0, Some(1e-6)),
      fillColor = fillHex,
      nmsThreshold = num("nms_threshold", 0.5, Some(0.0), Some(1.0)),
      verboseDetection = coerceBool(m.get("verbose_detection"), default = false),
      thresholds = coerceVector(m.get("thresholds"), "thresholds", DefaultThresholds),
      expandHorizontal = coerceVector(m.get("expand_horizontal"), "expand_horizontal", DefaultExpand, 0.0),
      expandVertical = coerceVector(m.get("expand_vertical"), "expand_vertical", DefaultExpand, 0.0),
      nudenet = coerceBool(m.get("nudenet"), default = true),
      safetyChecker = coerceBool(m.get("safety_checker"), default = false),
      safetyCheckerLivePreview = coerceBool(m.get("safety_checker_live_preview"), default = false),
      nsfwjs = coerceBool(m.get("nsfwjs"), default = false),
      nsfwjsSafetyLevel = m.get("nsfwjs_safety_level").collect { case ujson.Str(s) => s }.getOrElse("safe")
    )

  def scaleFactor(w: Int, h: Int): Double =
    math.sqrt((w.toDouble * w.toDouble + h.toDouble * h.toDouble) / 524288.0)

  // ---- preprocessing & ONNX detection ------------------------------------------------------------------------

  private[engine] def toRgb(src: BufferedImage): BufferedImage =
    if src.getType == BufferedImage.TYPE_INT_RGB then src
    else
      val out = new BufferedImage(src.getWidth, src.getHeight, BufferedImage.TYPE_INT_RGB)
      val g = out.createGraphics()
      try
        g.setColor(Color.WHITE)
        g.fillRect(0, 0, src.getWidth, src.getHeight)
        g.drawImage(src, 0, 0, null)
      finally g.dispose()
      out

  private[engine] def copyRgb(src: BufferedImage): BufferedImage =
    val out = new BufferedImage(src.getWidth, src.getHeight, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    try g.drawImage(src, 0, 0, null)
    finally g.dispose()
    out

  def preprocess(img: BufferedImage, width: Int = InputSize, height: Int = InputSize): Array[Float] =
    val rgb = toRgb(img)
    val w = rgb.getWidth
    val h = rgb.getHeight
    val (newW, newH) =
      if w > h then (width, math.max(1, math.round(height.toDouble * h / w).toInt))
      else (math.max(1, math.round(width.toDouble * w / h).toInt), height)
    val canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g = canvas.createGraphics()
    try
      g.setColor(Color.WHITE)
      g.fillRect(0, 0, width, height)
      g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      g.drawImage(rgb, (width - newW) / 2, (height - newH) / 2, newW, newH, null)
    finally g.dispose()
    val pixels = canvas.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    val plane = width * height
    val nchw = new Array[Float](3 * plane)
    var i = 0
    while i < plane do
      val argb = pixels(i)
      nchw(i) = ((argb >>> 16) & 0xff) / 255.0f
      nchw(plane + i) = ((argb >>> 8) & 0xff) / 255.0f
      nchw(2 * plane + i) = (argb & 0xff) / 255.0f
      i += 1
    nchw

  private def iou(ax: Double, ay: Double, aw: Double, ah: Double,
                  bx: Double, by: Double, bw: Double, bh: Double): Double =
    val x1 = math.max(ax, bx)
    val y1 = math.max(ay, by)
    val x2 = math.min(ax + aw, bx + bw)
    val y2 = math.min(ay + ah, by + bh)
    val iw = math.max(0.0, x2 - x1)
    val ih = math.max(0.0, y2 - y1)
    val inter = iw * ih
    if inter <= 0.0 then 0.0
    else
      val union = aw * ah + bw * bh - inter
      if union <= 0.0 then 0.0 else inter / union

  def decodeDetections(
      channels: Array[Array[Float]],
      imgW: Int,
      imgH: Int,
      thresholds: Array[Float],
      expandH: Array[Float],
      expandV: Array[Float],
      nmsThreshold: Double,
      inputW: Int = InputSize,
      inputH: Int = InputSize
  ): Seq[DetectionBox] =
    if channels.length < 4 + Labels.size then
      throw new IllegalArgumentException(s"Unexpected NudeNet channel count: ${channels.length}")
    val numBoxes = channels(0).length
    val candidates = ArrayBuffer.empty[(Double, Double, Double, Double, Int, Float)]
    var b = 0
    while b < numBoxes do
      var bestCls = 0
      var bestScore = channels(4)(b)
      var c = 1
      while c < Labels.size do
        val sc = channels(4 + c)(b)
        if sc > bestScore then
          bestScore = sc
          bestCls = c
        c += 1
      if bestScore > thresholds(bestCls) then
        val bw = channels(2)(b).toDouble
        val bh = channels(3)(b).toDouble
        val bx = channels(0)(b).toDouble - bw * 0.5
        val by = channels(1)(b).toDouble - bh * 0.5
        candidates += ((bx, by, bw, bh, bestCls, bestScore))
      b += 1

    if candidates.isEmpty then return Nil

    val picked =
      if nmsThreshold < 1.0 then
        val sorted = candidates.sortBy(-_._6)
        val kept = ArrayBuffer.empty[(Double, Double, Double, Double, Int, Float)]
        sorted.foreach { cand =>
          val overlaps = kept.exists(k => iou(k._1, k._2, k._3, k._4, cand._1, cand._2, cand._3, cand._4) > nmsThreshold)
          if !overlaps then kept += cand
        }
        kept
      else candidates

    val offset = math.abs(imgW - imgH) / 2.0
    picked.map { (bx0, by0, bw0, bh0, cls, score) =>
      val (sx, sy, sw, sh) =
        if imgW > imgH then
          val s = imgW.toDouble / inputW.toDouble
          (bx0 * s, by0 * s - offset, bw0 * s, bh0 * s)
        else
          val s = imgH.toDouble / inputH.toDouble
          (bx0 * s - offset, by0 * s, bw0 * s, bh0 * s)
      val w = sw * expandH(cls).toDouble
      val h = sh * expandV(cls).toDouble
      val x1 = math.rint(sx - (w - sw) * 0.5)
      val y1 = math.rint(sy - (h - sh) * 0.5)
      val x2 = math.rint((sx - (w - sw) * 0.5) + w)
      val y2 = math.rint((sy - (h - sh) * 0.5) + h)
      DetectionBox(x1, y1, x2, y2, w, h, cls, score)
    }.toVector

  def detectMask(
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

  // ---- mask rendering & Gaussian blur ------------------------------------------------------------------------

  def renderMask(imgW: Int, imgH: Int, boxes: Seq[DetectionBox], maskShape: String, roundRadius: Double): BufferedImage =
    val mask = new BufferedImage(imgW, imgH, BufferedImage.TYPE_BYTE_GRAY)
    val g = mask.createGraphics()
    try
      g.setColor(Color.BLACK)
      g.fillRect(0, 0, imgW, imgH)
      g.setColor(Color.WHITE)
      if maskShape == "Entire image" || maskShape == EntireImageAlways then
        g.fillRect(0, 0, imgW, imgH)
      else
        boxes.foreach { b =>
          val w = math.max(0.0, b.x2 - b.x1)
          val h = math.max(0.0, b.y2 - b.y1)
          if w > 0.0 && h > 0.0 then
            maskShape match
              case "Ellipse" =>
                g.fill(new Ellipse2D.Double(b.x1, b.y1, w, h))
              case "Rectangle" =>
                g.fill(new Rectangle2D.Double(b.x1, b.y1, w, h))
              case "Rounded rectangle" =>
                val bw = math.max(0.0, b.w)
                val bh = math.max(0.0, b.h)
                val r =
                  if roundRadius > 1.0 then roundRadius * math.sqrt((bw * bw + bh * bh) / 524288.0)
                  else if roundRadius < 0.0 then -roundRadius
                  else (math.min(bw, bh) / 2.0) * roundRadius
                val arc = math.max(0.0, math.rint(r) * 2.0)
                g.fill(new RoundRectangle2D.Double(b.x1, b.y1, w, h, arc, arc))
              case other =>
                invalid(s"mask_shape must be one of ${MaskShapes.mkString(", ")} (got '$other')")
        }
    finally g.dispose()
    mask

  private def gaussianKernel(sigma: Double): Array[Float] =
    if sigma <= 0.1 then Array(1.0f)
    else
      val r = math.max(1, math.ceil(sigma * 3.0).toInt)
      val k = new Array[Float](2 * r + 1)
      val denom = 2.0 * sigma * sigma
      var sum = 0.0
      var i = -r
      while i <= r do
        val v = math.exp(-(i * i) / denom)
        k(i + r) = v.toFloat
        sum += v
        i += 1
      var j = 0
      while j < k.length do
        k(j) = (k(j) / sum).toFloat
        j += 1
      k

  private[engine] def blurGray(src: BufferedImage, radius: Double): BufferedImage =
    val gray = toGray(src)
    if radius <= 0.1 then return copyGray(gray)
    val w = gray.getWidth
    val h = gray.getHeight
    val k = gaussianKernel(radius)
    val rad = k.length / 2
    val bytes = gray.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val horiz = new Array[Float](w * h)
    var y = 0
    while y < h do
      val row = y * w
      var x = 0
      while x < w do
        var acc = 0.0f
        var t = -rad
        while t <= rad do
          val sx = math.max(0, math.min(w - 1, x + t))
          acc += (bytes(row + sx) & 0xff) * k(t + rad)
          t += 1
        horiz(row + x) = acc
        x += 1
      y += 1
    val out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val outBytes = out.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    y = 0
    while y < h do
      val row = y * w
      var x = 0
      while x < w do
        var acc = 0.0f
        var t = -rad
        while t <= rad do
          val sy = math.max(0, math.min(h - 1, y + t))
          acc += horiz(sy * w + x) * k(t + rad)
          t += 1
        outBytes(row + x) = math.max(0, math.min(255, math.round(acc))).toByte
        x += 1
      y += 1
    out

  private[engine] def blurRgb(src: BufferedImage, radius: Double): BufferedImage =
    val rgb = toRgb(src)
    if radius <= 0.1 then return copyRgb(rgb)
    val w = rgb.getWidth
    val h = rgb.getHeight
    val k = gaussianKernel(radius)
    val rad = k.length / 2
    val pixels = rgb.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    val hr = new Array[Float](w * h)
    val hg = new Array[Float](w * h)
    val hb = new Array[Float](w * h)
    var y = 0
    while y < h do
      val row = y * w
      var x = 0
      while x < w do
        var ar = 0.0f; var ag = 0.0f; var ab = 0.0f
        var t = -rad
        while t <= rad do
          val sx = math.max(0, math.min(w - 1, x + t))
          val p = pixels(row + sx)
          val kv = k(t + rad)
          ar += ((p >>> 16) & 0xff) * kv
          ag += ((p >>> 8) & 0xff) * kv
          ab += (p & 0xff) * kv
          t += 1
        hr(row + x) = ar; hg(row + x) = ag; hb(row + x) = ab
        x += 1
      y += 1
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val outPx = out.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    y = 0
    while y < h do
      val row = y * w
      var x = 0
      while x < w do
        var ar = 0.0f; var ag = 0.0f; var ab = 0.0f
        var t = -rad
        while t <= rad do
          val sy = math.max(0, math.min(h - 1, y + t))
          val idx = sy * w + x
          val kv = k(t + rad)
          ar += hr(idx) * kv
          ag += hg(idx) * kv
          ab += hb(idx) * kv
          t += 1
        val r = math.max(0, math.min(255, math.round(ar)))
        val g = math.max(0, math.min(255, math.round(ag)))
        val b = math.max(0, math.min(255, math.round(ab)))
        outPx(row + x) = (r << 16) | (g << 8) | b
        x += 1
      y += 1
    out

  private[engine] def toGray(src: BufferedImage): BufferedImage =
    if src.getType == BufferedImage.TYPE_BYTE_GRAY then src
    else
      val out = new BufferedImage(src.getWidth, src.getHeight, BufferedImage.TYPE_BYTE_GRAY)
      val g = out.createGraphics()
      try g.drawImage(src, 0, 0, null)
      finally g.dispose()
      out

  private def copyGray(src: BufferedImage): BufferedImage =
    val out = new BufferedImage(src.getWidth, src.getHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g = out.createGraphics()
    try g.drawImage(src, 0, 0, null)
    finally g.dispose()
    out

  // ---- censor filters ----------------------------------------------------------------------------------------

  private def combineWithMask(base: BufferedImage, mask: BufferedImage, processed: BufferedImage): BufferedImage =
    val bRgb = toRgb(base)
    val pRgb = toRgb(processed)
    val mGray = toGray(mask)
    val w = bRgb.getWidth
    val h = bRgb.getHeight
    val bPx = bRgb.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    val pPx = pRgb.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    val mBy = mGray.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val oPx = out.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    var i = 0
    while i < bPx.length do
      val a = mBy(i) & 0xff
      if a == 0 then oPx(i) = bPx(i)
      else if a == 255 then oPx(i) = pPx(i)
      else
        val ia = 255 - a
        val bp = bPx(i); val pp = pPx(i)
        val r = (((bp >>> 16) & 0xff) * ia + ((pp >>> 16) & 0xff) * a + 127) / 255
        val g = (((bp >>> 8) & 0xff) * ia + ((pp >>> 8) & 0xff) * a + 127) / 255
        val b = ((bp & 0xff) * ia + (pp & 0xff) * a + 127) / 255
        oPx(i) = (r << 16) | (g << 8) | b
      i += 1
    out

  /** Pixelate filter (mosaic): area-average downscale by factor^2 then nearest-neighbor upscale, masked in. */
  def pixelate(img: BufferedImage, mask: BufferedImage, pixelationFactor: Double = 5.0): BufferedImage =
    val rgb = toRgb(img)
    val w = rgb.getWidth
    val h = rgb.getHeight
    val denom = math.max(1e-6, pixelationFactor * pixelationFactor)
    val sw = math.max(1, math.round(w / denom).toInt)
    val sh = math.max(1, math.round(h / denom).toInt)
    val px = rgb.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    val small = new BufferedImage(sw, sh, BufferedImage.TYPE_INT_RGB)
    val spx = small.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    var sy = 0
    while sy < sh do
      val y0 = (sy * h) / sh
      val y1 = math.max(y0 + 1, ((sy + 1) * h) / sh)
      var sx = 0
      while sx < sw do
        val x0 = (sx * w) / sw
        val x1 = math.max(x0 + 1, ((sx + 1) * w) / sw)
        var sr = 0L; var sg = 0L; var sb = 0L; var cnt = 0L
        var y = y0
        while y < y1 do
          val row = y * w
          var x = x0
          while x < x1 do
            val p = px(row + x)
            sr += (p >>> 16) & 0xff
            sg += (p >>> 8) & 0xff
            sb += p & 0xff
            cnt += 1L
            x += 1
          y += 1
        val c = math.max(1L, cnt)
        spx(sy * sw + sx) = (((sr / c).toInt & 0xff) << 16) | (((sg / c).toInt & 0xff) << 8) | ((sb / c).toInt & 0xff)
        sx += 1
      sy += 1
    val upscaled = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = upscaled.createGraphics()
    try
      g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR)
      g.drawImage(small, 0, 0, w, h, null)
    finally g.dispose()
    combineWithMask(rgb, mask, upscaled)

  def gaussianBlur(img: BufferedImage, mask: BufferedImage, blurRadius: Double = 10.0): BufferedImage =
    combineWithMask(img, mask, blurRgb(img, blurRadius))

  def variableBlur(img: BufferedImage, mask: BufferedImage, blurRadius: Double = 10.0, curve: Double = 3.0): BufferedImage =
    val rgb = toRgb(img)
    if curve <= 0.0 then return copyRgb(rgb)
    if curve > 6.0 then return blurRgb(rgb, blurRadius)
    val mGray = toGray(mask)
    val w = rgb.getWidth
    val h = rgb.getHeight
    val mBy = mGray.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val seen = new Array[Boolean](256)
    var i = 0
    while i < mBy.length do
      seen(mBy(i) & 0xff) = true
      i += 1
    val levels = (0 until 256).filter(seen).toArray
    if levels.isEmpty then return copyRgb(rgb)
    val exp = (12.0 - 2.0 * curve) / curve
    // Fast quantized blur ladder (<= 24 blur passes) with per-pixel linear level interpolation.
    val stepLevels: Array[Int] =
      if levels.length <= 24 then levels
      else (0 until 24).map(k => (k * 255) / 23).toArray
    val blurredLayers: Array[Array[Int]] = stepLevels.map { lvl =>
      val r = math.pow(lvl / 255.0, exp) * blurRadius
      blurRgb(rgb, r).getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    }
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val oPx = out.getRaster.getDataBuffer.asInstanceOf[DataBufferInt].getData
    if levels.length <= 24 then
      val levelMap = new Array[Int](256)
      stepLevels.zipWithIndex.foreach { (lvl, idx) => levelMap(lvl) = idx }
      i = 0
      while i < mBy.length do
        val idx = levelMap(mBy(i) & 0xff)
        oPx(i) = blurredLayers(idx)(i)
        i += 1
    else
      val maxIdx = stepLevels.length - 1
      i = 0
      while i < mBy.length do
        val v = mBy(i) & 0xff
        val pos = (v / 255.0) * maxIdx
        val lo = math.min(maxIdx, pos.toInt)
        val hi = math.min(maxIdx, lo + 1)
        val t = pos - lo
        if lo == hi || t <= 0.0 then oPx(i) = blurredLayers(lo)(i)
        else
          val p0 = blurredLayers(lo)(i)
          val p1 = blurredLayers(hi)(i)
          val r = math.round(((p0 >>> 16) & 0xff) * (1.0 - t) + ((p1 >>> 16) & 0xff) * t).toInt
          val g = math.round(((p0 >>> 8) & 0xff) * (1.0 - t) + ((p1 >>> 8) & 0xff) * t).toInt
          val b = math.round((p0 & 0xff) * (1.0 - t) + (p1 & 0xff) * t).toInt
          oPx(i) = (r << 16) | (g << 8) | b
        i += 1
    out

  def fillColor(img: BufferedImage, mask: BufferedImage, colorHex: String = "#000000"): BufferedImage =
    val rgb = toRgb(img)
    val (_, color) = parseColor(Some(ujson.Str(colorHex)))
    val filled = new BufferedImage(rgb.getWidth, rgb.getHeight, BufferedImage.TYPE_INT_RGB)
    val g = filled.createGraphics()
    try
      g.setColor(color)
      g.fillRect(0, 0, rgb.getWidth, rgb.getHeight)
    finally g.dispose()
    combineWithMask(rgb, mask, filled)

  def applyFilter(
      img: BufferedImage,
      mask: BufferedImage,
      filterType: String,
      blurRadius: Double,
      blurStrengthCurve: Double,
      pixelationFactor: Double,
      fillColorHex: String
  ): BufferedImage =
    filterType match
      case "Variable blur" => variableBlur(img, mask, blurRadius, blurStrengthCurve)
      case "Gaussian Blur" => gaussianBlur(img, mask, blurRadius)
      case "Pixelate"      => pixelate(img, mask, pixelationFactor)
      case "Fill color"    => fillColor(img, mask, fillColorHex)
      case "No censor" | "Disable" => copyRgb(img)
      case other           => invalid(s"filter_type must be one of ${Filters.mkString(", ")} (got '$other')")

  // ---- workflows: generation, preview, and Extras tool -------------------------------------------------------

  type Detector = (BufferedImage, Array[Float], Array[Float], Array[Float], Double, String, Double, Option[String => Unit]) => Option[BufferedImage]

  def censorGeneration(
      img: BufferedImage,
      cfg: GenerationConfig,
      home: AppHome,
      modelPath: Option[Path] = None,
      log: Option[String => Unit] = None,
      detectorOverride: Option[Detector] = None
  ): CensorGenerationResult =
    if !cfg.nudenetActive(preview = false) then
      return CensorGenerationResult(img, None, None, detected = false)
    val logIfVerbose = if cfg.verboseDetection then log else None
    val found = detectorOverride match
      case Some(det) =>
        det(img, cfg.thresholds, cfg.expandHorizontal, cfg.expandVertical, cfg.nmsThreshold,
          cfg.maskShape, cfg.rectangleRoundRadius, logIfVerbose)
      case None =>
        detectMask(img, home, cfg.thresholds, cfg.expandHorizontal, cfg.expandVertical, cfg.nmsThreshold,
          cfg.maskShape, cfg.rectangleRoundRadius, modelPath, logIfVerbose)
    found match
      case None => CensorGenerationResult(img, None, None, detected = false)
      case Some(rawMask) =>
        val rgb = toRgb(img)
        val before = Option.when(cfg.saveBeforeCensor)(copyRgb(rgb))
        val factor = scaleFactor(rgb.getWidth, rgb.getHeight)
        val blend = if cfg.genFilterType == "Variable blur" then cfg.maskBlendRadiusVariableBlur else cfg.maskBlendRadius
        val blendedMask = blurGray(rawMask, blend * factor)
        val censored = applyFilter(
          rgb, blendedMask, cfg.genFilterType,
          cfg.blurRadius * factor, cfg.blurStrengthCurve, cfg.pixelationFactor, cfg.fillColor
        )
        CensorGenerationResult(censored, before, Option.when(cfg.saveMask)(blendedMask), detected = true)

  def censorPreviewB64(
      b64: String,
      cfg: GenerationConfig,
      home: AppHome,
      modelPath: Option[Path] = None,
      detectorOverride: Option[Detector] = None
  ): String =
    if !cfg.nudenetActive(preview = true) || b64 == null || b64.isEmpty then return b64
    val decoded = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder.decode(b64)))
    if decoded == null then return b64
    val found = detectorOverride match
      case Some(det) =>
        det(decoded, cfg.thresholds, cfg.expandHorizontal, cfg.expandVertical, cfg.nmsThreshold,
          cfg.maskShape, cfg.rectangleRoundRadius, None)
      case None =>
        detectMask(decoded, home, cfg.thresholds, cfg.expandHorizontal, cfg.expandVertical, cfg.nmsThreshold,
          cfg.maskShape, cfg.rectangleRoundRadius, modelPath, None)
    found match
      case None => b64
      case Some(rawMask) =>
        val rgb = toRgb(decoded)
        val factor = scaleFactor(rgb.getWidth, rgb.getHeight)
        val blend = if cfg.genFilterType == "Variable blur" then cfg.maskBlendRadiusVariableBlur else cfg.maskBlendRadius
        val blendedMask = blurGray(rawMask, blend * factor)
        val censored = applyFilter(
          rgb, blendedMask, cfg.livePreviewFilterType,
          cfg.blurRadius * factor, cfg.blurStrengthCurve, cfg.pixelationFactor, cfg.fillColor
        )
        encodeJpegB64(censored)

  private def encodeJpegB64(img: BufferedImage): String =
    val rgb = toRgb(img)
    val baos = new java.io.ByteArrayOutputStream()
    val writers = ImageIO.getImageWritersByFormatName("jpeg")
    val writer = writers.next()
    try
      val param = writer.getDefaultWriteParam
      if param.canWriteCompressed then
        param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT)
        param.setCompressionQuality(0.85f)
      Using.resource(new MemoryCacheImageOutputStream(baos)) { out =>
        writer.setOutput(out)
        writer.write(null, new IIOImage(rgb, null, null), param)
      }
    finally writer.dispose()
    Base64.getEncoder.encodeToString(baos.toByteArray)

  private def maskUnion(existing: BufferedImage, added: BufferedImage): BufferedImage =
    val a = toGray(existing)
    val b = toGray(added)
    val w = a.getWidth
    val h = a.getHeight
    val aBy = a.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val bBy = b.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val oBy = out.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    var i = 0
    while i < oBy.length do
      val va = aBy(i) & 0xff
      val vb = bBy(i) & 0xff
      oBy(i) = (if vb > 0 then vb else va).toByte
      i += 1
    out

  def censorTool(
      params: ujson.Obj,
      home: AppHome,
      outputDir: Path,
      log: Option[String => Unit] = None,
      detectorOverride: Option[Detector] = None
  ): ujson.Obj =
    val m = params.value
    val checker = coerceChoice(
      m.get("checker").filter { case ujson.Str(s) => s.trim.nonEmpty; case _ => false },
      "nudenet", "checker", ToolCheckers
    )
    if checker != "nudenet" then
      throw EngineError(Protocol.ErrorCodes.Unsupported,
        s"NSFW Checker ($checker): native scala-torch mode currently supports NudeNet only; switch Engine backend to Python to use $checker.")

    val imagePathStr = m.get("image_path").collect { case ujson.Str(s) if s.trim.nonEmpty => s.trim }
      .getOrElse(invalid("image_path is required"))
    val imagePath = Path.of(imagePathStr)
    val srcImg = PngMetadata.readImage(imagePath)
    val w = srcImg.getWidth
    val h = srcImg.getHeight

    val filterType = coerceChoice(
      m.get("filter_type").filter { case ujson.Str(s) => s.trim.nonEmpty; case _ => false },
      "Variable blur", "filter_type", Filters
    )
    val maskShape = coerceChoice(
      m.get("mask_shape").filter { case ujson.Str(s) => s.trim.nonEmpty; case _ => false },
      "Ellipse", "mask_shape", MaskShapes :+ EntireImageAlways
    )
    val enableNudenet = coerceBool(m.get("enable_nudenet"), default = true)
    val thresholds = coerceVector(m.get("thresholds"), "thresholds", DefaultThresholds)
    val expandH = coerceVector(m.get("expand_horizontal"), "expand_horizontal", DefaultExpand, 0.0)
    val expandV = coerceVector(m.get("expand_vertical"), "expand_vertical", DefaultExpand, 0.0)
    val nms = m.get("nms_threshold").filter(_ != ujson.Null).fold(0.5)(coerceNum(_, "nms_threshold", Some(0.0), Some(1.0)))
    val roundRadius = m.get("rectangle_round_radius").filter(_ != ujson.Null).fold(0.5)(coerceNum(_, "rectangle_round_radius"))
    val verbose = coerceBool(m.get("verbose_detection"), default = false)
    val explicitModelPath = m.get("model_path").collect { case ujson.Str(s) if s.trim.nonEmpty => Path.of(s.trim) }

    var censorMaskOpt: Option[BufferedImage] = None
    var detected = false

    if maskShape == EntireImageAlways then
      censorMaskOpt = Some(renderMask(w, h, Nil, EntireImageAlways, roundRadius))
    else
      m.get("mask_path").collect { case ujson.Str(s) if s.trim.nonEmpty => Path.of(s.trim) }.foreach { customPath =>
        val customImg = PngMetadata.readImage(customPath)
        val scaled = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        val g = scaled.createGraphics()
        try
          g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
          g.drawImage(toGray(customImg), 0, 0, w, h, null)
        finally g.dispose()
        censorMaskOpt = Some(scaled)
      }
      if enableNudenet then
        val logIfVerbose = if verbose then log else None
        val found = detectorOverride match
          case Some(det) => det(srcImg, thresholds, expandH, expandV, nms, maskShape, roundRadius, logIfVerbose)
          case None => detectMask(srcImg, home, thresholds, expandH, expandV, nms, maskShape, roundRadius, explicitModelPath, logIfVerbose)
        found.foreach { foundMask =>
          detected = true
          censorMaskOpt = Some(censorMaskOpt.fold(toGray(foundMask))(maskUnion(_, foundMask)))
        }

    Files.createDirectories(outputDir)
    val outFile = outputDir.resolve(s"nsfw-${UUID.randomUUID().toString.take(8)}.png")
    censorMaskOpt match
      case None =>
        PngMetadata.write(toRgb(srcImg), outFile, Nil)
        ujson.Obj(
          "path" -> outFile.toAbsolutePath.toString,
          "outputs" -> ujson.Obj(),
          "info" -> ujson.Obj("censored" -> 0, "detected" -> 0)
        )
      case Some(rawMask) =>
        val factor = scaleFactor(w, h)
        val defaultBlend = if filterType == "Variable blur" then 10.0 else 0.0
        val blend = m.get("mask_blend_radius").filter(_ != ujson.Null).fold(defaultBlend)(coerceNum(_, "mask_blend_radius", Some(0.0)))
        val blendedMask = blurGray(rawMask, blend * factor)
        val blurRadius = m.get("blur_radius").filter(_ != ujson.Null).fold(10.0 * factor)(coerceNum(_, "blur_radius", Some(0.0)))
        val curve = m.get("blur_strength_curve").filter(_ != ujson.Null).fold(3.0)(coerceNum(_, "blur_strength_curve"))
        val pixFactor = m.get("pixelation_factor").filter(_ != ujson.Null).fold(5.0)(coerceNum(_, "pixelation_factor", Some(1e-6)))
        val (fillHex, _) = parseColor(m.get("fill_color"), "#000000")
        val outImg = applyFilter(srcImg, blendedMask, filterType, blurRadius, curve, pixFactor, fillHex)
        PngMetadata.write(outImg, outFile, Nil)
        val outputs = ujson.Obj()
        if coerceBool(m.get("output_mask"), default = false) then
          val maskFile = outputDir.resolve(s"nsfw-mask-${UUID.randomUUID().toString.take(8)}.png")
          PngMetadata.write(blendedMask, maskFile, Nil)
          outputs("mask") = maskFile.toAbsolutePath.toString
        ujson.Obj(
          "path" -> outFile.toAbsolutePath.toString,
          "outputs" -> outputs,
          "info" -> ujson.Obj("censored" -> 1, "detected" -> (if detected then 1 else 0))
        )

private final class NativeNsfwCensorMarker
