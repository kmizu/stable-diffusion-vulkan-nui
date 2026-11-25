package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.model.GenerationParams
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.*

import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer

class NativeNsfwCensorSuite extends munit.FunSuite:
  private val home = AppHome(Path.of(".").toAbsolutePath.normalize())

  private def patternedImage(w: Int = 256, h: Int = 256): BufferedImage =
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until h; x <- 0 until w do
      val r = ((x * 5 + y * 2) & 0xff)
      val g = ((x * 3 + y * 7) & 0xff)
      val b = ((x + y * 11) & 0xff)
      img.setRGB(x, y, (r << 16) | (g << 8) | b)
    img

  private def rectMask(w: Int, h: Int, x1: Int, y1: Int, x2: Int, y2: Int): BufferedImage =
    val m = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val g = m.createGraphics()
    try
      g.setColor(Color.BLACK)
      g.fillRect(0, 0, w, h)
      g.setColor(Color.WHITE)
      g.fillRect(x1, y1, x2 - x1, y2 - y1)
    finally g.dispose()
    m

  test("bundled NudeNet model passes SHA-256 verification and rejects tampering") {
    val modelPath = NativeNsfwCensor.resolveModelPath(home)
    assert(modelPath.isDefined, "models/nudenet/best.onnx must be resolved")
    val verified = NativeNsfwCensor.verifyModel(modelPath.get)
    assertEquals(Files.size(verified), NativeNsfwCensor.ModelSize)
    assert(NativeNsfwCensor.available(home))

    val tmp = Files.createTempFile("bad-nudenet-", ".onnx")
    try
      Files.write(tmp, Array.fill[Byte](128)(1.toByte))
      intercept[EngineError](NativeNsfwCensor.verifyModel(tmp))
    finally Files.deleteIfExists(tmp)
  }

  test("real ONNX Runtime JVM inference runs cleanly on bundled NudeNet model") {
    val blank = new BufferedImage(320, 240, BufferedImage.TYPE_INT_RGB)
    val mask = NativeNsfwCensor.detectMask(
      blank,
      home,
      NativeNsfwCensor.DefaultThresholds,
      NativeNsfwCensor.DefaultExpand,
      NativeNsfwCensor.DefaultExpand,
      0.5,
      "Ellipse",
      0.5
    )
    assertEquals(mask, None)
  }

  test("decodeDetections performs greedy NMS, rescaling, and per-class expansion") {
    val numBoxes = 3
    val ch = Array.fill(4 + NativeNsfwCensor.Labels.size, numBoxes)(0.0f)
    // box 0: Face_female (cls 1), score 0.90 at (160, 160, 80, 80)
    ch(0)(0) = 160f; ch(1)(0) = 160f; ch(2)(0) = 80f; ch(3)(0) = 80f; ch(4 + 1)(0) = 0.90f
    // box 1: overlaps box 0 heavily, lower score 0.70 -> suppressed when nms=0.5, kept when nms=1.0
    ch(0)(1) = 162f; ch(1)(1) = 162f; ch(2)(1) = 80f; ch(3)(1) = 80f; ch(4 + 1)(1) = 0.70f
    // box 2: distant box, score 0.80 -> kept
    ch(0)(2) = 40f; ch(1)(2) = 40f; ch(2)(2) = 30f; ch(3)(2) = 30f; ch(4 + 1)(2) = 0.80f

    val thresholds = Array.fill(NativeNsfwCensor.Labels.size)(1.0f)
    thresholds(1) = 0.25f
    val expandH = Array.fill(NativeNsfwCensor.Labels.size)(1.0f)
    expandH(1) = 1.5f
    val expandV = Array.fill(NativeNsfwCensor.Labels.size)(1.0f)

    val suppressed = NativeNsfwCensor.decodeDetections(ch, 640, 640, thresholds, expandH, expandV, 0.5)
    assertEquals(suppressed.size, 2)
    assertEquals(suppressed.head.score, 0.90f)
    assertEquals(math.round(suppressed.head.x2 - suppressed.head.x1).toInt, 240) // 80 * 2 * 1.5

    val noNms = NativeNsfwCensor.decodeDetections(ch, 640, 640, thresholds, expandH, expandV, 1.0)
    assertEquals(noNms.size, 3)
  }

  test("renderMask draws Ellipse, Rectangle, Rounded rectangle, and Entire image shapes") {
    val box = NativeNsfwCensor.DetectionBox(20, 20, 100, 100, 80, 80, 1, 0.9f)
    val ellipse = NativeNsfwCensor.renderMask(120, 120, Seq(box), "Ellipse", 0.5)
    // Corner (21, 21) is outside inscribed ellipse, inside rectangle
    assertEquals(ellipse.getRGB(21, 21) & 0xff, 0)
    assertEquals(ellipse.getRGB(60, 60) & 0xff, 255)

    val rect = NativeNsfwCensor.renderMask(120, 120, Seq(box), "Rectangle", 0.5)
    assertEquals(rect.getRGB(21, 21) & 0xff, 255)

    val rounded = NativeNsfwCensor.renderMask(120, 120, Seq(box), "Rounded rectangle", -12)
    assertEquals(rounded.getRGB(21, 21) & 0xff, 0)
    assertEquals(rounded.getRGB(60, 60) & 0xff, 255)

    val full = NativeNsfwCensor.renderMask(120, 120, Nil, NativeNsfwCensor.EntireImageAlways, 0.5)
    assertEquals(full.getRGB(0, 0) & 0xff, 255)
  }

  test("Pixelate (mosaic), Gaussian Blur, Variable blur, and Fill color only modify masked region") {
    val src = patternedImage(128, 128)
    val mask = rectMask(128, 128, 32, 32, 96, 96)

    val pix = NativeNsfwCensor.pixelate(src, mask, pixelationFactor = 4.0)
    // Outside mask stays bit-exact
    assertEquals(pix.getRGB(10, 10), src.getRGB(10, 10))
    assertEquals(pix.getRGB(115, 115), src.getRGB(115, 115))
    // Inside mask, mosaic cell pixels become uniform across neighbors in the same block
    val c0 = pix.getRGB(40, 40)
    val c1 = pix.getRGB(41, 41)
    assertEquals(c0, c1)
    assert(c0 != src.getRGB(40, 40) || c1 != src.getRGB(41, 41))

    val blurred = NativeNsfwCensor.gaussianBlur(src, mask, 6.0)
    assertEquals(blurred.getRGB(5, 5), src.getRGB(5, 5))
    assert(blurred.getRGB(64, 64) != src.getRGB(64, 64))

    val vblurred = NativeNsfwCensor.variableBlur(src, NativeNsfwCensor.blurGray(mask, 8.0), 8.0, 3.0)
    assertEquals(vblurred.getRGB(5, 5), src.getRGB(5, 5))
    assert(vblurred.getRGB(64, 64) != src.getRGB(64, 64))

    val filled = NativeNsfwCensor.fillColor(src, mask, "#112233")
    assertEquals(filled.getRGB(5, 5), src.getRGB(5, 5))
    assertEquals(filled.getRGB(64, 64) & 0xffffff, 0x112233)
  }

  test("censorGeneration produces mosaicked image, before-censor copy, and censor-mask") {
    val src = patternedImage(128, 128)
    val cfg = NativeNsfwCensor.GenerationConfig(
      genFilterType = "Pixelate",
      pixelationFactor = 4.0,
      saveBeforeCensor = true,
      saveMask = true
    )
    val fixedDetector: NativeNsfwCensor.Detector = (img, _, _, _, _, _, _, log) =>
      log.foreach(_("Face female: score 0.99"))
      Some(rectMask(img.getWidth, img.getHeight, 32, 32, 96, 96))

    val logs = ArrayBuffer.empty[String]
    val res = NativeNsfwCensor.censorGeneration(
      src, cfg.copy(verboseDetection = true), home,
      log = Some(logs += _), detectorOverride = Some(fixedDetector)
    )
    assert(res.detected)
    assertEquals(logs.toSeq, Seq("Face female: score 0.99"))
    assert(res.beforeCensor.isDefined)
    assert(res.mask.isDefined)
    assertEquals(res.beforeCensor.get.getRGB(64, 64), src.getRGB(64, 64))
    assertEquals(res.image.getRGB(8, 8), src.getRGB(8, 8))
    assertEquals(res.image.getRGB(40, 40), res.image.getRGB(41, 41))
  }

  test("censorTool supports Pixelate + Entire image (always) and custom mask union") {
    val tmp = Files.createTempDirectory("nudenet-tool-test")
    try
      val srcPath = tmp.resolve("input.png")
      val customMaskPath = tmp.resolve("mask.png")
      val src = patternedImage(128, 128)
      ImageIO.write(src, "png", srcPath.toFile)
      ImageIO.write(rectMask(128, 128, 16, 16, 48, 48), "png", customMaskPath.toFile)

      val res = NativeNsfwCensor.censorTool(
        ujson.Obj(
          "image_path" -> srcPath.toString,
          "filter_type" -> "Pixelate",
          "mask_shape" -> "Ellipse",
          "mask_path" -> customMaskPath.toString,
          "enable_nudenet" -> false,
          "pixelation_factor" -> 4.0,
          "output_mask" -> true
        ),
        home,
        tmp
      )
      assertEquals(res("info")("censored").num.toInt, 1)
      val outImg = PngMetadata.readImage(Path.of(res("path").str))
      assertEquals(outImg.getRGB(5, 5), src.getRGB(5, 5))
      assertEquals(outImg.getRGB(24, 24), outImg.getRGB(25, 25))
      assert(res("outputs").obj.contains("mask"))
      assert(Files.isRegularFile(Path.of(res("outputs")("mask").str)))
    finally
      val walk = Files.walk(tmp)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists) finally walk.close()
  }

  test("resolveModelPath prefers home.root model and available reports true") {
    val tmp = Files.createTempDirectory("nudenet-gen-test")
    try
      val appHome = AppHome(tmp)
      val nudenetDir = tmp.resolve("models/nudenet")
      Files.createDirectories(nudenetDir)
      val repoModel = Path.of("models/nudenet/best.onnx").toAbsolutePath.normalize()
      val homeModel = nudenetDir.resolve("best.onnx")
      Files.copy(repoModel, homeModel)
      assertEquals(NativeNsfwCensor.resolveModelPath(appHome), Some(homeModel.toAbsolutePath.normalize()))
      assert(NativeNsfwCensor.available(appHome))
    finally
      val walk = Files.walk(tmp)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists) finally walk.close()
  }
