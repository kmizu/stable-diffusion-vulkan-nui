package sdfx.vulkan

import sdfx.core.model.GenerationParams
import sdfx.core.protocol.{EngineError, GenerationRequest, Json}

class VulkanWorkerSuite extends munit.FunSuite:
  test("masked generation restores source pixels and original full-resolution canvas") {
    import java.awt.image.BufferedImage
    import java.nio.file.Files
    import javax.imageio.ImageIO
    import sdfx.core.model.Img2ImgParams
    val sourceFile = Files.createTempFile("source", ".png")
    val maskFile = Files.createTempFile("mask", ".png")
    try
      val source = new BufferedImage(128, 96, BufferedImage.TYPE_INT_RGB)
      val mask = new BufferedImage(128, 96, BufferedImage.TYPE_INT_RGB)
      val mark = 0x123456
      source.setRGB(0, 0, mark)
      for x <- 20 until 50; y <- 30 until 65 do mask.setRGB(x, y, 0xffffff)
      ImageIO.write(source, "png", sourceFile.toFile)
      ImageIO.write(mask, "png", maskFile.toFile)
      val ip = Img2ImgParams(sourceFile.toString, Some(maskFile.toString), maskBlur = 0,
        inpaintFullRes = true, inpaintFullResPadding = 3)
      val inputs = VulkanImageInputs.prepare(ip, 64, 64, 5)
      val white = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB)
      for x <- 0 until 64; y <- 0 until 64 do white.setRGB(x, y, 0xffffff)
      val restored = inputs.finish(white)
      assertEquals((restored.getWidth, restored.getHeight), (128, 96))
      assertEquals(restored.getRGB(0, 0) & 0xffffff, mark)
      assertEquals(restored.getRGB(25, 40) & 0xffffff, 0xffffff)
      assertEquals(restored.getRGB(19, 40) & 0xffffff, 0)
    finally
      Files.deleteIfExists(sourceFile)
      Files.deleteIfExists(maskFile)
  }

  test("hires target preserves aspect ratio when one resize side is explicit") {
    import sdfx.core.model.HiresParams
    val p = GenerationParams(width = 512, height = 384,
      hires = HiresParams(enabled = true, resizeX = 1024, upscaler = "Latent (nearest-exact)"))
    val h = VulkanImageInputs.hires(GenerationRequest(p,"outputs")).get
    assertEquals((h.width(), h.height(), h.upscaler()), (1024, 768, "latent_nearest_exact"))
  }

  test("GUI defaults and allocated variation seed select the default native sampler") {
    val request = GenerationRequest(GenerationParams(subseed = 1234), "outputs", vaeRecoveryDir = "outputs")
    val parsed = VulkanWorker.generation(Json.toJs(request))
    assertEquals(parsed.request.params.subseed, 1234L)
    assertEquals(parsed.scheduler, "Karras")
  }

  test("an enabled engine hook is rejected by name") {
    val request = GenerationRequest(GenerationParams(hooks = Map("freeu" -> ujson.Obj("enabled" -> true))), "outputs")
    val error = intercept[EngineError](VulkanWorker.generation(Json.toJs(request)))
    assertEquals(error.code, "unsupported")
    assert(error.message.contains("freeu"))
  }

  test("registry uses final extension and preserves names of nested checkpoints") {
    val root = java.nio.file.Files.createTempDirectory("vulkan-catalog")
    val folder = java.nio.file.Files.createDirectory(root.resolve("anime"))
    val model = java.nio.file.Files.createFile(folder.resolve("animagine-xl-3.1.SAFETENSORS"))
    try
      val files = VulkanProtocol.models(ujson.Obj("dirs" -> ujson.Arr(root.toString, folder.toString)))("models").arr
      assertEquals(files.map(_("name").str).toList, List("anime/animagine-xl-3.1"))
      assertEquals(files.head("path").str, model.toString)
    finally
      java.nio.file.Files.delete(model)
      java.nio.file.Files.delete(folder)
      java.nio.file.Files.delete(root)
  }

  test("adapter selection round-trips through the UI and rejects ambiguous names") {
    val a = new VulkanInference.Device("Vulkan0", "Example")
    val b = new VulkanInference.Device("Vulkan1", "Example")
    val all = Seq(a, b)
    assertEquals(VulkanProtocol.select(all, None), a)
    assertEquals(VulkanProtocol.select(all, Some("vulkan1")), b)
    assertEquals(VulkanProtocol.select(all, Some(VulkanProtocol.display(b))), b)
    intercept[EngineError](VulkanProtocol.select(all, Some("Example")))
  }
