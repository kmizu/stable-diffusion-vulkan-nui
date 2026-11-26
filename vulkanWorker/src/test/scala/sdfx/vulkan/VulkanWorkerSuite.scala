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

  test("graph Hires uses the aligned latent canvas for sampling and inpainting") {
    import sdfx.core.model.{HiresParams, Img2ImgParams}
    val p = GenerationParams(width = 800, height = 776, samplerName = "Euler", scheduler = "Karras",
      hires = HiresParams(enabled = true, resizeX = 864, resizeY = 808, upscaler = "Latent"),
      img2img = Some(Img2ImgParams("source")))
    assertEquals(VulkanGraph.nativeSize(p.width, p.height), (832, 768))
    val graph = VulkanHiresGraph.refinement(GenerationRequest(p, "outputs"), "test", Nil, 3,
      "unet", "text", "", "Vulkan0", customNoise = false, tiled = "", inpaint = true, masked = true)
    for prefix <- Seq("hr_base", "hr_inpaint_condition", "hr_mask") do
      assert(graph.exists(line => line.startsWith(prefix + " =") && line.contains("size = [112,104]")), prefix)
  }

  test("latent upscale keeps the requested number of denoiser steps at fractional strength") {
    import sdfx.core.model.Img2ImgParams
    val p = GenerationParams(steps = 17, img2img = Some(Img2ImgParams("source", denoisingStrength = 0.3)))
    val config = VulkanLatentUpscale.parse(p,ujson.Obj("scheduler" -> "automatic"),"Karras")
    assertEquals(config.fullSteps,57)
    assertEquals(config.scheduler,"Karras")
    intercept[IllegalArgumentException] {
      VulkanLatentUpscale.parse(p.copy(img2img = p.img2img.map(_.copy(denoisingStrength = 1e-300))),ujson.Obj(),"Karras")
    }
  }

  test("GUI defaults and allocated variation seed select the default native sampler") {
    val request = GenerationRequest(GenerationParams(subseed = 1234), "outputs", vaeRecoveryDir = "outputs")
    val parsed = VulkanWorker.generation(Json.toJs(request))
    assertEquals(parsed.request.params.subseed, 1234L)
    assertEquals(parsed.scheduler, "Karras")
  }

  test("hires seed variations follow iteration, repeat and batch order with exact 64-bit primary seeds") {
    val c = ujson.Obj("seed_enabled" -> true,"extras" -> true,"seed" -> "9007199254740993","subseed" -> 77,
      "batch_count" -> 2,"strength" -> 0.5,"resize_width" -> 448,"resize_height" -> 256)
    val order = VulkanHiresTweaks.plan(c,GenerationParams(batchSize=2,nIter=2),3,4)
    assertEquals(order.map(_._1),Vector(0,1,0,1,2,3,2,3))
    assert(order.forall(_._2.seed==9007199254740993L))
    assertEquals(order.map(_._2.subseed),Vector(77L,78L,78L,79L,79L,80L,80L,81L))
    assert(order.forall(x => x._2.width==448 && x._2.height==256))
  }

  test("an unknown enabled engine hook is rejected by name") {
    val request = GenerationRequest(GenerationParams(hooks = Map("unknown_test_hook" -> ujson.Obj("enabled" -> true))), "outputs")
    val error = intercept[EngineError](VulkanWorker.generation(Json.toJs(request)))
    assertEquals(error.code, "unsupported")
    assert(error.message.contains("unknown_test_hook"))
  }

  test("registry uses final extension and preserves names of nested checkpoints") {
    val root = java.nio.file.Files.createTempDirectory("vulkan-catalog")
    val folder = java.nio.file.Files.createDirectory(root.resolve("anime"))
    val models = List("animagine-xl-3.1.SAFETENSORS", "animagine-xl-3.1.ckpt", "paint-xl.Q8.GGUF")
      .map(name => java.nio.file.Files.createFile(folder.resolve(name)))
    try
      val files = VulkanProtocol.models(ujson.Obj("dirs" -> ujson.Arr(root.toString, folder.toString)))("models").arr
      assertEquals(files.map(_("name").str).toList, List("anime/animagine-xl-3.1", "anime/animagine-xl-3.1.ckpt", "anime/paint-xl.Q8.GGUF"))
      assertEquals(files.map(_("path").str).toList, models.map(_.toString))
    finally
      models.foreach(java.nio.file.Files.delete)
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

  test("Incantations SEG supplies only a negative predictor option and preserves the source window") {
    val config = ujson.Obj("seg_active" -> "yes", "seg_blur_sigma" -> 2.5,
      "seg_start_step" -> 4, "seg_end_step" -> 8)
    val hooks = Map("incantations" -> config)
    assertEquals(VulkanIncantations.segArguments(hooks, 768, 512, 6),
      ", seg_blur = 2.5, seg_start = 4, seg_end = 8, seg_width = 768, seg_height = 512")
    assertEquals(VulkanIncantations.segArguments(hooks, 768, 512, 5), "")
    config("seg_blur_sigma") = 0
    assertEquals(VulkanIncantations.segArguments(hooks, 768, 512, 6), "")
    config("seg_blur_sigma") = 11
    config("enabled") = false
    assertEquals(VulkanIncantations.segArguments(hooks, 768, 512, 6), "")
  }

  test("Incantations SEG stays on the negative Hires branch with the base aspect") {
    import sdfx.core.model.HiresParams
    val p = GenerationParams(width = 768, height = 512, samplerName = "Euler",
      hires = HiresParams(enabled = true, resizeX = 896, resizeY = 640, upscaler = "Latent"),
      hooks = Map("incantations" -> ujson.Obj("seg_active" -> true)))
    val graph = VulkanHiresGraph.refinement(GenerationRequest(p, "outputs"), "test", Nil, 3,
      "unet", "text", "", "Vulkan0", customNoise = false, tiled = "")
    assert(graph.exists(line => line.startsWith("hr_neg =") &&
      line.contains("seg_blur = 11.0, seg_start = 0, seg_end = 150, seg_width = 768, seg_height = 512")))
    assert(!graph.exists(line => line.startsWith("hr_pos =") && line.contains("seg_blur")))
    val incompatibleHires = p.copy(hires = p.hires.copy(enabled = false),
      hooks = p.hooks.updated("img2img_hires_fix", ujson.Obj("sampler" -> "Heun", "denoise" -> 0.5)))
    val error = intercept[EngineError](VulkanIncantations.validate(incompatibleHires))
    assert(error.message.contains("SEG with sampler Heun"))
  }

  test("Incantations PAG adds a windowed perturbed positive branch only when reachable") {
    val hooks = Map("incantations" -> ujson.Obj("pag_active" -> "yes", "pag_scale" -> 1.75,
      "pag_sanf" -> true, "pag_start_step" -> 2, "pag_end_step" -> 6))
    assertEquals(VulkanIncantations.pagGraph(hooks, 4, "unet", "controlled", "", "pos", "base", "guided"),
      Vector(
        "guided_perturbed = predictor(unet, controlled, pag_regions = 512)",
        "guided = incantations_pag(base, pos, guided_perturbed, scale = 1.75, start_step = 2, end_step = 6, sanf = true)"
      ))
    assertEquals(VulkanIncantations.pagGraph(hooks, 3, "unet", "controlled", "", "pos", "base", "guided"),
      Vector("guided = base"))
  }

  test("Incantations CFG follows the lagged source clock and its noise interval") {
    val config = ujson.Obj("cfg_interval_enable" -> true, "cfg_interval_schedule" -> "Linear")
    assertEquals(VulkanIncantations.schedule(config, 8, 6, 5), Vector(12f, 12f, 10.5f, 9f, 7.5f))
    config("cfg_interval_low") = ujson.Num(79)
    assertEquals(VulkanIncantations.schedule(config, 8, 6, 5), Vector(12f, 12f, 1f, 1f, 1f))
    config("cfg_interval_schedule") = ujson.Str("Constant")
    assertEquals(VulkanIncantations.schedule(config, 8, 6, 5), Vector.fill(5)(6f))
    val longHires = ujson.Obj("cfg_interval_schedule" -> "Linear")
    assertEquals(VulkanIncantations.schedule(longHires, 4, 6, 8), Vector(12f, 12f, 9f, 6f, 3f, 0f, -3f, -6f))
    for value <- Seq(ujson.Num(1), ujson.Str("on")) do
      assert(VulkanIncantations.cfg(Map("incantations" -> ujson.Obj("cfg_interval_enable" -> value))).nonEmpty)
  }

  test("Incantations CFG curves match the legacy scheduler at two interior steps") {
    val expected = Vector(
      ("Constant", 6.0, 6.0),
      ("Clamp-Linear (c=4.0)", 7.5, 4.0),
      ("Clamp-Linear (c=2.0)", 7.5, 3.0),
      ("Clamp-Linear (c=1.0)", 7.5, 3.0),
      ("Linear", 7.5, 3.0),
      ("Inverse-Linear", 4.5, 9.0),
      ("Cosine", 8.296100594, 1.757359313),
      ("Clamp-Cosine (c=4.0)", 8.296100594, 4.0),
      ("Clamp-Cosine (c=2.0)", 8.296100594, 2.0),
      ("Clamp-Cosine (c=1.0)", 8.296100594, 1.757359313),
      ("Sine", 3.703899406, 10.242640687),
      ("Interval", 6.0, 1.0),
      ("PCS (s=0.01)", 5.999674504, 5.997194433),
      ("PCS (s=0.1)", 5.968846397, 5.755320533),
      ("PCS (s=1.0)", 4.148050297, 0.878679656),
      ("PCS (s=2.0)", 1.989330440, 0.057644159),
      ("PCS (s=4.0)", 0.338141608, 0.000225894))
    for (name, third, sixth) <- expected do
      val values = VulkanIncantations.schedule(ujson.Obj("cfg_interval_schedule" -> name), 8, 6, 8)
      assert(math.abs(values(4) - third) < 1e-6, name)
      assert(math.abs(values(7) - sixth) < 1e-6, name)
  }
