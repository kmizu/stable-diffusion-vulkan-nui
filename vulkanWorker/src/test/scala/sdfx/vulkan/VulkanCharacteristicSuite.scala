package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanCharacteristicSuite extends munit.FunSuite:
  test("rounds evaluation windows to even and clamps them to the pass length") {
    val settings = VulkanCharacteristic.parse(ujson.Obj("start_step" -> .25, "stop_step" -> .75))
    assertEquals(settings.window(10), (2, 8))
    assertEquals(VulkanCharacteristic.parse(ujson.Obj("start_step" -> -2, "stop_step" -> 2)).window(7), (0, 7))
    assertEquals(settings.args(10, 1), "")
    assert(settings.args(10, 6).contains("start = 2, stop = 8"))
  }

  test("preserves solver controls and rejects invalid legacy values") {
    val settings = VulkanCharacteristic.parse(ujson.Obj("noise_base" -> 3, "aa_dim" -> "2", "res" -> -3))
    assert(settings.args(8, 7).contains("bins = 3, memory = 2"))
    intercept[EngineError](VulkanCharacteristic.parse(ujson.Obj("reg_w" -> 0)))
    intercept[EngineError](VulkanCharacteristic.parse(ujson.Obj("ite" -> 1.5)))
    intercept[EngineError](settings.args(8, .8))
  }

  test("built-in pixel Hires keeps float pixels through the VAE round trip") {
    import sdfx.core.model.{GenerationParams, HiresParams}
    import sdfx.core.protocol.GenerationRequest
    val params = GenerationParams(cfgScale = 10, hires = HiresParams(enabled = true, upscaler = "Lanczos"))
    val graph = VulkanHiresGraph.refinement(GenerationRequest(params, "outputs"), "test", Nil, 3,
      "unet", "text", "", "Vulkan0", customNoise = false, tiled = "",
      characteristic = Some(VulkanCharacteristic.parse(ujson.Obj())))
    assert(graph.exists(line => line.startsWith("hr_pixels =") && line.contains("dtype = \"f32\"")))
    assert(graph.exists(line => line.startsWith("hr_base =") && line.contains("vae_encode(vae, hr_resized")))
  }
