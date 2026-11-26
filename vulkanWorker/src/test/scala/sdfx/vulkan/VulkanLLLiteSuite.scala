package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanLLLiteSuite extends munit.FunSuite:
  test("preserves the selected model, fallback hint, strength and timestep window") {
    val config = VulkanLLLite.parse(ujson.Obj("model" -> "canny", "strength" -> 0.7,
      "start_percent" -> 0.2, "end_percent" -> 0.8))
    assertEquals(config.model, "canny")
    assertEquals(config.source(Some("init.png")), "init.png")
    assertEquals(config.copy(image = Some("map.png")).source(Some("init.png")), "map.png")
    assertEquals(config.args("canny.safetensors", "ctrl"),
      """, lllite_path = "canny.safetensors", lllite_hint = ctrl, lllite_strength = 0.7, lllite_start = 0.2, lllite_end = 0.8""")
  }

  test("rejects invalid ranges and missing required inputs") {
    intercept[EngineError](VulkanLLLite.parse(ujson.Obj()))
    assert(!VulkanLLLite.parse(ujson.Obj("model" -> "x", "start_percent" -> 0.9, "end_percent" -> 0.1)).active)
    assert(!VulkanLLLite.parse(ujson.Obj("model" -> "x", "strength" -> 0)).active)
    intercept[EngineError](VulkanLLLite.parse(ujson.Obj("model" -> "x", "start_percent" -> 1.1)))
    intercept[EngineError](VulkanLLLite.parse(ujson.Obj("model" -> "x")).source(None))
  }
