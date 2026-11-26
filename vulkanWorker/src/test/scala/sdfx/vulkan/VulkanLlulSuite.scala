package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanLlulSuite extends munit.FunSuite:
  test("uses the legacy centered region and output-only settings by default") {
    val cfg = VulkanLlul.parse(ujson.Obj())
    assertEquals(cfg.origin(512, 768), (0.25, 0.25))
    assertEquals(cfg.layerKinds, 16)
    assertEquals(cfg.filters, Vector("OUT"))
    assert(cfg.args(512, 768, "", 20).contains("llul_start = 5, llul_end = 20"))
  }

  test("skips unreachable windows and filters while leaving the output layer independent") {
    assert(!VulkanLlul.parse(ujson.Obj("start_steps" -> 100)).canApply(20))
    val resblock = ujson.Arr("resblock")
    assert(!VulkanLlul.parse(ujson.Obj("apply_to" -> resblock, "layers" -> "no-such-layer")).canApply(20))
    assert(VulkanLlul.parse(ujson.Obj("apply_to" -> resblock, "layers" -> "OUT03")).canApply(20))
    assert(VulkanLlul.parse(ujson.Obj("layers" -> "no-such-layer")).canApply(20))
  }

  test("preserves masks, advanced layer selection and independent AA modes") {
    val cfg = VulkanLlul.parse(ujson.Obj("mask" -> "gray.png", "intp" -> "slerp", "down_aa" -> true,
      "up" -> "bicubic", "up_aa" -> true, "apply_to" -> ujson.Arr("resblock", "s. attn.", "x. attn."),
      "layers" -> "M00, IN04", "x" -> 32, "y" -> 64))
    assertEquals(cfg.origin(512, 512), (0.0625, 0.125))
    assertEquals(cfg.layerKinds, 13)
    assertEquals(cfg.filters, Vector("M00", "IN04"))
    assert(!cfg.spherical)
    intercept[EngineError](VulkanLlul.parse(ujson.Obj("up" -> "nearest", "up_aa" -> true)))
    intercept[EngineError](cfg.origin(64, 512))
  }
