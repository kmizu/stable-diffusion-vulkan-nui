package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanReferenceSuite extends munit.FunSuite:
  test("reference settings preserve the legacy controls and predictor branch") {
    val options = VulkanReference.parse(ujson.Obj("image" -> "reference.png",
      "style_fidelity" -> 0.75, "attention_auto_machine_weight" -> 1.5))
    assertEquals(options.image, "reference.png")
    val graph = options.prepare("ref", "")
    assert(graph.head.startsWith("ref_rng = make_rng"))
    assert(graph(1).contains("rng = ref_rng, posterior"))
    assertEquals(options.args("ref", positive = true),
      ", reference_latent = ref_latent, reference_noise = ref_noise, reference_weight = 1.5, reference_style = 0.75, reference_positive = true")
    assert(options.args("ref", positive = false).endsWith("reference_positive = false"))
  }

  test("disabled reference weight bypasses attention and invalid settings fail") {
    val off = VulkanReference.parse(ujson.Obj("image" -> "reference.png", "attention_auto_machine_weight" -> 0))
    assertEquals(off.args("ref", positive = true), "")
    intercept[EngineError](VulkanReference.parse(ujson.Obj("style_fidelity" -> 1)))
    intercept[EngineError](VulkanReference.parse(ujson.Obj("image" -> "reference.png", "style_fidelity" -> 1.1)))
  }
