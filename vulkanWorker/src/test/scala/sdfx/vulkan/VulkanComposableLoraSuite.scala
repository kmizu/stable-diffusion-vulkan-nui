package sdfx.vulkan

import sdfx.core.model.LoraRef

class VulkanComposableLoraSuite extends munit.FunSuite:
  test("routes repeated tags and adds manual component weights only to their own component") {
    val plan = VulkanComposableLora.plan(
      "fox <lora:fur:0.2:0> AND owl <lora:fur:0.8:0.6> <lyco:wings:0:1>:0.5", "blur",
      Some(Seq(LoraRef("shared", .4, Some(.1)))), Nil,
      ujson.Obj("uc_text_model_encoder" -> true))
    assertEquals(plan.positive.map(_.text), Vector("fox", "owl"))
    assertEquals(plan.positive.map(_.coefficient), Vector(1.0, .5))
    assertEquals(plan.positive.head.textAdapters, Vector("fur" -> .2, "shared" -> .4))
    assertEquals(plan.positive.head.unetAdapters, Vector("shared" -> .1))
    assertEquals(plan.positive.last.unetAdapters, Vector("fur" -> .6, "shared" -> .1, "wings" -> 1.0))
    assertEquals(plan.negative.textAdapters, Vector("fur" -> .8, "shared" -> .4))
    assertEquals(plan.negative.unetAdapters, Vector.empty)
  }

  test("single prompts enable both negative components unless explicitly excluded") {
    val plan = VulkanComposableLora.plan("fox <lora:fur:1> #decrease", "blur", Some(Nil), Nil, ujson.Obj())
    assertEquals(plan.positive.head.text, "fox")
    assertEquals(plan.negative.textAdapters, Vector("fur" -> 1.0))
    assertEquals(plan.negative.unetAdapters, Vector("fur" -> 1.0))
  }

  test("known one-component files skip the other component") {
    assertEquals(VulkanComposableLora.components(Vector("lora_unet_a.lora_down.weight")), (false, true))
    assertEquals(VulkanComposableLora.components(Vector("lora_te2_a.lora_up.weight")), (true, false))
    assertEquals(VulkanComposableLora.components(Vector("other_format_a.weight")), (true, true))
  }

  test("legacy requests use the discovered ratio for every U-Net branch") {
    val plan = VulkanComposableLora.plan("fox <lora:fur:.8> AND owl <lora:fur:.2>", "blur", None,
      Seq(LoraRef("fur", .8, Some(.4))), ujson.Obj())
    assertEquals(plan.positive.map(_.unetAdapters), Vector(Vector("fur" -> .4), Vector("fur" -> .1)))
  }

  test("a single ordinary prompt without adapters retains the regular routing") {
    import sdfx.core.model.GenerationParams
    import sdfx.core.protocol.GenerationRequest
    assert(!VulkanComposableLora.needsGraph(GenerationRequest(GenerationParams(prompt = "fox:0.5"), "outputs")))
    assert(VulkanComposableLora.needsGraph(GenerationRequest(GenerationParams(prompt = "fox AND owl"), "outputs")))
  }
