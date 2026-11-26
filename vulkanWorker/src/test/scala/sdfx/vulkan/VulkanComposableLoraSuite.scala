package sdfx.vulkan

import sdfx.core.model.LoraRef

class VulkanComposableLoraSuite extends munit.FunSuite:
  test("step controllers respect inclusive fractional windows, alternating branches, and ramps") {
    val plan = VulkanComposableLora.plan(
      "fox [[<lora:fur:1:1> #increment::.75]:.25] AND owl [<lora:a:1>|<lora:b:1>]", "",
      Some(Nil), Nil, ujson.Obj("with_step" -> true), steps = 4)
    assertEquals(plan.positive.map(_.text), Vector("fox", "owl"))
    assertEquals(plan.positive.head.textAdapters, Vector("fur" -> 1.0))
    assertEquals(plan.positive.head.unetSchedule, Vector(Vector.empty, Vector("fur" -> .5),
      Vector("fur" -> 1.0), Vector.empty))
    assertEquals(plan.positive(1).unetSchedule.map(_.head._1), Vector("b", "a", "b", "a"))
  }

  test("a Hires clock can begin after the base pass") {
    val plan = VulkanComposableLora.plan("fox [<lora:fur:1>:3]", "", Some(Nil), Nil,
      ujson.Obj("with_step" -> true), steps = 2, clockOffset = 2, clockTotal = 4)
    assertEquals(plan.positive.head.unetSchedule, Vector.fill(2)(Vector("fur" -> 1.0)))
    assertEquals(plan.negative.unetSchedule, plan.positive.head.unetSchedule)
    val local = VulkanComposableLora.plan("fox [<lora:fur:1>:3]", "", Some(Nil), Nil,
      ujson.Obj("with_step" -> true), steps = 2)
    assert(local.positive.head.unetSchedule.forall(_.isEmpty))
    val repeated = VulkanComposableLora.plan("fox [<lora:fur:.2><lora:fur:.8>:2]", "", Some(Nil), Nil,
      ujson.Obj("with_step" -> true), steps = 2)
    assertEquals(repeated.positive.head.textAdapters, Vector("fur" -> .8))
    val ordinaryGroup = VulkanLoraSteps.parse("fox [striped <lora:fur:.8>]")
    assertEquals(ordinaryGroup.prompt, "fox [striped]")
    assertEquals(ordinaryGroup.weights(1, 2), Vector.empty)
    val static = VulkanLoraSteps.parse("fox <lora:fur:.8>")
    assertEquals(static.weights(3, 2).map(_._1.name), Vector("fur"))
    val p = sdfx.core.model.GenerationParams(steps = 10, img2img =
      Some(sdfx.core.model.Img2ImgParams("source", denoisingStrength = .5)))
    assertEquals(VulkanComposableLora.clock(p, ujson.Obj(), actual = 6, hires = false), (0, 5))
  }

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
