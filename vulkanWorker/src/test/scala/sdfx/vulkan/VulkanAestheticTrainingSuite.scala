package sdfx.vulkan

import sdfx.core.protocol.EngineError
import sdfx.core.model.LoraRef
import sdfx.vulkan.tools.VulkanAestheticTraining

class VulkanAestheticTrainingSuite extends munit.FunSuite:
  test("training uses legacy defaults and rejects invalid values before loading a checkpoint") {
    val config = VulkanAestheticTraining.settings(ujson.Obj("embedding" -> "style", "lr" -> "0.0001"))
    assert(config.active)
    assertEquals(config.steps, 20)
    assertEquals(config.rate, .0001f)
    assert(!VulkanAestheticTraining.settings(ujson.Obj("steps" -> 0)).active)
    intercept[EngineError](VulkanAestheticTraining.settings(ujson.Obj("embedding" -> "style", "steps" -> 2.5)))
    intercept[EngineError](VulkanAestheticTraining.settings(ujson.Obj("embedding" -> "style", "embedding_sha256" -> "x")))
  }

  test("training chunks keep BOS, EOS padding, and BREAK boundaries while ignoring emphasis") {
    val ordinary = VulkanAestheticTraining.chunks("red fox BREAK white owl")
    val weighted = VulkanAestheticTraining.chunks("(red fox:1.2) BREAK white owl")
    assertEquals(weighted, ordinary)
    assertEquals(ordinary.map(_.size), Vector(77, 77))
    for chunk <- ordinary do
      assertEquals(chunk.head, 49406)
      assertEquals(chunk.last, 49407)
      assert(chunk.drop(chunk.indexOf(49407)).forall(_ == 49407))
  }

  test("aesthetic text selects its first CLIP chunk only when slerp changes the target") {
    val config = VulkanAestheticTraining.settings(ujson.Obj("embedding" -> "style", "slerp" -> true,
      "text" -> "red fox BREAK white owl", "slerp_angle" -> .35))
    val style = VulkanAestheticTraining.styleChunk(config).get
    assertEquals(style, VulkanAestheticTraining.chunks("red fox").head)
    assertEquals(VulkanAestheticTraining.styleChunk(config.copy(style = "red fox" + " BREAK owl" * 20)), Some(style))
    assertEquals(VulkanAestheticTraining.styleChunk(config.copy(angle = 0)), None)
    assertEquals(VulkanAestheticTraining.styleChunk(config.copy(style = "  ")), None)
    assertEquals(VulkanAestheticTraining.styleChunk(config.copy(slerp = false)), None)
  }

  test("personalization leaves a blank negative untouched and uses each refinement caption") {
    val config = VulkanAestheticTraining.settings(ujson.Obj("embedding" -> "style", "negative" -> true))
    val captions = Vector.newBuilder[(String, Seq[LoraRef])]
    val conditioning = VulkanAestheticTraining.Conditioning(config, sdxl = true, (caption, loras) => {
      captions += caption -> loras
      "trained"
    }, _ => true -> true)
    val lines = Vector.newBuilder[String]
    conditioning.encode("neg", "base", "   ", "width = 512", negative = true, Nil, lines)
    conditioning.encode("pos", "base", "red fox", "width = 512", negative = false, Nil, lines)
    val adapters = Seq(LoraRef("style", .7))
    conditioning.encode("hr_neg", "base", "blue sky", "width = 576", negative = true, adapters, lines)
    val program = lines.result()
    assertEquals(captions.result(), Vector("blue sky" -> adapters))
    assertEquals(program.head, "neg = encode_prompt(base, \"\", width = 512)")
    assert(program.contains("hr_neg_trained = encode_prompt(trained, \"blue sky\", width = 576)"))
    assert(program.last.contains("vector = hr_neg_base.vector"))
  }
