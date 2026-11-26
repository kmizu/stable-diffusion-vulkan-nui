package sdfx.vulkan

import sdfx.core.protocol.EngineError
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
