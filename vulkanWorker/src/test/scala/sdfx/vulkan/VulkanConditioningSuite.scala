package sdfx.vulkan

import sdfx.core.model.{ControlNetUnit, GenerationParams, LoraRef}
import sdfx.core.protocol.{EngineError, GenerationRequest, LoadModelRequest}

class VulkanConditioningSuite extends munit.FunSuite:
  test("map UI placement and precision to native context options") {
    val offload = LoadModelRequest("base", dtype = "bf16", vramMode = "model_offload", vaeFp16 = true, fp8Storage = true)
    val native = VulkanConditioning.options(offload, "Vulkan1")
    assertEquals(native.placement(), "cpu")
    assertEquals(native.weights(), "bf16")
    assert(native.tensorTypes().contains("=f8_e4m3"))
    assert(native.tensorTypes().contains("=f16"))
    assertEquals(VulkanConditioning.options(offload.copy(vramMode = "sequential_offload"), "Vulkan1").placement(), "disk")
    assertEquals(VulkanConditioning.options(offload.copy(vramMode = "auto"), "Vulkan1").placement(), "")
  }

  test("select embedding tokens from positive and negative prompts, including punctuation") {
    val files = Seq("foo.safetensors", "foo.bar.safetensors", "style(2).safetensors", "short.safetensors")
    val req = GenerationRequest(GenerationParams(prompt = "(Foo.Bar), style(2)", negativePrompt = "shorter"),
      "out", embeddingPaths = files)
    assertEquals(VulkanConditioning.activeEmbeddings(req), files.take(3))
  }

  test("reject unsupported LoRA multipliers and simultaneous ControlNet units") {
    val load = LoadModelRequest("base")
    val split = GenerationParams(loras = Seq(LoraRef("x", unet = Some(0.5))))
    val multiple = GenerationParams(controlnet = Seq(ControlNetUnit(), ControlNetUnit()))
    for params <- Seq(split, multiple) do
      val e = intercept[EngineError](VulkanConditioning.prepare(GenerationRequest(params, "out"), load, () => false))
      assertEquals(e.code, "unsupported")
  }
