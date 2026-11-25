package sdfx.vulkan

import sdfx.core.model.GenerationParams
import sdfx.core.protocol.{EngineError, GenerationRequest, Json}

class VulkanWorkerSuite extends munit.FunSuite:
  test("GUI defaults and allocated variation seed select the default native sampler") {
    val request = GenerationRequest(GenerationParams(subseed = 1234), "outputs", vaeRecoveryDir = "outputs")
    val parsed = VulkanWorker.generation(Json.toJs(request))
    assertEquals(parsed.request.params.subseed, 1234L)
    assertEquals(parsed.scheduler, "Karras")
  }

  test("an enabled engine hook is rejected by name") {
    val request = GenerationRequest(GenerationParams(hooks = Map("freeu" -> ujson.Obj("enabled" -> true))), "outputs")
    val error = intercept[EngineError](VulkanWorker.generation(Json.toJs(request)))
    assertEquals(error.code, "unsupported")
    assert(error.message.contains("freeu"))
  }

  test("registry uses final extension and preserves names of nested checkpoints") {
    val root = java.nio.file.Files.createTempDirectory("vulkan-catalog")
    val folder = java.nio.file.Files.createDirectory(root.resolve("anime"))
    val model = java.nio.file.Files.createFile(folder.resolve("animagine-xl-3.1.SAFETENSORS"))
    try
      val files = VulkanProtocol.models(ujson.Obj("dirs" -> ujson.Arr(root.toString, folder.toString)))("models").arr
      assertEquals(files.map(_("name").str).toList, List("anime/animagine-xl-3.1"))
      assertEquals(files.head("path").str, model.toString)
    finally
      java.nio.file.Files.delete(model)
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
