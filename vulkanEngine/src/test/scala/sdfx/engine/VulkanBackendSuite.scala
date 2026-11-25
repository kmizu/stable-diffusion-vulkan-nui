package sdfx.engine

import java.nio.file.Files
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, Settings}

final class VulkanBackendSuite extends munit.FunSuite:
  test("settings and hello use compatible optional snake_case fields") {
    import sdfx.core.json.SnakePickle
    import sdfx.core.protocol.{EngineHello, Json}
    assertEquals(SnakePickle.read[Settings]("{}").vulkanDevice, None)
    val saved = Settings(engineBackend = "vulkan", vulkanDevice = Some("Vulkan1"))
    val json = ujson.read(SnakePickle.write(saved))
    assertEquals(json("vulkan_device").str, "Vulkan1")
    assertEquals(SnakePickle.read[Settings](json).vulkanDevice, Some("Vulkan1"))
    val hello = ujson.Obj("protocol" -> 1, "engine_version" -> "test")
    assertEquals(Json.fromJs[EngineHello](hello).vulkanDevices, Nil)
    hello("vulkan_devices") = ujson.Arr("Vulkan0: Adapter A", "Vulkan1: Adapter B")
    assertEquals(Json.fromJs[EngineHello](hello).vulkanDevices, Seq("Vulkan0: Adapter A", "Vulkan1: Adapter B"))
  }

  test("Vulkan-only discovery uses the bundled worker even with a saved legacy backend") {
    val root = Files.createTempDirectory("vulkan package ")
    try
      val runtime = root.resolve("vulkan")
      Files.createDirectories(runtime.resolve("jre/bin"))
      Files.createFile(runtime.resolve("jre/bin/java.exe"))
      Files.createFile(runtime.resolve("vulkan-worker.jar"))
      Files.createDirectory(runtime.resolve("bin"))
      val home = AppHome(root)
      assertEquals(EngineSelection.availableBackends, Vector("vulkan"))
      val legacy = Settings(dtype = "fp32", vramMode = "sequential_offload", vaeTiling = true, fp8Storage = true)
      assertEquals(EngineBackend.loadRequest("checkpoint", legacy), sdfx.core.protocol.LoadModelRequest("checkpoint", "fp32"))
      for saved <- Seq("auto", "vulkan", "python", "scala-torch") do
        val settings = Settings(engineBackend = saved, vulkanDevice = Some("Vulkan0"))
        val env = Map("SDFX_VULKAN_DEVICE" -> "1", "Path" -> "inherited")
        assert(EngineSelection.usesNativeBackend(settings, env))
        assert(!EngineSetup.needsSetup(home, settings, env))
        val launch = EngineBackend.launch(home, settings, false, env)
        assertEquals(launch.command, Seq(runtime.resolve("jre/bin/java.exe").toString,
          "--enable-native-access=ALL-UNNAMED", "-jar", runtime.resolve("vulkan-worker.jar").toString))
        assertEquals(launch.workDir, root)
        assertEquals(launch.env("SDFX_VULKAN_DEVICE"), "1")
        assertEquals(EngineBackend.launch(home, settings, false, Map.empty).env("SDFX_VULKAN_DEVICE"), "Vulkan0")
        assertEquals(launch.env("SDFX_VULKAN_LIBRARY_DIR"), runtime.resolve("bin").toString)
        assertEquals(launch.env("PATH"), runtime.resolve("bin").toString + java.io.File.pathSeparator + "inherited")
      intercept[IllegalArgumentException](EngineBackend.launch(home, Settings(), false, Map("SDFX_ENGINE_BACKEND" -> "python")))
      val sourceHome = AppHome(Files.createDirectory(root.resolve("source-home")))
      val dry = EngineBackend.launch(sourceHome, Settings(), true, Map.empty)
      assertEquals(dry.command.last, "sdfx.engine.JvmDryRunWorker")
      assert(!dry.command.contains("--enable-native-access=ALL-UNNAMED"))
      assert(!dry.env.contains("SDFX_VULKAN_LIBRARY_DIR"))
    finally
      val paths = Files.walk(root)
      try paths.iterator.asScala.toVector.reverse.foreach(Files.delete)
      finally paths.close()
  }
