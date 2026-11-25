package sdfx.mcp

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.model.GenerationParams
import sdfx.engine.{AppServices, GenerationListener}
import sdfx.plugins.{NsfwCheckerExtrasPlugin, NsfwCheckerPlugin, StandardPlugins}

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}

class NativeNsfwMcpSuite extends munit.FunSuite:
  test("Python-free scala-torch dry-run worker supports nsfw_checker generation hook and Extras Pixelate") {
    assume(sdfx.engine.EngineSelection.availableBackends.contains("scala-torch"), "scala-torch provider required")
    val tmp = Files.createTempDirectory("nudenet-mcp-native-")
    val appHome = AppHome(tmp)
    val nudenetDir = tmp.resolve("models/nudenet")
    Files.createDirectories(nudenetDir)
    val repoModel = Path.of("models/nudenet/best.onnx").toAbsolutePath.normalize()
    Files.copy(repoModel, nudenetDir.resolve("best.onnx"))

    val settings = Settings(engineBackend = "scala-torch")
    lazy val services: AppServices = new AppServices(appHome, settings, dryRun = true,
      builtins = StandardPlugins.all(() => services.registry.xyzAxes),
      persistSettings = false, installShutdownHook = false)
    try
      val hello = services.start()
      assertEquals(hello.python, "")
      assert(hello.features.contains("nsfw_checker"), "expected native nsfw_checker feature")

      // 1. Generation with nsfw_checker hook enabled (Pixelate filter) completes natively without Python
      val hookParams = new NsfwCheckerPlugin().process(
        GenerationParams(prompt = "portrait", width = 128, height = 128, steps = 2),
        sdfx.core.plugins.api.Values(
          sdfx.core.plugins.api.Values.defaults(new NsfwCheckerPlugin().controls).map ++
            Map("enabled" -> "true", "gen_filter_type" -> "Pixelate", "pixelation_factor" -> "5")
        )
      )
      val out = services.generator.forJob(GenerationListener.none).generate(hookParams)
      assertEquals(out.size, 1)
      assert(out.head.infotext.contains(NsfwCheckerPlugin.InfotextKey), "expected NudeNet infotext")

      // 2. Extras NSFW Checker with Pixelate + Entire image (always) applies native mosaic & produces mask
      val img = new BufferedImage(128, 128, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 128; x <- 0 until 128 do
        img.setRGB(x, y, (((x * 7) & 0xff) << 16) | (((y * 5) & 0xff) << 8) | ((x + y) & 0xff))

      val extrasPlugin = new NsfwCheckerExtrasPlugin
      val extrasVals = sdfx.core.plugins.api.Values(
        sdfx.core.plugins.api.Values.defaults(extrasPlugin.controls).map ++ Map(
          "checker" -> "NudeNet",
          "enable_nudenet" -> "false",
          "filter_type" -> "Pixelate",
          "mask_shape" -> "Entire image (always)",
          "pixelation_factor" -> "4",
          "save_mask" -> "true"
        )
      )
      val results = extrasPlugin.transform(img, GenerationParams(), extrasVals, services.generator.forJob(GenerationListener.none))
      assertEquals(results.size, 2)
      val censored: BufferedImage = results.head._1
      // Neighboring pixels inside a mosaic cell share the same color
      assertEquals(censored.getRGB(20, 20), censored.getRGB(21, 21), "mosaic block pixels must match")
      assertEquals(results(1)._2.last, "Output" -> "mask")
    finally
      services.shutdown()
      TestEnv.deleteTree(tmp)
  }
