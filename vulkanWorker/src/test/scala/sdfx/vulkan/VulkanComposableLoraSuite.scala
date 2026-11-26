package sdfx.vulkan

import sdfx.core.model.LoraRef

class VulkanComposableLoraSuite extends munit.FunSuite:
  test("math commands use legacy helpers, window progress and the preparation clock") {
    val sine = VulkanLoraSteps.parse("""fox [<lora:A:2> #cmd\(sin\(life\)\):10]""")
    assertEquals(sine.prompt, "fox")
    assertEqualsDouble(sine.weights(15, 20).head._2, 1.0, 1e-12)
    assertEquals(sine.weights(-1, 20).head._2, 1.0)
    val warmup = VulkanLoraSteps.parse("""[<lora:A:2> \u0023cmd\(warmup\(0.5\)\):10]""")
    assertEqualsDouble(warmup.weights(12, 20).head._2, .4, 1e-12)
    val cubeRoot = VulkanLoraSteps.parse("[<lora:A:1> #cmd(cbrt(-8)):10]")
    assertEqualsDouble(cubeRoot.weights(15, 20).head._2, -2.0, 1e-12)
  }

  test("math command comparisons leave outer AND routing intact and recognize the negative branch") {
    val plan = VulkanComposableLora.plan(
      "fox [<lora:fur:1> #cmd(.4 if step > 2 and not is_negative else 0)] AND owl",
      "blur", Some(Nil), Nil, ujson.Obj("with_step" -> true), steps = 4)
    assertEquals(plan.positive.map(_.text), Vector("fox", "owl"))
    assertEquals(plan.positive.head.unetSchedule,
      Vector(Vector.empty, Vector.empty, Vector("fur" -> .4), Vector("fur" -> .4)))
    val single = VulkanComposableLora.plan(
      "fox [<lora:fur:1> #cmd(.3 if is_negative else .5)]", "blur",
      Some(Nil), Nil, ujson.Obj("with_step" -> true), steps = 2)
    assertEquals(single.negative.unetSchedule, Vector.fill(2)(Vector("fur" -> .3)))
    assertEquals(single.positive.head.unetSchedule, Vector.fill(2)(Vector("fur" -> .5)))
  }

  test("non-finite controller results fail before native inference") {
    val command = VulkanLoraSteps.parse("[<lora:A:1> #cmd(1e309):10]")
    val error = intercept[sdfx.core.protocol.EngineError](command.weights(15, 20))
    assert(error.getMessage.contains("finite"), error.getMessage)
  }

  test("weight in a command uses the selected U-Net component or the legacy tag before ratio") {
    val prompt = "fox [<lora:fur:.2:.8> #cmd(weight)]"
    val config = ujson.Obj("with_step" -> true)
    val modern = VulkanComposableLora.plan(prompt, "", Some(Nil), Nil, config)
    assertEqualsDouble(modern.positive.head.unetSchedule.head.head._2, .64, 1e-12)
    val legacy = VulkanComposableLora.plan(prompt, "", None, Seq(LoraRef("fur", .2, Some(.8))), config)
    assertEqualsDouble(legacy.positive.head.unetSchedule.head.head._2, .16, 1e-12)
  }

  test("scheduled predictors use one runtime adapter set and put each branch weight in binding order") {
    import java.nio.file.Files
    import sdfx.core.model.GenerationParams
    import sdfx.core.protocol.GenerationRequest

    VulkanClipIo.withScratch("clora-schedule-test-") { directory =>
      val a = Files.createFile(directory.resolve("a.pt"))
      val z = Files.createFile(directory.resolve("z.pt"))
      val prompt = "fox [<lora:z:0:.5> #cmd(step / steps)] AND owl <lora:a:0:.8>"
      val request = GenerationRequest(GenerationParams(prompt = prompt), directory.toString,
        loraPaths = Map("a" -> a.toString, "z" -> z.toString))
      val plan = VulkanComposableLora.plan(prompt, "", Some(Nil), Nil, ujson.Obj("with_step" -> true), steps = 2)
      val lines = VulkanComposableLora.prepare(plan, request, "clora", "unet", "text", "encoding", "",
        supportsBlank = false, runtimeWeights = true)
      assertEquals(lines.filter(_.contains(" = apply_lora(")).size, 2)
      assert(lines.exists(line => line.startsWith("clora_neg = predictor") && line.contains("[0.0, 0.0]")))
      assert(lines.exists(line => line.startsWith("clora_branch0_stage0 = predictor") && line.contains("[0.0, 0.25]")))
      assert(lines.exists(line => line.startsWith("clora_branch0_stage1 = predictor") && line.contains("[0.0, 0.5]")))
      assert(lines.exists(line => line.startsWith("clora_branch1 = predictor") && line.contains("[0.8, 0.0]")))
    }
  }

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
    assertEquals(local.traceCandidates, Vector("fur"))
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

  test("weight trace keeps repeated evaluations, inactive weights, image indices, and passes") {
    import java.nio.file.{Files, Path}
    import javax.imageio.ImageIO
    import sdfx.core.protocol.GeneratedImage

    VulkanClipIo.withScratch("clora-trace-test-") { directory =>
      val run = new VulkanLoraTrace.Run(ujson.Obj())
      val prompt = "雪の森 \"fox\""
      def record(pass: String, step: Int, weight: Double): Unit =
        val label = VulkanLoraTrace.label(pass, "branch0", prompt, false, Seq("fur" -> weight))
        run.record(s"$step $label")
      record("base", 1, .25)
      record("base", 1, .75)
      record("base", 2, 0)
      record("hires", 1, .5)
      run.imageIndex = 1
      record("base", 1, 1)

      val first = GeneratedImage("first.png", seed = 17, width = 64, height = 64)
      val (plot, manifest) = run.finish(directory, first, None, _ => ())
      val trace = Path.of(manifest.path)
      assertEquals(plot.seed, 17L)
      assertEquals(ImageIO.read(Path.of(plot.path).toFile).getWidth, 1200)
      val rows = ujson.read(Files.readString(trace))("records").arr.toVector
      Files.delete(trace)
      Files.delete(trace.getParent)
      Files.delete(trace.getParent.getParent)
      assertEquals(rows.map(r => (r("image_index").num.toInt, r("pass_name").str, r("step").num.toInt)),
        Vector((0, "base", 1), (0, "base", 2), (0, "hires", 1), (1, "base", 1)))
      assertEquals(rows.map(_("mean_weight").num), Vector(.5, 0.0, .5, 1.0))
      assertEquals(rows.head("min_weight").num, .25)
      assertEquals(rows.head("max_weight").num, .75)
      assertEquals(rows.head("model_evaluations").num, 2.0)
      assertEquals(rows.head("prompt").str, prompt)
    }
  }
