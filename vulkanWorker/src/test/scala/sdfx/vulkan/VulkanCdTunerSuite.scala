package sdfx.vulkan

class VulkanCdTunerSuite extends munit.FunSuite:
  test("removes tuner directives before parsing extra networks") {
    assertEquals(VulkanCdTuner.cleanPrompt("fox <cdt:d1=6> <lora:fox:0.5> <cdtc:cols=1 2 3>"),
      "fox  <lora:fox:0.5> ")
  }
  test("inherits the base weights when Hires Stop Step is zero") {
    val options = VulkanCdTuner.parse(ujson.Obj("d1" -> 5, "d2" -> 3, "con1" -> 2,
      "hd1" -> -5, "st1" -> 6, "st2" -> 0, "hrs" -> true), sdxl = true)
    assertEquals(options.weights(1, baseSteps = 6), VulkanCdTuner.Weights(5, 3, 2, -1))
    assertEquals(options.weights(1, baseSteps = 7), VulkanCdTuner.Weights(0, 0, 0, 0))
    assertEquals(options.weights(4, baseSteps = 6), VulkanCdTuner.Weights(5, 3, 2, -1))
  }

  test("scales Hires detail by latent area and maps denoised color by model family") {
    val raw = ujson.Obj("d1" -> 5, "d2" -> 4, "hd1" -> -2, "hrs" -> true, "bri" -> 3, "col1" -> -3)
    val xl = VulkanCdTuner.parse(raw, sdxl = true)
    val sd = VulkanCdTuner.parse(raw, sdxl = false)
    assertEquals(xl.weights(4, baseSteps = 6), VulkanCdTuner.Weights(-8, 16, 0, -1))
    assertEquals(xl.shifts, Vector(0.0, -.06, 0.0, .06))
    sd.shifts.zip(Vector(0.0, -.12, -.04, .04)).foreach { (actual, expected) =>
      assertEqualsDouble(actual, expected, 1e-12)
    }
  }

  test("builds separate horizontal and vertical regions and preserves Color Map only Hires attenuation") {
    val base = ujson.Obj("split_ratio" -> "1,3", "colors" -> "0 5 0 0;0 0 5 0", "map_stop_step" -> 3)
    val horizontal = VulkanCdTuner.parse(base, sdxl = true).colorMap.get
    assertEquals(horizontal.regions.map(r => (r.left, r.top, r.right, r.bottom)),
      Vector((0.0, 0.0, 1.0, .25), (0.0, .25, 1.0, 1.0)))
    base("split_by") = "Vertical"
    val vertical = VulkanCdTuner.parse(base, sdxl = true).colorMap.get
    assertEquals(vertical.regions.map(r => (r.left, r.top, r.right, r.bottom)),
      Vector((0.0, 0.0, .25, 1.0), (.25, 0.0, 1.0, 1.0)))
    assertEquals(vertical.forHires(firstOnly = true).steps, 1)
    assertEqualsDouble(vertical.forHires(firstOnly = true).strength, .175, 1e-12)
  }
