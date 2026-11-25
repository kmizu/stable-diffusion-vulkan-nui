package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class PostDiffSuite extends munit.FunSuite:
  private val plugin = new PostDiffPlugin
  private def enabled(extra: (String, String)*) =
    Values(Values.defaults(plugin.controls).map ++ Map("enabled" -> "true") ++ extra)

  test("defaults are off, registered once before FastMode, and need engine feature postdiff"):
    assert(!plugin.isActive(Values.defaults(plugin.controls)))
    assertEquals(plugin.requiresEngineFeatures, Seq("postdiff"))
    val ids = StandardPlugins.alwaysOn.map(_.id)
    assertEquals(ids.count(_.==(plugin.id)), 1)
    assert(ids.indexOf(plugin.id) < ids.indexOf(FastModePlugin.Id))
    assert(plugin.process(GenerationParams(), enabled()).hooks.contains("postdiff"))

  test("hook fields, infotext round trip, and validation"):
    val params = plugin.process(GenerationParams(), enabled("scale" -> "0.65", "portion" -> "0.35"))
    assertEqualsDouble(params.hooks("postdiff")("scale").num, 0.65, 1e-9)
    assertEqualsDouble(params.hooks("postdiff")("portion").num, 0.35, 1e-9)
    val fields = Infotext.parse(Infotext.write(params, ImageFacts(seed = 4))).fields
    val back = PostDiffPlugin.replay(fields).get
    assertEquals(back.str("enabled"), "true")
    assertEqualsDouble(back.double("scale", 0), 0.65, 1e-9)
    assertEqualsDouble(back.double("portion", 0), 0.35, 1e-9)
    assertEquals(PostDiffPlugin.replay(Map("Steps" -> "20")).get.str("enabled"), "false")
    assertEquals(PostDiffPlugin.replay(Map(PostDiffPlugin.ScaleKey -> "0.1")), None)
    assertEquals(PostDiffPlugin.replay(Map(PostDiffPlugin.ScaleKey -> "0.5", PostDiffPlugin.PortionKey -> "0.95")), None)
