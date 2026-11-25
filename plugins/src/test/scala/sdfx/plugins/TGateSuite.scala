package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class TGateSuite extends munit.FunSuite:
  private val plugin = new TGatePlugin
  private def enabled(extra: (String, String)*) =
    Values(Values.defaults(plugin.controls).map ++ Map("enabled" -> "true") ++ extra)

  test("defaults are off, registered once, and need the tgate engine feature"):
    assert(!plugin.isActive(Values.defaults(plugin.controls)))
    assertEquals(plugin.requiresEngineFeatures, Seq("tgate"))
    assertEquals(StandardPlugins.alwaysOn.count(_.id == plugin.id), 1)
    assert(plugin.process(GenerationParams(), enabled()).hooks.contains("tgate"))

  test("hook fields and infotext keys"):
    val params = plugin.process(GenerationParams(), enabled("gate_step" -> "10"))
    assertEquals(params.hooks("tgate")("gate_step").num, 10.0)
    assertEquals(params.extra, Map("T-GATE gate step" -> "10"))

  test("PNG infotext round trip restores the controls"):
    val params = plugin.process(GenerationParams(), enabled("gate_step" -> "7"))
    val fields = Infotext.parse(Infotext.write(params, ImageFacts(seed = 4))).fields
    val back = TGatePlugin.replay(fields).get
    assertEquals(back.str("enabled"), "true")
    assertEquals(back.int("gate_step"), 7)

  test("replay switches off without the key and rejects out-of-range values"):
    assertEquals(TGatePlugin.replay(Map("Steps" -> "20")).get.str("enabled"), "false")
    assertEquals(TGatePlugin.replay(Map("T-GATE gate step" -> "0")), None)
    assertEquals(TGatePlugin.replay(Map("T-GATE gate step" -> "51")), None)
