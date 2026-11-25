package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class CfgCacheSuite extends munit.FunSuite:
  private val plugin = new CfgCachePlugin
  private def enabled(extra: (String, String)*) =
    Values(Values.defaults(plugin.controls).map ++ Map("enabled" -> "true") ++ extra)

  test("defaults are off, registered once before FastMode, and need engine feature cfg_cache"):
    assert(!plugin.isActive(Values.defaults(plugin.controls)))
    assertEquals(plugin.requiresEngineFeatures, Seq("cfg_cache"))
    val ids = StandardPlugins.alwaysOn.map(_.id)
    assertEquals(ids.count(_.==(plugin.id)), 1)
    assert(ids.indexOf(plugin.id) < ids.indexOf(FastModePlugin.Id))
    assert(plugin.process(GenerationParams(), enabled()).hooks.contains("cfg_cache"))

  test("hook fields, infotext keys, and PNG round trip"):
    val params = plugin.process(GenerationParams(), enabled(
      "start" -> "0.3", "interval" -> "4", "adaptive_threshold" -> "0.94", "hf_weight" -> "0.15"))
    val h = params.hooks("cfg_cache")
    assertEqualsDouble(h("start").num, 0.3, 1e-9)
    assertEquals(h("interval").num, 4.0)
    assertEqualsDouble(h("adaptive_threshold").num, 0.94, 1e-9)
    assertEqualsDouble(h("hf_weight").num, 0.15, 1e-9)
    val fields = Infotext.parse(Infotext.write(params, ImageFacts(seed = 4))).fields
    val back = CfgCachePlugin.replay(fields).get
    assertEquals(back.str("enabled"), "true")
    assertEquals(back.int("interval"), 4)
    assertEqualsDouble(back.double("start", 0), 0.3, 1e-9)
    assertEqualsDouble(back.double("adaptive_threshold", 0), 0.94, 1e-9)
    assertEqualsDouble(back.double("hf_weight", 0), 0.15, 1e-9)

  test("optional adaptive_threshold and hf_weight are omitted at zero and bad replay values fail"):
    val plain = plugin.process(GenerationParams(), enabled())
    assert(!plain.hooks("cfg_cache").obj.contains("adaptive_threshold"))
    assert(!plain.hooks("cfg_cache").obj.contains("hf_weight"))
    assertEquals(CfgCachePlugin.replay(Map("Steps" -> "20")).get.str("enabled"), "false")
    assertEquals(CfgCachePlugin.replay(Map(CfgCachePlugin.IntervalKey -> "1")), None)
    assertEquals(CfgCachePlugin.replay(Map(CfgCachePlugin.IntervalKey -> "21")), None)
    assertEquals(CfgCachePlugin.replay(Map(CfgCachePlugin.IntervalKey -> "3", CfgCachePlugin.StartKey -> "1.2")), None)
    assertEquals(CfgCachePlugin.replay(Map(CfgCachePlugin.IntervalKey -> "3", CfgCachePlugin.HfWeightKey -> "0.8")), None)
