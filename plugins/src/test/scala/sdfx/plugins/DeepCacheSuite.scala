package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class DeepCacheSuite extends munit.FunSuite:
  private val plugin = new DeepCachePlugin
  private def enabled(extra: (String, String)*) =
    Values(Values.defaults(plugin.controls).map ++ Map("enabled" -> "true") ++ extra)

  test("defaults are off, registered once, and need the engine feature"):
    assert(!plugin.isActive(Values.defaults(plugin.controls)))
    assertEquals(plugin.requiresEngineFeatures, Seq("deepcache"))
    assertEquals(StandardPlugins.alwaysOn.count(_.id == plugin.id), 1)
    assert(plugin.process(GenerationParams(), enabled()).hooks.contains("deepcache"))

  test("hook fields and infotext keys"):
    val params = plugin.process(GenerationParams(), enabled("interval" -> "5", "branch" -> "1", "full_first" -> "8"))
    assertEquals(params.hooks("deepcache")("interval").num, 5.0)
    assertEquals(params.hooks("deepcache")("branch").num, 1.0)
    assertEquals(params.extra, Map("DeepCache interval" -> "5", "DeepCache branch" -> "1", "DeepCache full first" -> "8"))

  test("PNG infotext round trip restores the controls"):
    val params = plugin.process(GenerationParams(), enabled("interval" -> "4", "branch" -> "2", "full_first" -> "7"))
    val fields = Infotext.parse(Infotext.write(params, ImageFacts(seed = 4))).fields
    val back = DeepCachePlugin.replay(fields).get
    assertEquals(back.str("enabled"), "true")
    assertEquals(back.int("interval"), 4)
    assertEquals(back.int("branch"), 2)
    assertEquals(back.int("full_first"), 7)

  test("replay switches off without the key and rejects bad values"):
    assertEquals(DeepCachePlugin.replay(Map("Steps" -> "20")).get.str("enabled"), "false")
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "1")), None)
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "11")), None)
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "3", "DeepCache branch" -> "4")), None)
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "3", "DeepCache order" -> "-1")), None)
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "3", "DeepCache order" -> "3")), None)
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "3", "DeepCache contraction" -> "0.05")), None)
    assertEquals(DeepCachePlugin.replay(Map("DeepCache interval" -> "3", "DeepCache contraction" -> "1.2")), None)

  test("TaylorSeer order is omitted when 0 and round-trips through infotext when > 0"):
    val zero = plugin.process(GenerationParams(), enabled("order" -> "0"))
    assert(!zero.hooks("deepcache").obj.contains("order"))
    assert(!zero.extra.contains(DeepCachePlugin.OrderKey))
    assert(!zero.hooks("deepcache").obj.contains("contraction"))
    val two = plugin.process(GenerationParams(), enabled("interval" -> "4", "order" -> "2", "contraction" -> "0.7"))
    assertEquals(two.hooks("deepcache")("order").num, 2.0)
    assertEqualsDouble(two.hooks("deepcache")("contraction").num, 0.7, 1e-9)
    assertEquals(two.extra(DeepCachePlugin.OrderKey), "2")
    assertEquals(two.extra(DeepCachePlugin.ContractionKey), "0.7")
    val fields = Infotext.parse(Infotext.write(two, ImageFacts(seed = 4))).fields
    val back = DeepCachePlugin.replay(fields).get
    assertEquals(back.int("order"), 2)
    assertEqualsDouble(back.double("contraction", 1.0), 0.7, 1e-9)
