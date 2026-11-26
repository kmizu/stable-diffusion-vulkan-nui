package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanDeepCacheSuite extends munit.FunSuite:
  test("reads the legacy controls and retains the TaylorSeer contraction") {
    assertEquals(VulkanDeepCache.parse(ujson.Obj()), VulkanDeepCache.Config(3, 0, 5, 0, 1))
    assertEquals(VulkanDeepCache.parse(ujson.Obj("interval" -> 2, "branch" -> 1, "full_first" -> 0,
      "order" -> 2, "contraction" -> 0.7)), VulkanDeepCache.Config(2, 1, 0, 2, 0.7))
    intercept[EngineError](VulkanDeepCache.parse(ujson.Obj("interval" -> 2.5)))
    intercept[EngineError](VulkanDeepCache.parse(ujson.Obj("contraction" -> 0)))
  }

  test("sends both CFG controls to a cache-enabled sampler") {
    val hooks = Map(
      "cfg_cache" -> ujson.Obj("interval" -> 4, "start" -> 0.25),
      "cfg_schedule" -> ujson.Obj("start_cfg" -> 6, "end_cfg" -> 1, "start" -> 0, "end" -> 1))
    val args = VulkanGraph.cacheCfgArguments(hooks, cfg = 6, steps = 4)
    assert(args.contains("schedule = ["))
    assert(args.endsWith(", cfg_cache_start = 0.25, cfg_cache_interval = 4"))
    intercept[EngineError](VulkanGuidance.cacheSchedule(Map("cfg_cache" -> ujson.Obj("interval" -> 2.5))))
    intercept[EngineError](VulkanGuidance.cacheSchedule(Map("cfg_cache" -> ujson.Obj("hf_weight" -> 0.1))))
  }
