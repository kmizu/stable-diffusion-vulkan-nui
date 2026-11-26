package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanAgentAttentionSuite extends munit.FunSuite:
  test("preserves both merge settings and the sampler step switch") {
    val cfg = VulkanAgentAttention.parse(ujson.Obj("sp_step" -> 3, "use_sp" -> true,
      "sx" -> 4, "ratio" -> 0.4, "sp_sx" -> 2, "sp_agent_ratio" -> 0.8, "use_fp32" -> true))
    val args = cfg.args(123L, hires = false)
    assert(args.contains("agent_switch = 3"))
    assert(args.contains("agent_sx = 4"))
    assert(args.contains("agent_second_sx = 2"))
    assert(args.contains("agent_second_ratio_agents = 0.8"))
    assert(args.contains("agent_fp32 = true"))
    assertEquals(cfg.args(123L, hires = true), "")
  }

  test("honors Hires-only and validates only active stage settings") {
    val cfg = VulkanAgentAttention.parse(ujson.Obj("hires_fix_only" -> true, "sp_step" -> 0,
      "sx" -> 0, "ratio" -> 0, "use_sp" -> true))
    assertEquals(cfg.args(1L, hires = false), "")
    assert(!cfg.applicable("sd15", hiresEnabled = false))
    assert(cfg.applicable("sd15", hiresEnabled = true))
    assert(!cfg.applicable("sdxl", hiresEnabled = true))
    assert(!VulkanAgentAttention.parse(ujson.Obj("sp_step" -> 0)).applicable("sd15", hiresEnabled = false))
    assert(cfg.args(1L, hires = true).contains("agent_second_sx = 2"))
    intercept[EngineError](VulkanAgentAttention.parse(ujson.Obj("sx" -> 0)))
    intercept[EngineError](VulkanAgentAttention.parse(ujson.Obj("sp_step" -> 0.5)))
    intercept[EngineError](VulkanAgentAttention.parse(ujson.Obj("agent_ratio" -> 1.1)))
  }
