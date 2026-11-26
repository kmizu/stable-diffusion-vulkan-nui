package sdfx.vulkan

import sdfx.core.protocol.EngineError

class VulkanTGateSuite extends munit.FunSuite:
  test("matches the T-GATE control default and bounds") {
    assertEquals(VulkanTGate.gate(ujson.Obj()), 8)
    assertEquals(VulkanTGate.gate(ujson.Obj("gate_step" -> 1)), 1)
    assertEquals(VulkanTGate.gate(ujson.Obj("gate_step" -> 50)), 50)
    assertEquals(VulkanTGate.gate(ujson.Obj("gate_step" -> 1000)), 1000)
    for value <- Seq(0.0, 1.5, 1001.0) do
      intercept[EngineError](VulkanTGate.gate(ujson.Obj("gate_step" -> value)))
  }
