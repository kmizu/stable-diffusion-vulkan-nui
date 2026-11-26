package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanTGate:
  val id = "tgate"

  def gate(config: Value): Int =
    val fields = config.obj
    if !fields.keySet.forall(Set("enabled", "gate_step")) then VulkanProtocol.invalid("Unknown T-GATE option")
    val step = fields.get("gate_step").fold(8.0)(_.numOpt.getOrElse(Double.NaN))
    if !step.isWhole || step < 1 || step > 1000 then VulkanProtocol.invalid("T-GATE gate_step must be an integer in 1..1000")
    step.toInt
