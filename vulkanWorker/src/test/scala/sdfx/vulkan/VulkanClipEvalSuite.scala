package sdfx.vulkan

import munit.FunSuite
import sdfx.core.protocol.EngineError

class VulkanClipEvalSuite extends FunSuite:
  test("Clip expressions preserve Python precedence and chained comparisons") {
    val expression = VulkanClipEval.compile("-2 ** 2 + 2 ** -2 if 1 <= t < 3 and not d else 7")
    assertEquals(expression(0, 2, 0), -3.75f)
    assertEquals(expression(0, 3, 0), 7f)
    assertEquals(expression(0, 2, 1), 7f)
  }

  test("Clip expressions resolve current values and zero-based indices") {
    val expression = VulkanClipEval.compile("torch.where(t % 3 == 0, clamp(i + o * .5,-2,2), d // 2)")
    assertEquals(expression(1, 3, 0), 1.5f)
    assertEquals(expression(2, 3, 0), 2f)
    assertEquals(expression(0, 2, 7), 3f)
  }

  test("Clip expressions refuse names outside the numeric grammar") {
    val error = intercept[EngineError](VulkanClipEval.compile("torch.save(i)"))
    assertEquals(error.code, "invalid_params")
  }
