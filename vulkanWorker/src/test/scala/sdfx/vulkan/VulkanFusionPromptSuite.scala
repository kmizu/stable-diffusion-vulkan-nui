package sdfx.vulkan

import munit.FunSuite
import sdfx.core.protocol.EngineError
import VulkanFusionPrompt.{Plan, schedule}

class VulkanFusionPromptSuite extends FunSuite:
  test("curve uses the zero-based sampling clock and retains text around each control") {
    val frames = schedule("a [left:right:,.5] scene", 8)
    val controls = Vector(Plan.Leaf("a left scene"), Plan.Leaf("a right scene"))
    assertEquals(frames(0), Plan.Blend("linear",0,controls))
    assertEquals(frames(2), Plan.Blend("linear",.5,controls))
    assertEquals(frames(4), Plan.Blend("linear",1,controls))
  }

  test("attention ramps use the surrounding edit interval") {
    val frames = schedule("[plain:(accent:1,3):.5]",8)
    assertEquals(frames(3), Plan.Leaf("plain"))
    assertEquals(frames(4), Plan.Leaf("(accent:1.0)"))
    assertEquals(frames(7), Plan.Leaf("(accent:3.0)"))
  }

  test("new curve coordinates become the outer interpolation over earlier prompt spans") {
    val frame = schedule("[a:b::mean] [c:d::mean]",4).head
    val first = Plan.Blend("mean",0,Vector(Plan.Leaf("a c"),Plan.Leaf("b c")),Vector(.5,.5))
    val second = Plan.Blend("mean",0,Vector(Plan.Leaf("a d"),Plan.Leaf("b d")),Vector(.5,.5))
    assertEquals(frame, Plan.Blend("mean",0,Vector(first,second),Vector(.5,.5)))
  }

  test("reusable functions expand arguments and recursive substitutions are bounded") {
    assertEquals(schedule("$pose($x) = portrait of $x\n$pose(a fox)",1),Vector(Plan.Leaf("portrait of a fox")))
    intercept[EngineError](schedule("$loop = $loop\n$loop",1))
  }
