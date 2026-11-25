// Channel projection adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan

import ujson.Value

private[vulkan] final class VulkanColorGrading(config: Value, sdxl: Boolean, width: Int, height: Int, total: Int):
  private def enabled(key: String, default: Boolean): Boolean = config.obj.get(key).fold(default)(_.bool)
  private def number(key: String, default: Double, low: Double, high: Double): Double =
    val n = config.obj.get(key).fold(default)(_.num)
    require(n.isFinite && n >= low && n <= high,s"Invalid color grading $key")
    n
  private val recenter = enabled("recenter",true)
  private val normalize = enabled("normalize",false)
  private val shift = number("recenter_strength",1,0,2)*2/total
  private val strength = number("normalize_strength",1,0,1)
  private val target =
    val r = number("r",0,-1,1)
    val g = number("g",0,-1,1)
    val b = number("b",0,-1,1)
    VulkanLatentMath.rgb(sdxl,r,g,b).map(_*0.5)

  def append(lines: scala.collection.mutable.Builder[String,Vector[String]], input: String, step: Int): String =
    var value = input
    def put(name: String,expression: String): String =
      val variable = s"grade${step}_$name"
      lines += s"$variable = $expression"
      variable
    if recenter then
      val mean = VulkanLatentMath.mean(value,width,height,"",put)
      val wanted = target.map(_.toString).mkString("[",",","]")
      val targetTensor = put("target",s"tensor::reshape(tensor::vector($wanted, dtype = \"f32\"), [1,1,4,1])")
      val delta = put("delta",s"tensor::add($targetTensor, tensor::mul($mean,-1))")
      val chroma = put("chroma",s"tensor::index_fill($delta, axis = 2, indices = [0], value = 0)")
      value = put("centered",s"tensor::add($value, tensor::mul($chroma,$shift))")
    if normalize && step==total then
      val pool = s"kernel = [$width,$height], axes = [0,1]"
      val low = put("low",s"tensor::min_pool2d($value, $pool)")
      val high = put("high",s"tensor::max_pool2d($value, $pool)")
      val mid = put("mid",s"tensor::mul(tensor::add($high,$low),0.5)")
      val half = put("half",s"tensor::clamp(tensor::mul(tensor::add($high,tensor::mul($low,-1)),0.5), min = 0.000001, max = 1.0e30)")
      val aim = if sdxl then 3.5 else 4.0
      val multiplier = put("mult",s"tensor::clamp(tensor::div($aim,$half), min = 0, max = 1.5)")
      val change = put("change",s"tensor::mul(tensor::add($value,tensor::mul($mid,-1)), tensor::mul(tensor::add($multiplier,-1),$strength))")
      value = put("normalized",s"tensor::add($value,$change)")
    value
