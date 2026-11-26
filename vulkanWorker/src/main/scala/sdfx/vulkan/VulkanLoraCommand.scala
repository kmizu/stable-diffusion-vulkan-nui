package sdfx.vulkan

import java.util.concurrent.ThreadLocalRandom
import org.apache.commons.math3.special.Gamma

private[vulkan] object VulkanLoraCommand:
  val name = "Composable LoRA #cmd"

  final case class Context(weight: Double, progress: Double, step: Int, steps: Int, negative: Boolean,
      loraCount: Int, branchLoraCount: Int)
  type Expression = VulkanScalarEval.Term[Context]
  private val turn = 2 * math.Pi
  private val constants = Map("pi" -> math.Pi, "nan" -> Double.NaN, "inf" -> Double.PositiveInfinity,
    "True" -> 1.0, "False" -> 0.0, "enable_prepare_step" -> 0.0)
  private val variables: Map[String, Expression] = Map(
    "weight" -> (_.weight), "life" -> (c => if c.step < 0 then 0.0 else c.progress),
    "step" -> (_.step.toDouble), "steps" -> (_.steps.toDouble),
    "is_negative" -> (c => if c.negative then 1.0 else 0.0),
    "lora_count" -> (_.loraCount.toDouble), "block_lora_count" -> (_.branchLoraCount.toDouble))

  private val unary: Map[String, Double => Double] = Map(
    "abs" -> math.abs, "ceil" -> math.ceil, "floor" -> math.floor,
    "trunc" -> (x => if x < 0 then math.ceil(x) else math.floor(x)),
    "gamma" -> Gamma.gamma, "sqrt" -> math.sqrt, "cbrt" -> math.cbrt, "exp" -> math.exp,
    "log" -> math.log, "ln" -> math.log, "log2" -> (x => math.log(x) / math.log(2.0)), "log10" -> math.log10,
    "clamp" -> (x => x.max(0.0).min(1.0)),
    "asin" -> (x => (math.acos(1 - 2 * x) + turn) / turn),
    "acos" -> (x => (math.acos(2 * x - 1) + turn) / turn),
    "atan" -> (x => (math.atan(x) + math.Pi) / turn),
    "sin" -> (x => (math.sin(x * turn - math.Pi / 2) + 1) / 2),
    "cos" -> (x => (math.cos(x * turn + math.Pi / 2) + 1) / 2),
    "tan" -> (x => math.tan(x * turn)),
    "sinr" -> math.sin, "cosr" -> math.cos, "tanr" -> math.tan, "asinr" -> math.asin,
    "acosr" -> math.acos, "atanr" -> math.atan, "sinh" -> math.sinh, "cosh" -> math.cosh, "tanh" -> math.tanh,
    "asinh" -> (x => math.copySign(math.log1p(math.abs(x) + math.abs(x) * (math.abs(x) / (1 + math.hypot(x, 1)))), x)),
    "acosh" -> (x => math.log(x) + math.log1p(math.sqrt((1 - 1 / x) * (1 + 1 / x)))),
    "atanh" -> (x => (math.log1p(x) - math.log1p(-x)) / 2),
    "abssin" -> (x => math.abs(math.sin(x * turn))), "abscos" -> (x => math.abs(math.cos(x * turn))))

  private def integer(value: Double): BigInt =
    if !value.isWhole || math.abs(value) > 9007199254740992.0 then
      VulkanProtocol.invalid(s"$name requires an exactly representable integer argument")
    BigInt(value.toLong)

  private def count(n: Double, k: Double, combinations: Boolean): Double =
    val size = integer(n).toLong
    val selected = integer(k).toLong
    if size < 0 || selected < 0 then VulkanProtocol.invalid(s"$name requires nonnegative counts")
    if selected > size then 0
    else
      val terms = if combinations then selected.min(size - selected) else selected
      var result = 1.0
      var index = 1L
      while index <= terms && result.isFinite do
        val factor = (size - index + 1).toDouble
        result *= (if combinations then factor / index else factor)
        index += 1
      result

  private def lcm(a: BigInt, b: BigInt): BigInt =
    if a == 0 || b == 0 then BigInt(0) else (a / a.gcd(b) * b).abs

  private val binary: Map[String, (Double, Double) => Double] = Map(
    "pow" -> math.pow, "fmod" -> (_ % _), "log" -> ((a, b) => math.log(a) / math.log(b)),
    "ln" -> ((a, b) => math.log(a) / math.log(b)),
    "perm" -> ((a, b) => count(a, b, combinations = false)),
    "comb" -> ((a, b) => count(a, b, combinations = true)))

  private def function(helper: String, args: Vector[Expression]): Option[Expression] =
    (helper, args) match
      case ("random", Vector()) => Some(_ => ThreadLocalRandom.current().nextDouble())
      case ("warmup", Vector(at)) => Some { c =>
        val boundary = at(c)
        if c.progress < boundary then c.progress / boundary else 1
      }
      case ("cooldown", Vector(at)) => Some { c =>
        val boundary = at(c)
        if c.progress > boundary then (1 - c.progress) / (1 - boundary) else 1
      }
      case ("gcd", values) => Some(c => values.foldLeft(BigInt(0))((g, term) => g.gcd(integer(term(c)))).toDouble)
      case ("lcm", values) => Some(c => values.foldLeft(BigInt(1))((m, term) => lcm(m, integer(term(c)))).toDouble)
      case ("perm", Vector(n)) => Some(c => { val size = n(c); count(size, size, combinations = false) })
      case (_, Vector(arg)) => unary.get(helper).map(fn => c => fn(arg(c)))
      case (_, Vector(a, b)) => binary.get(helper).map(fn => c => fn(a(c), b(c)))
      case _ => None

  def compile(source: String): Expression =
    VulkanScalarEval.compile[Context](source, name, variables, constants, function, identity, pythonBooleans = true)
