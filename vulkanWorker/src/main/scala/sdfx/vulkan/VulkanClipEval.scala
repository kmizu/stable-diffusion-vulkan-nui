package sdfx.vulkan

import org.apache.commons.math3.special.Erf

/** The scalar operations of the Clip I/O expression language, applied to F32 conditioning. */
private[vulkan] object VulkanClipEval:
  type Expression = (Float, Int, Int) => Float
  private type Unary = Double => Double
  private type Binary = (Double, Double) => Double
  private val constants = Map("pi" -> math.Pi, "e" -> math.E, "tau" -> (2 * math.Pi),
    "inf" -> Double.PositiveInfinity, "nan" -> Double.NaN, "True" -> 1.0, "False" -> 0.0)
  private val unary: Map[String, Unary] = Map(
    "sin" -> math.sin, "cos" -> math.cos, "tan" -> math.tan, "asin" -> math.asin,
    "acos" -> math.acos, "atan" -> math.atan, "sinh" -> math.sinh, "cosh" -> math.cosh,
    "tanh" -> math.tanh, "exp" -> math.exp, "log" -> math.log, "log2" -> (x => math.log(x) / math.log(2)),
    "log10" -> math.log10, "sqrt" -> math.sqrt, "abs" -> math.abs, "fabs" -> math.abs,
    "floor" -> math.floor, "ceil" -> math.ceil, "trunc" -> (x => if x < 0 then math.ceil(x) else math.floor(x)),
    "erf" -> Erf.erf, "sigmoid" -> (x => 1.0 / (1.0 + math.exp(-x))),
    "sign" -> (x => if x > 0 then 1 else if x < 0 then -1 else 0))
  private val binary: Map[String, Binary] = Map(
    "atan2" -> math.atan2, "pow" -> math.pow, "fmod" -> (_ % _), "copysign" -> math.copySign,
    "min" -> math.min, "max" -> math.max)

  private case class Cell(value: Float, token: Int, dimension: Int)
  private type Term = VulkanScalarEval.Term[Cell]
  private val variables: Map[String, Term] = Map("i" -> (_.value.toDouble), "o" -> (_.value.toDouble),
    "t" -> (_.token.toDouble), "d" -> (_.dimension.toDouble))
  private def round(value: Double): Double = value.toFloat.toDouble
  private def function(name: String, args: Vector[Term]): Option[Term] = args match
    case Vector(a) if unary.contains(name) =>
      val fn = unary(name)
      Some(ctx => round(fn(a(ctx))))
    case Vector(a,b) if binary.contains(name) =>
      val fn = binary(name)
      Some(ctx => round(fn(a(ctx), b(ctx))))
    case Vector(test,yes,no) if name == "where" =>
      Some(ctx => if test(ctx) != 0.0 then yes(ctx) else no(ctx))
    case Vector(x,low,high) if name == "clamp" =>
      Some(ctx => math.min(math.max(x(ctx), low(ctx)), high(ctx)))
    case _ => None

  def compile(source: String): Expression =
    val expression = VulkanScalarEval.compile[Cell](source, "Clip I/O ?eval", variables, constants, function, round,
      qualifiers = Set("math", "torch"))
    (value, token, dimension) => expression(Cell(value, token, dimension)).toFloat
