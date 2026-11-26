package sdfx.vulkan

import org.apache.commons.math3.special.Erf

/** The scalar operations of the Clip I/O expression language, applied to F32 conditioning. */
private[vulkan] object VulkanClipEval:
  private def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"Clip I/O ?eval: $detail")
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
  private val operators: Map[String, Binary] = Map(
    "+" -> (_ + _), "-" -> (_ - _), "*" -> (_ * _), "/" -> (_ / _),
    "//" -> ((a,b) => math.floor(a / b)),
    "%" -> ((a,b) => { val r = a % b; if r != 0 && (r < 0) != (b < 0) then r + b else r }),
    "**" -> math.pow)
  private val comparisons: Map[String, (Float, Float) => Boolean] =
    Map("<" -> (_ < _), "<=" -> (_ <= _), ">" -> (_ > _), ">=" -> (_ >= _), "==" -> (_ == _), "!=" -> (_ != _))
  private def truth(value: Boolean): Float = if value then 1f else 0f
  private def constant(value: Float): Expression = (_, _, _) => value
  private def operation(fn: Binary, a: Expression, b: Expression): Expression =
    (v,t,d) => fn(a(v,t,d),b(v,t,d)).toFloat

  private val token = """(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[A-Za-z_]\w*|\*\*|//|<=|>=|==|!=|[+*/%<>,().-]""".r

  def compile(source: String): Expression =
    if source.isBlank || source.length > 2000 then invalid("expression must be between 1 and 2000 characters")
    val matcher = token.pattern.matcher(source)
    val parts = Vector.newBuilder[String]
    var index = 0
    var count = 0
    while index < source.length do
      if source(index).isWhitespace then index += 1
      else
        matcher.region(index, source.length)
        if !matcher.lookingAt() then invalid(s"unexpected character at column ${index+1}")
        parts += matcher.group()
        index = matcher.end()
        count += 1
        if count > 400 then invalid("expression is too complex")
    new Parser(parts.result()).read()

  private final class Parser(tokens: Vector[String]):
    private var index = 0
    private var depth = 0
    private def next: String = tokens.lift(index).getOrElse("")
    private def take(): String =
      val value = next
      if value.isEmpty then invalid("unexpected end of expression")
      index += 1
      value
    private def accept(value: String): Boolean =
      if next == value then { index += 1; true } else false
    private def expect(value: String): Unit =
      if !accept(value) then invalid(s"expected '$value'")
    private def nested[A](body: => A): A =
      depth += 1
      if depth > 64 then invalid("expression nesting exceeds 64")
      try body finally depth -= 1
    def read(): Expression =
      val result = conditional()
      if next.nonEmpty then invalid(s"unexpected token '$next'")
      result

    private def conditional(): Expression = nested {
      val yes = logicalOr()
      if !accept("if") then yes
      else
        val condition = logicalOr()
        expect("else")
        val no = conditional()
        (v,t,d) => if condition(v,t,d) != 0f then yes(v,t,d) else no(v,t,d)
    }
    private def repeatedBoolean(word: String, term: () => Expression, combine: (Boolean, Boolean) => Boolean): Expression =
      val first = term()
      if next != word then first
      else
        val operands = Vector.newBuilder[Expression]
        operands += first
        while accept(word) do operands += term()
        val expressions = operands.result()
        (v,t,d) => truth(expressions.iterator.map(_(v,t,d) != 0f).reduceLeft(combine))
    private def logicalOr(): Expression = repeatedBoolean("or", () => logicalAnd(), _ || _)
    private def logicalAnd(): Expression = repeatedBoolean("and", () => logicalNot(), _ && _)
    private def logicalNot(): Expression =
      if !accept("not") then comparison()
      else
        val inner = nested(logicalNot())
        (v,t,d) => truth(inner(v,t,d) == 0f)
    private def comparison(): Expression =
      val first = sum()
      if !comparisons.contains(next) then first
      else
        val entries = Vector.newBuilder[((Float, Float) => Boolean, Expression)]
        while comparisons.contains(next) do entries += comparisons(take()) -> sum()
        val chain = entries.result()
        (v,t,d) =>
          var lhs = first(v,t,d)
          val accepted = chain.forall { (compare, expr) =>
            val rhs = expr(v,t,d)
            val matches = compare(lhs,rhs)
            lhs = rhs
            matches
          }
          truth(accepted)
    private def repeated(allowed: Set[String], term: () => Expression): Expression =
      var result = term()
      while allowed(next) do result = operation(operators(take()),result,term())
      result
    private def sum(): Expression = repeated(Set("+","-"), () => product())
    private def product(): Expression = repeated(Set("*","/","//","%"), () => signed())
    private def signed(): Expression =
      if accept("+") then nested(signed())
      else if accept("-") then
        val expr = nested(signed())
        (v,t,d) => -expr(v,t,d)
      else
        val lhs = primary()
        if accept("**") then operation(operators("**"),lhs,nested(signed())) else lhs

    private def primary(): Expression =
      val head = take()
      if head == "(" then
        val value = conditional()
        expect(")")
        value
      else if head.head.isDigit || head.startsWith(".") then
        constant(head.toFloatOption.getOrElse(invalid(s"invalid number '$head'")))
      else
        val qualified = Set("math","torch")(head) && accept(".")
        val name = if qualified then take() else head
        if accept("(") then function(name)
        else if qualified && (head != "math" || !constants.contains(name) || Set("True","False")(name)) then
          invalid(s"unknown name '$head.$name'")
        else if constants.contains(name) then constant(constants(name).toFloat)
        else name match
          case "i" | "o" => (v,_,_) => v
          case "t" => (_,t,_) => t.toFloat
          case "d" => (_,_,d) => d.toFloat
          case _ => invalid(s"unknown name '$name'")

    private def function(name: String): Expression =
      val args = Vector.newBuilder[Expression]
      if !accept(")") then
        args += conditional()
        while accept(",") && next != ")" do args += conditional()
        expect(")")
      args.result() match
        case Vector(a) if unary.contains(name) =>
          val fn = unary(name)
          (v,t,d) => fn(a(v,t,d)).toFloat
        case Vector(a,b) if binary.contains(name) => operation(binary(name),a,b)
        case Vector(test,yes,no) if name == "where" =>
          (v,t,d) => if test(v,t,d) != 0f then yes(v,t,d) else no(v,t,d)
        case Vector(x,low,high) if name == "clamp" =>
          (v,t,d) => math.min(math.max(x(v,t,d),low(v,t,d)),high(v,t,d))
        case _ => invalid(s"unknown function or wrong number of arguments for '$name'")
