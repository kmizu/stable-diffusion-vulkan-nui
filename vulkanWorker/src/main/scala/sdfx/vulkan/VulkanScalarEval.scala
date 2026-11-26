package sdfx.vulkan

/** Compiles a bounded scalar expression without invoking host code. */
private[vulkan] object VulkanScalarEval:
  type Term[C] = C => Double
  private type Binary = (Double, Double) => Double
  private val operators: Map[String, Binary] = Map(
    "+" -> (_ + _), "-" -> (_ - _), "*" -> (_ * _), "/" -> (_ / _),
    "//" -> ((a,b) => math.floor(a / b)),
    "%" -> ((a,b) => { val r = a % b; if r != 0 && (r < 0) != (b < 0) then r + b else r }),
    "**" -> math.pow)
  private val comparisons: Map[String, (Double, Double) => Boolean] =
    Map("<" -> (_ < _), "<=" -> (_ <= _), ">" -> (_ > _), ">=" -> (_ >= _), "==" -> (_ == _), "!=" -> (_ != _))
  private val token = """(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[A-Za-z_]\w*|\*\*|//|<=|>=|==|!=|[+*/%<>,().-]""".r

  def compile[C](source: String, operation: String, variables: Map[String, Term[C]],
      constants: Map[String, Double], call: (String, Vector[Term[C]]) => Option[Term[C]],
      round: Double => Double, qualifiers: Set[String] = Set.empty, pythonBooleans: Boolean = false): Term[C] =
    def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"$operation: $detail")
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
    new Parser(parts.result(), invalid, variables, constants, call, round, qualifiers, pythonBooleans).read()

  private final class Parser[C](tokens: Vector[String], invalid: String => Nothing,
      variables: Map[String, Term[C]], constants: Map[String, Double],
      call: (String, Vector[Term[C]]) => Option[Term[C]], round: Double => Double,
      qualifiers: Set[String], pythonBooleans: Boolean):
    private def truth(value: Boolean): Double = if value then 1.0 else 0.0
    private def constant(value: Double): Term[C] = _ => value
    private def operation(fn: Binary, a: Term[C], b: Term[C]): Term[C] =
      ctx => round(fn(a(ctx), b(ctx)))

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
    def read(): Term[C] =
      val result = conditional()
      if next.nonEmpty then invalid(s"unexpected token '$next'")
      result

    private def conditional(): Term[C] = nested {
      val yes = logicalOr()
      if !accept("if") then yes
      else
        val condition = logicalOr()
        expect("else")
        val no = conditional()
        ctx => if condition(ctx) != 0.0 then yes(ctx) else no(ctx)
    }
    private def repeatedBoolean(word: String, term: () => Term[C]): Term[C] =
      val first = term()
      if next != word then first
      else
        val operands = Vector.newBuilder[Term[C]]
        operands += first
        while accept(word) do operands += term()
        val expressions = operands.result()
        ctx =>
          val values = expressions.iterator.map(_(ctx))
          if pythonBooleans then
            var result = values.next()
            while values.hasNext && ((result != 0.0) == (word == "and")) do result = values.next()
            result
          else
            val combine = if word == "and" then (a: Boolean, b: Boolean) => a && b
              else (a: Boolean, b: Boolean) => a || b
            truth(values.map(_ != 0.0).reduceLeft(combine))
    private def logicalOr(): Term[C] = repeatedBoolean("or", () => logicalAnd())
    private def logicalAnd(): Term[C] = repeatedBoolean("and", () => logicalNot())
    private def logicalNot(): Term[C] =
      if !accept("not") then comparison()
      else
        val inner = nested(logicalNot())
        ctx => truth(inner(ctx) == 0.0)
    private def comparison(): Term[C] =
      val first = sum()
      if !comparisons.contains(next) then first
      else
        val entries = Vector.newBuilder[((Double, Double) => Boolean, Term[C])]
        while comparisons.contains(next) do entries += comparisons(take()) -> sum()
        val chain = entries.result()
        ctx =>
          var lhs = first(ctx)
          val accepted = chain.forall { (compare, expr) =>
            val rhs = expr(ctx)
            val matches = compare(lhs,rhs)
            lhs = rhs
            matches
          }
          truth(accepted)
    private def repeated(allowed: Set[String], term: () => Term[C]): Term[C] =
      var result = term()
      while allowed(next) do result = operation(operators(take()),result,term())
      result
    private def sum(): Term[C] = repeated(Set("+","-"), () => product())
    private def product(): Term[C] = repeated(Set("*","/","//","%"), () => signed())
    private def signed(): Term[C] =
      if accept("+") then nested(signed())
      else if accept("-") then
        val expr = nested(signed())
        ctx => -expr(ctx)
      else
        val lhs = primary()
        if accept("**") then operation(operators("**"),lhs,nested(signed())) else lhs

    private def primary(): Term[C] =
      val head = take()
      if head == "(" then
        val value = conditional()
        expect(")")
        value
      else if head.head.isDigit || head.startsWith(".") then
        constant(round(head.toDoubleOption.getOrElse(invalid(s"invalid number '$head'"))))
      else
        val qualified = qualifiers(head) && accept(".")
        val name = if qualified then take() else head
        if accept("(") then function(name)
        else if qualified && (head != "math" || !constants.contains(name) || Set("True", "False")(name)) then
          invalid(s"unknown name '$head.$name'")
        else constants.get(name).map(value => constant(round(value)))
          .orElse(variables.get(name)).getOrElse(invalid(s"unknown name '$name'"))

    private def function(name: String): Term[C] =
      val args = Vector.newBuilder[Term[C]]
      if !accept(")") then
        args += conditional()
        while accept(",") && next != ")" do args += conditional()
        expect(")")
      call(name, args.result()).getOrElse(invalid(s"unknown function or wrong number of arguments for '$name'"))
