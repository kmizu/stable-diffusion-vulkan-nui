// Prompt syntax and interpolation timing adapted from ljleb/prompt-fusion-extension (MIT).
package sdfx.vulkan

import scala.collection.mutable

private[vulkan] object VulkanFusionPrompt:
  val id = "prompt_fusion"
  val textHooks = Set(id, VulkanWeightGradient.id)
  private val syntax = """(?<!\\)\$[A-Za-z_]|(?<!\\)\[[^\]]*:[^\]]*,[^\]]*\]|(?<!\\)\[[^\]]*:\s*(?:linear|bezier|catmull|mean)\s*\]|(?<!\\)\[[^\]]*\|[^\]]*:[^\]]+\]|(?<!\\)\([^()]*:[^()]*,[^()]*\)""".r
  def active(text: String): Boolean =
    if text.length > 65536 then invalid("prompt exceeds its length limit")
    syntax.findFirstIn(text).nonEmpty
  def applies(hook: String, positive: String, negative: String): Boolean =
    if hook == id then active(positive) || active(negative)
    else if hook == VulkanWeightGradient.id then VulkanWeightGradient.hasGradient(positive) || VulkanWeightGradient.hasGradient(negative)
    else true
  private def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"Prompt Fusion: $detail")

  enum Plan:
    case Leaf(text: String)
    case Blend(method: String, ratio: Double, points: Vector[Plan], weights: Vector[Double] = Vector.empty)
    def mapLeaves(f: String => String): Plan = this match
      case Leaf(text) => Leaf(f(text))
      case Blend(method,ratio,points,weights) => Blend(method,ratio,points.map(_.mapLeaves(f)),weights)
    def visit(f: String => Unit): Unit = this match
      case Leaf(text) => f(text)
      case Blend(_,_,points,_) => points.foreach(_.visit(f))

  private enum Node:
    case Text(value: String)
    case Group(children: Vector[Node])
    case Weight(child: Node, left: Double, right: Double)
    case Wrap(child: Node, open: Char, close: Char, suffix: String)
    case Edit(before: Node, after: Node, boundary: String)
    case Alternate(children: Vector[Node], rate: Option[Double])
    case Curve(children: Vector[Node], positions: Vector[Option[Double]], method: String)

  private def closing(open: Char): Char = if open == '[' then ']' else ')'
  private def close(text: String, start: Int): Int =
    val stack = mutable.ArrayBuffer(closing(text(start)))
    var i = start+1
    while i < text.length && stack.nonEmpty do
      text(i) match
        case '\\' => i += 1
        case '(' | '[' => stack += closing(text(i))
        case c if c == stack.last => stack.remove(stack.size-1)
        case _ => ()
      i += 1
    if stack.nonEmpty then -1 else i-1

  private def split(text: String, separator: Char): Vector[String] =
    val result = Vector.newBuilder[String]
    var i = 0
    var start = 0
    while i < text.length do
      text(i) match
        case '\\' => i += 1
        case '(' | '[' =>
          val end = close(text,i)
          if end >= 0 then i = end
        case c if c == separator =>
          result += text.substring(start,i)
          start = i+1
        case _ => ()
      i += 1
    result += text.substring(start)
    result.result()

  private val declaration = """\s*\$([A-Za-z_]\w*)(?:\(([^()]*)\))?\s*=\s*(.*)""".r
  private val symbol = """[A-Za-z_]\w*""".r
  private case class Definition(parameters: Vector[String], body: String)
  private def substitute(text: String): String =
    val definitions = mutable.Map.empty[String,Definition]
    def expand(value: String, locals: Map[String,String], depth: Int): String =
      if depth > 32 || value.length > 65536 then invalid("substitution expansion exceeds its limit")
      val result = new StringBuilder()
      var i = 0
      while i < value.length do
        if value(i) == '\\' && i+1 < value.length then
          result.append(value(i)).append(value(i+1))
          i += 2
        else if value(i) == '$' then
          val name = symbol.findPrefixOf(value.substring(i+1)).getOrElse(invalid("expected a substitution name"))
          i += name.length+1
          val args =
            if i < value.length && value(i) == '(' then
              val end = close(value,i)
              if end < 0 then invalid(s"unclosed call to $$${name}")
              val body = value.substring(i+1,end)
              i = end+1
              if body.isBlank then Vector.empty else split(body,',').map(x => expand(x.trim,locals,depth+1))
            else Vector.empty
          locals.get(name) match
            case Some(binding) if args.isEmpty => result.append(binding)
            case _ =>
              val binding = definitions.getOrElse(name,invalid(s"unknown substitution $$${name}"))
              if binding.parameters.size != args.size then invalid(s"wrong number of arguments for $$${name}")
              result.append(expand(binding.body,locals ++ binding.parameters.zip(args),depth+1))
        else
          result.append(value(i))
          i += 1
        if result.length > 65536 then invalid("substitution expansion exceeds its limit")
      result.toString
    val lines = text.linesIterator.flatMap {
      case declaration(name,raw,body) =>
        val parameters = Option(raw).filterNot(_.isBlank).toVector.flatMap(_.split(",")).map(_.trim)
        if parameters.exists(x => !x.startsWith("$") || !symbol.matches(x.drop(1))) ||
            parameters.distinct.size != parameters.size then invalid(s"invalid parameters for $$${name}")
        definitions(name) = Definition(parameters.map(_.drop(1)),body)
        None
      case line => Some(expand(line,Map.empty,0))
    }
    lines.mkString("\n").trim

  private def number(s: String): Option[Double] = s.trim.toDoubleOption.filter(_.isFinite)
  private def parse(text: String, fusion: Boolean, depth: Int = 0): Node =
    import Node.*
    if depth > 48 then invalid("bracket nesting exceeds 48")
    def child(s: String): Node = parse(s,fusion,depth+1)
    def branch(s: String): Node = child(if fusion then s.trim else s)
    def bracket(body: String, open: Char): Node =
      val parts = split(body,':')
      if open == '(' then
        val weights = split(parts.last,',')
        if fusion && parts.size == 2 && weights.size == 2 && weights.forall(w => w.isBlank || number(w).nonEmpty) then
          Weight(child(parts.head),number(weights.head).getOrElse(1),number(weights.last).getOrElse(1))
        else if parts.size == 2 && number(parts.last).nonEmpty then Wrap(child(parts.head),open,')',s":${parts.last}")
        else Wrap(child(body),open,')',"")
      else
        val alternates = split(parts.head,'|')
        val method = parts.last.trim
        if alternates.size > 1 && (parts.size == 1 || fusion && parts.size == 2 && number(parts.last).nonEmpty) then
          Alternate(alternates.map(branch),if parts.size == 1 then None else number(parts.last))
        else if fusion && parts.size >= 3 && (Set("linear","catmull","bezier","mean")(method) || parts.last.contains(",")) then
          val named = Set("linear","catmull","bezier","mean")(method)
          val controls = parts.dropRight(if named then 2 else 1)
          val values = split(parts(parts.size-(if named then 2 else 1)),',').map { value =>
            if value.isBlank then None else Some(number(value).getOrElse(invalid(s"invalid curve position '$value'")))
          }
          if controls.size < 2 then invalid("a curve requires at least two prompts")
          val curve = if named then method else "linear"
          if curve != "mean" && values.size != controls.size then invalid("each curve point needs a position or an empty slot")
          if values.size > controls.size then invalid("too many weights for average")
          Curve(controls.map(s => child(s.trim)),values.padTo(controls.size,None),curve)
        else if parts.size >= 2 && parts.size <= 3 && number(parts.last).nonEmpty then
          Edit(if parts.size == 3 then branch(parts.head) else Text(""),branch(parts(parts.size-2)),parts.last.trim)
        else Wrap(child(body),open,']',"")
    val nodes = Vector.newBuilder[Node]
    var start = 0
    var index = 0
    while index < text.length do
      text(index) match
        case '\\' => index += 1
        case open @ ('[' | '(') =>
          val end = close(text,index)
          if end >= 0 then
            if start < index then nodes += Text(text.substring(start,index))
            nodes += bracket(text.substring(index+1,end),open)
            index = end
            start = end+1
        case _ => ()
      index += 1
    if start < text.length then nodes += Text(text.substring(start))
    Group(nodes.result())

  private def fusionBoundary(value: Double, steps: Int): Int =
    (if value > 0 && value < 1 then value*steps else value+1).toInt

  def schedule(text: String, steps: Int, fusionClock: Boolean = true, gradient: Boolean = false,
      requestedSteps: Option[Int] = None): Vector[Plan] =
    if steps <= 0 then invalid("empty sampling pass")
    if text.length > 65536 then invalid("prompt exceeds its length limit")
    val resolved = if fusionClock && text.contains("$") then substitute(text) else text
    val hasGradient = gradient && VulkanWeightGradient.hasGradient(resolved)
    val parsed = Option.when(!hasGradient)(parse(resolved,fusionClock))
    val prompts = mutable.Set.empty[String]
    def point(node: Node, step: Int, start: Int, stop: Int, prefix: Plan): Plan =
      import Node.*
      def nested(child: Node, initial: Plan = prefix): Plan = point(child,step,start,stop,initial)
      def append(value: String): Plan = prefix.mapLeaves(_+value)
      def blend(method: String, ratio: Double, children: Vector[Node], weights: Vector[Double] = Vector.empty): Plan =
        var leaves = 0
        val points = children.map { child =>
          val plan = nested(child)
          plan.visit { _ =>
            leaves += 1
            if leaves > 512 then invalid("expansion exceeds 512 conditioning prompts")
          }
          plan
        }
        Plan.Blend(method,ratio,points,weights)
      node match
        case Text(value) => append(value)
        case Group(children) => children.foldLeft(prefix)((value,child) => nested(child,value))
        case Wrap(child,open,close,suffix) => nested(child,append(open.toString)).mapLeaves(_+suffix+close)
        case Weight(child,first,last) =>
          val weight = first + (last-first) * (step-start) / math.max(stop-start-1,1)
          nested(child,append("(")).mapLeaves(_+s":$weight)")
        case Edit(before,after,at) =>
          val value = at.toDouble
          val phase = if fusionClock then fusionBoundary(value,steps)
            else if at.contains('.') then (value*requestedSteps.getOrElse(steps)).toInt
            else value.toInt
          val bound = phase.max(start).min(stop)
          if step < bound then point(before,step,start,bound,prefix) else point(after,step,bound,stop,prefix)
        case Alternate(children,None) => nested(children(step % children.size))
        case Alternate(children,Some(speed)) =>
          val position = ((step-start)*speed / children.size) % 1
          val t = if position < 0 then position+1 else position
          blend("linear",t,children :+ children.head)
        case Curve(children,positions,"mean") =>
          val weights = positions.flatten
          val sum = weights.sum
          if weights.nonEmpty && sum == 0 then invalid("explicit average weights sum to zero")
          val actual = positions.map(_.fold(1.0/children.size)(_ / sum * weights.size / children.size))
          blend("mean",0,children,actual)
        case Curve(children,positions,method) =>
          val knots = positions.map(_.map(x => fusionBoundary(x,steps).toDouble)).toArray
          if knots.head.isEmpty then knots(0) = Some(start.toDouble)
          if knots.last.isEmpty then knots(knots.length-1) = Some(stop.toDouble)
          var i = 0
          while i < knots.length-1 do
            var end = i+1
            while knots(end).isEmpty do end += 1
            val left = knots(i).get
            val right = knots(end).get
            if right < left then invalid("curve positions must be ordered")
            for k <- i+1 until end do knots(k) = Some(left + (right-left)*(k-i)/(end-i))
            i = end
          val ks = knots.map(_.get)
          val fraction =
            if step <= ks.head then 0d
            else if step >= ks.last then 1d
            else
              val lower = ks.lastIndexWhere(_ < step)
              (lower + (step-ks(lower))/(ks(lower+1)-ks(lower))) / (ks.length-1)
          blend(method,fraction,children)
    Vector.tabulate(steps) { step =>
      val root = parsed.getOrElse(parse(VulkanWeightGradient.render(resolved,step+1,steps),fusionClock))
      val plan = point(root,step,0,steps,Plan.Leaf(""))
      var leaves = 0
      plan.visit { prompt =>
        leaves += 1
        prompts += prompt
        if leaves > 512 || prompts.size > 512 || prompt.length > 65536 then invalid("expansion exceeds 512 conditioning prompts")
      }
      plan
    }
