package sdfx.vulkan

import sdfx.core.model.LoraRef

private[vulkan] object VulkanLoraSteps:
  private val tag = """<(?:lora|lyco):([^:>]+)((?::[^:>]*)*)>""".r
  private val block = """(?<!\\)\[\s*\[[^\[\]]+\]\s*:[^\[\]]+\]|(?<!\\)\[[^\[\]]+\]""".r
  private val modifier = """(?:\\u0023|#)\s*(increment|decrease)\b""".r
  private val custom = """(?:\\u0023|#)\s*cmd\s*(?:\(|\\\()""".r
  private val placeholder = """\uE000(\d+)\uE001""".r
  private val commandMarker = """\uE002(\d+)\uE003""".r

  private case class Command(start: Int, end: Int, body: String)
  private def commands(prompt: String): Vector[Command] =
    val found = Vector.newBuilder[Command]
    var after = 0
    for m <- custom.findAllMatchIn(prompt) if m.start >= after do
      var depth = 1
      var end = m.end
      while end < prompt.length && depth > 0 do
        prompt(end) match
          case '(' => depth += 1
          case ')' => depth -= 1
          case _ => ()
        end += 1
      if depth != 0 then VulkanProtocol.invalid("Unclosed Composable LoRA #cmd expression")
      val body = prompt.substring(m.end, end - 1).stripSuffix("\\").replace("\\(", "(").replace("\\)", ")")
      found += Command(m.start, end, body)
      after = end
    found.result()

  private def replaceCommands(prompt: String, found: Vector[Command], replacement: (Command, Int) => String): String =
    val result = new StringBuilder
    var start = 0
    for (command, i) <- found.zipWithIndex do
      result.append(prompt.substring(start, command.start))
      result.append(replacement(command, i))
      start = command.end
    result.append(prompt.substring(start)).result()

  def maskCommands(prompt: String): String =
    replaceCommands(prompt, commands(prompt), (command, _) => " " * (command.end - command.start))

  private enum Controller:
    case Static, Increase, Decrease
    case Math(expression: VulkanLoraCommand.Expression)

  private case class Rule(choices: Vector[Vector[LoraRef]], start: Double, end: Double, controller: Controller):
    def weights(step: Int, total: Int, negative: Boolean, loraCount: Int, branchLoraCount: Int,
        expressionWeight: LoraRef => Double): Vector[(LoraRef, Double)] =
      def weight(ref: LoraRef, progress: Double): (LoraRef, Double) =
        val multiplier = controller match
          case Controller.Math(expression) =>
            val base = expressionWeight(ref)
            val value = expression(VulkanLoraCommand.Context(base, progress, step, total, negative,
              loraCount, branchLoraCount))
            if !value.isFinite || !(value * base).isFinite then
              VulkanProtocol.invalid("Composable LoRA #cmd result must be finite")
            if step < 0 then 1.0 else value
          case Controller.Increase if step >= 0 => progress
          case Controller.Decrease if step >= 0 => 1.0 - progress
          case _ => 1.0
        ref -> multiplier
      if step < 0 then
        val enabled = if choices.size == 1 then choices.head else
          (0 until choices.map(_.size).max).flatMap { index =>
            choices.flatMap(_.lift(index)).distinctBy(_.name)
          }.toVector
        enabled.map(weight(_, -1))
      else
        def resolve(endpoint: Double) = if endpoint < 1 then endpoint * total else endpoint
        val from = resolve(start)
        val to = if end < 0 then total.toDouble else resolve(end)
        if step < from || step > to then Vector.empty
        else
          val progress = (step - from) / math.max(to - from, 1.0)
          choices(step % choices.size).map(weight(_, progress))

  final class Program private[VulkanLoraSteps] (val prompt: String, private val rules: Vector[Rule], branchLoraCount: Int):
    def weights(step: Int, total: Int, negative: Boolean = false, loraCount: Option[Int] = None,
        expressionWeight: LoraRef => Double = _.weight): Vector[(LoraRef, Double)] =
      rules.flatMap(_.weights(step, total, negative, loraCount.getOrElse(branchLoraCount), branchLoraCount, expressionWeight))

  def parse(prompt: String): Program =
    import VulkanProtocol.{invalid, unsupported}
    val found = commands(prompt)
    val protectedPrompt = replaceCommands(prompt, found, (_, i) => "\uE002" + i + "\uE003")
    val refs = tag.findAllMatchIn(protectedPrompt).map(m => sdfx.core.model.LoraSyntax.parseTag(m.group(1).trim, m.group(2))).toVector
    var index = 0
    val encoded = tag.replaceAllIn(protectedPrompt, _ =>
      val key = "\uE000" + index + "\uE001"
      index += 1
      key)
    def names(text: String) = placeholder.findAllMatchIn(text).map(m => refs(m.group(1).toInt)).toVector
    def boundary(text: String): Double =
      text.trim.toDoubleOption.filter(_.isFinite).getOrElse(invalid(s"Invalid Composable LoRA step index '$text'"))
    val rules = Vector.newBuilder[Rule]
    val surroundings = block.replaceAllIn(encoded, m =>
      val inner = m.matched.drop(1).dropRight(1).trim
      if names(inner).isEmpty then scala.util.matching.Regex.quoteReplacement(m.matched)
      else
        val nested = inner.startsWith("[") && inner.indexOf(']') >= 0
        val split = if nested then inner.indexOf(']') else -1
        val schedule = if nested then inner.substring(1, split) else inner
        val ramp = modifier.findFirstMatchIn(schedule)
        val command = commandMarker.findFirstMatchIn(schedule)
        val controller = command.filter(c => ramp.forall(_.start > c.start)).fold {
          ramp.map(_.group(1)) match
            case Some("increment") => Controller.Increase
            case Some("decrease") => Controller.Decrease
            case _ => Controller.Static
        }(c => Controller.Math(VulkanLoraCommand.compile(found(c.group(1).toInt).body)))
        val fields = commandMarker.replaceAllIn(modifier.replaceAllIn(schedule, ""), "").split(":", -1).map(_.trim).toVector
        def choices(text: String): Vector[Vector[LoraRef]] =
          text.split("\\|", -1).toVector.map { value =>
            if placeholder.replaceAllIn(value, "").trim.nonEmpty then
              unsupported("prompt text inside a Composable LoRA step block")
            names(value)
          }
        val isPlain = fields.size == 1 && !nested && controller == Controller.Static && !fields.head.contains("|")
        if isPlain then
          val remaining = placeholder.replaceAllIn(inner, "").trim
          if remaining.isEmpty then "" else scala.util.matching.Regex.quoteReplacement(s"[$remaining]")
        else
          val result = fields match
            case Vector(body, empty, end) if empty.isEmpty =>
              val start = if nested then boundary(inner.substring(split + 1).trim.stripPrefix(":")) else 0.0
              Vector(Rule(choices(body), start, boundary(end), controller))
            case Vector(body, start) if !nested =>
              Vector(Rule(choices(body), boundary(start), -1, controller))
            case Vector(before, after, at) if !nested =>
              val endpoint = boundary(at)
              Vector(Rule(choices(before), 0, endpoint, controller), Rule(choices(after), endpoint, -1, controller))
            case Vector(body) if !nested =>
              Vector(Rule(choices(body), 0, -1, controller))
            case _ => invalid("Invalid Composable LoRA step block")
          rules ++= result
          "")
    if names(surroundings).nonEmpty then
      // Ungrouped tags have the legacy static weight for the full pass.
      rules += Rule(Vector(names(surroundings)), 0, Double.PositiveInfinity, Controller.Static)
    val clean = commandMarker.replaceAllIn(placeholder.replaceAllIn(surroundings, ""), "").trim
    new Program(clean, rules.result(), refs.distinctBy(_.name).size)
