package sdfx.vulkan

import sdfx.core.model.LoraRef

private[vulkan] object VulkanLoraSteps:
  private val tag = """<(?:lora|lyco):([^:>]+)((?::[^:>]*)*)>""".r
  private val block = """(?<!\\)\[\s*\[[^\[\]]+\]\s*:[^\[\]]+\]|(?<!\\)\[[^\[\]]+\]""".r
  private val modifier = """(?:\\u0023|#)\s*(increment|decrease)\b""".r
  private val custom = """(?:\\u0023|#)\s*cmd\s*(?:\(|\\\()""".r
  private val placeholder = """\uE000(\d+)\uE001""".r

  private case class Rule(choices: Vector[Vector[LoraRef]], start: Double, end: Double, ramp: String):
    def weights(step: Int, total: Int): Vector[(LoraRef, Double)] =
      if step < 0 then
        val enabled = if choices.size == 1 then choices.head else
          (0 until choices.map(_.size).max).flatMap { index =>
            choices.flatMap(_.lift(index)).distinctBy(_.name)
          }.toVector
        enabled.map(_ -> 1.0)
      else
        def resolve(endpoint: Double) = if endpoint < 1 then endpoint * total else endpoint
        val from = resolve(start)
        val to = if end < 0 then total.toDouble else resolve(end)
        if step < from || step > to then Vector.empty
        else
          val progress = (step - from) / math.max(to - from, 1.0)
          val multiplier = ramp match
            case "increment" => progress
            case "decrease" => 1 - progress
            case _ => 1.0
          choices(step % choices.size).map(_ -> multiplier)

  final class Program private[VulkanLoraSteps] (val prompt: String, private val rules: Vector[Rule]):
    def weights(step: Int, total: Int): Vector[(LoraRef, Double)] = rules.flatMap(_.weights(step, total))

  def parse(prompt: String): Program =
    import VulkanProtocol.{invalid, unsupported}
    if custom.findFirstIn(prompt).nonEmpty then unsupported("Composable LoRA #cmd controllers")
    val refs = tag.findAllMatchIn(prompt).map(m => sdfx.core.model.LoraSyntax.parseTag(m.group(1).trim, m.group(2))).toVector
    var index = 0
    val encoded = tag.replaceAllIn(prompt, _ =>
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
        val ramp = modifier.findFirstMatchIn(schedule).fold("")(_.group(1))
        val fields = modifier.replaceAllIn(schedule, "").split(":", -1).map(_.trim).toVector
        def choices(text: String): Vector[Vector[LoraRef]] =
          text.split("\\|", -1).toVector.map { value =>
            if placeholder.replaceAllIn(value, "").trim.nonEmpty then
              unsupported("prompt text inside a Composable LoRA step block")
            names(value)
          }
        val isPlain = fields.size == 1 && !nested && ramp.isEmpty && !fields.head.contains("|")
        if isPlain then
          val remaining = placeholder.replaceAllIn(inner, "").trim
          if remaining.isEmpty then "" else scala.util.matching.Regex.quoteReplacement(s"[$remaining]")
        else
          val result = fields match
            case Vector(body, empty, end) if empty.isEmpty =>
              val start = if nested then boundary(inner.substring(split + 1).trim.stripPrefix(":")) else 0.0
              Vector(Rule(choices(body), start, boundary(end), ramp))
            case Vector(body, start) if !nested =>
              Vector(Rule(choices(body), boundary(start), -1, ramp))
            case Vector(before, after, at) if !nested =>
              val endpoint = boundary(at)
              Vector(Rule(choices(before), 0, endpoint, ramp), Rule(choices(after), endpoint, -1, ramp))
            case Vector(body) if !nested =>
              Vector(Rule(choices(body), 0, -1, ramp))
            case _ => invalid("Invalid Composable LoRA step block")
          rules ++= result
          "")
    if names(surroundings).nonEmpty then
      // Ungrouped tags have the legacy static weight for the full pass.
      rules += Rule(Vector(names(surroundings)), 0, Double.PositiveInfinity, "")
    new Program(placeholder.replaceAllIn(surroundings, "").trim, rules.result())
