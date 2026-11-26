package sdfx.vulkan

import sdfx.core.model.{GenerationParams, LoraRef, LoraSyntax}
import sdfx.core.protocol.GenerationRequest
import java.nio.file.Path
import scala.util.Using
import ujson.Value

private[vulkan] object VulkanComposableLora:
  val id = "composable_lora"
  private val controllerNote = """(?:\\u0023|#)[^:#\[\]]+""".r
  private def cleanPrompt(text: String): String =
    controllerNote.replaceAllIn(LoraSyntax.extract(text).prompt, "").trim
  private def flag(config: Value, key: String): Boolean = config.obj.get(key).exists {
    case ujson.Bool(value) => value
    case ujson.Num(value) => value != 0
    case ujson.Str(value) => Set("enable", "enabled", "true", "yes", "on", "1")(value.trim.toLowerCase(java.util.Locale.ROOT))
    case _ => false
  }
  def clock(p: GenerationParams, config: Value, actual: Int, hires: Boolean): (Int, Int) =
    val global = flag(config, "hires_global_step")
    val offset = if global && hires then p.steps else 0
    val total = if global && p.hires.enabled && !p.isImg2Img then
      p.steps + (if p.hires.secondPassSteps == 0 then p.steps else p.hires.secondPassSteps)
    else if !hires && p.isImg2Img then
      p.img2img.fold(actual)(i => (p.steps * i.denoisingStrength).toInt.max(1).min(p.steps))
    else actual
    (offset, total)
  final case class Branch(text: String, coefficient: Double, textAdapters: Vector[(String, Double)],
      unetAdapters: Vector[(String, Double)], unetSchedule: Vector[Vector[(String, Double)]] = Vector.empty)
  final case class Plan(negative: Branch, positive: Vector[Branch]):
    def guidance(prefix: String): String =
      if positive.size == 1 && positive.head.coefficient == 1 then ""
      else positive.zipWithIndex.map { (branch, i) =>
        s"{predict = ${prefix}_branch$i, weight = ${branch.coefficient}}"
      }.mkString(", composable = [", ", ", "]")

  private def tags(text: String): Vector[LoraRef] =
    val found = Vector.newBuilder[LoraRef]
    LoraSyntax.rewriteTags(text, (name, args) =>
      found += LoraSyntax.parseTag(name, args)
      args)
    found.result()

  def needsGraph(req: GenerationRequest): Boolean =
    val p = req.params
    val prompts = Vector(p.prompt) ++ Option.when(p.hires.enabled && p.hires.prompt.nonEmpty)(p.hires.prompt)
    p.loras.nonEmpty || req.manualLoras.exists(_.nonEmpty) || prompts.exists { prompt =>
      tags(prompt).nonEmpty || VulkanNeutralPrompt.leaves(VulkanNeutralPrompt.parse(prompt)).size > 1
    }

  def components(keys: Iterable[String]): (Boolean, Boolean) =
    def text(key: String): Boolean = key.startsWith("lora_te") || key.startsWith("text_encoder") ||
      key.startsWith("lora.cond_stage_model.") || key.startsWith("lora.text_encoders.")
    def unet(key: String): Boolean = key.startsWith("lora_unet") || key.startsWith("unet.") ||
      key.startsWith("lora.diffusion_model.") || key.startsWith("diffusion_model.")
    // Skip a component only when all keys are known to belong to the other component.
    (!keys.forall(unet), !keys.forall(text))

  def plan(prompt: String, negative: String, manual: Option[Seq[LoraRef]], discovered: Seq[LoraRef],
      config: Value, alternatePrompt: String = "", steps: Int = 1, clockOffset: Int = 0, clockTotal: Int = 0): Plan =
    import VulkanNeutralPrompt.{Group, Leaf, Node}
    import VulkanProtocol.{invalid, unsupported}
    val useSteps = flag(config, "with_step")
    val total = if clockTotal > 0 then clockTotal else steps
    if flag(config, "plot_weights") then unsupported("Composable LoRA weight plots")
    val leaves = Vector.newBuilder[(String, Double)]
    def visit(node: Node, scale: Double): Unit =
      if node.kind != "AND" then unsupported("extended AND operators with Composable LoRA")
      val weight = node.weight * scale
      if !weight.toFloat.isFinite then invalid("Composable LoRA branch weight must be finite")
      node match
        case Leaf(text, _, _) => leaves += text -> weight
        case Group(children, _, _) => children.foreach(visit(_, weight))
    visit(VulkanNeutralPrompt.parse(prompt), 1)
    val positive = leaves.result()
    if positive.size > 64 then invalid("Composable LoRA supports up to 64 branches")
    val promptTags = tags(prompt)
    val attached = (promptTags ++ tags(alternatePrompt)).map(_.name).toSet
    val shared = manual.getOrElse(discovered.filterNot(lora => attached(lora.name)))
    val legacy = Option.when(manual.isEmpty)(discovered.map(ref => ref.name -> ref).toMap).getOrElse(Map.empty)
    def check(ref: LoraRef): Unit =
      if ref.lbw.nonEmpty || ref.start.nonEmpty || ref.stop.nonEmpty || ref.dyn.nonEmpty then
        unsupported("Composable LoRA block weights, native timing fields or dynamic ranks")
      if !ref.weight.toFloat.isFinite || ref.unet.exists(w => !w.toFloat.isFinite) then
        invalid("Composable LoRA adapter weights must be finite")
    (promptTags ++ shared).foreach(check)
    legacy.values.foreach { ref =>
      check(ref)
      if ref.weight == 0 && ref.unet.exists(_ != 0) then
        unsupported("legacy Composable LoRA with zero text weight and nonzero U-Net weight")
    }
    def weights(local: Seq[LoraRef], unet: Boolean, controlled: Option[Vector[(LoraRef, Double)]] = None): Vector[(String, Double)] =
      def selected(ref: LoraRef): Double = if unet then ref.unet.getOrElse(ref.weight) else ref.weight
      def component(ref: LoraRef): Double =
        val weight = if unet && manual.isEmpty then
          val ratio = legacy.get(ref.name).flatMap(spec => spec.unet.map(w => if spec.weight == 0 then 0 else w / spec.weight))
          ref.weight * ratio.getOrElse(1.0)
        else selected(ref)
        weight
      val values = controlled.fold(local.iterator.map(ref => ref.name -> component(ref)).toMap) { entries =>
        entries.foldLeft(Map.empty[String, Double]) { case (acc, (ref, multiplier)) =>
          val value = component(ref) * multiplier
          if math.abs(value) > math.abs(acc.getOrElse(ref.name, 0.0)) then acc.updated(ref.name, value) else acc
        }
      }
      val added = shared.groupMapReduce(_.name)(selected)(_ + _)
      (values.keySet ++ added.keySet).toVector.sorted.flatMap { name =>
        val weight = values.getOrElse(name, 0.0) + added.getOrElse(name, 0.0)
        if !weight.toFloat.isFinite then invalid("Combined Composable LoRA weight must be finite")
        Option.when(weight != 0)(name -> weight)
      }
    val singleDefault = positive.size == 1 && !flag(config, "single_no_uc")
    def schedule(program: Option[VulkanLoraSteps.Program]): Vector[Vector[(String, Double)]] =
      program.fold(Vector.empty[Vector[(String, Double)]]) { stepsProgram =>
        Vector.tabulate(steps)(i => weights(Nil, true, Some(stepsProgram.weights(i + 1 + clockOffset, total))))
      }
    val negativeUnet = singleDefault || flag(config, "uc_diffusion_model")
    val negativeSchedule = schedule(Option.when(useSteps && negativeUnet && positive.size == 1)(VulkanLoraSteps.parse(prompt)))
    val uncond = Branch(cleanPrompt(negative), 1,
      if singleDefault || flag(config, "uc_text_model_encoder") then weights(promptTags, false) else Vector.empty,
      negativeSchedule.headOption.getOrElse(if negativeUnet then weights(promptTags, true) else Vector.empty),
      negativeSchedule)
    Plan(uncond, positive.map { (text, coefficient) =>
      val local = tags(text)
      val program = Option.when(useSteps)(VulkanLoraSteps.parse(text))
      val timeline = schedule(program)
      Branch(cleanPrompt(program.fold(text)(_.prompt)), coefficient,
        weights(local, false, program.map(_.weights(-1, total))),
        timeline.headOption.getOrElse(weights(local, true)), timeline)
    })

  def prepare(plan: Plan, req: GenerationRequest, prefix: String, unet: String, text: String, enc: String,
      concat: String, supportsBlank: Boolean): Vector[String] =
    def quoted(value: String): String = ujson.write(ujson.Str(value))
    val lines = Vector.newBuilder[String]
    val branches = (s"${prefix}_neg" -> plan.negative) +:
      plan.positive.zipWithIndex.map((branch, i) => s"${prefix}_branch$i" -> branch)
    val names = branches.flatMap { (_, branch) =>
      (branch.textAdapters ++ branch.unetAdapters ++ branch.unetSchedule.flatten).map(_._1)
    }.distinct
    val adapters = names.zipWithIndex.map { (name, i) =>
      val path = VulkanConditioning.file(req.loraPaths.getOrElse(name,
        VulkanProtocol.invalid(s"Unresolved Composable LoRA adapter: $name")), "LoRA")
      val key = s"${prefix}_adapter$i"
      lines += s"$key = load_lora(${quoted(path)})"
      val (textWeights, unetWeights) = if path.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
        Using.resource(sdfx.vulkan.tools.CheckpointIO.open(Path.of(path)))(file => components(file.tensors.keys))
      else (true, true)
      name -> (key, textWeights, unetWeights)
    }.toMap
    val models = scala.collection.mutable.Map.empty[(String, Vector[(String, Double)]), String]
    def applyAdapters(model: String, weights: Vector[(String, Double)]): String =
      models.getOrElseUpdate((model, weights), weights.foldLeft(model) { case (base, (name, weight)) =>
        val (adapter, _, _) = adapters(name)
        val key = s"${prefix}_model${models.size}_$adapter"
        lines += s"""$key = apply_lora($base, $adapter, weight = $weight, mode = "at_runtime")"""
        key
      })
    for (name, branch) <- branches do
      val encoder = applyAdapters(text, branch.textAdapters.filter(entry => adapters(entry._1)._2))
      val blank = if supportsBlank && name.endsWith("_neg") && branch.text.isEmpty then ", sdxl_blank_negative = true" else ""
      val condition = s"${name}_cond"
      lines += s"$condition = encode_prompt($encoder, ${quoted(branch.text)}, $enc)"
      def predict(symbol: String, weights: Vector[(String, Double)]): String =
        val denoiser = applyAdapters(unet, weights.filter(entry => adapters(entry._1)._3))
        lines += s"$symbol = predictor($denoiser, $condition$concat$blank)"
        symbol
      val timeline = branch.unetSchedule
      val distinct = timeline.distinct
      val scheduleSource = if distinct.size <= 1 then
        predict(name, timeline.headOption.getOrElse(branch.unetAdapters))
      else
        val variants = distinct.zipWithIndex.map { (weights, i) =>
          weights -> predict(s"${name}_stage$i", weights)
        }.toMap
        val changes = timeline.zipWithIndex.collect {
          case (weights, i) if i == 0 || timeline(i - 1) != weights => (i, variants(weights))
        }
        var serial = 0
        def connect(from: Int, until: Int): String =
          if from + 1 == until then changes(from)._2
          else
            val mid = (from + until) / 2
            val before = connect(from, mid)
            val after = connect(mid, until)
            serial += 1
            val symbol = s"${name}_switch$serial"
            lines += s"$symbol = switch_predictor($before, $after, after_steps = ${changes(mid)._1})"
            symbol
        val selected = connect(0, changes.size)
        lines += s"$name = $selected"
        changes.head._2
      if name == s"${prefix}_branch0" then lines += s"${prefix}_schedule = $scheduleSource"
    lines.result()
