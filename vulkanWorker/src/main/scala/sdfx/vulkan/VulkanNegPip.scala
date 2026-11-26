package sdfx.vulkan

import scala.collection.mutable
import sdfx.core.model.GenerationParams

private[vulkan] object VulkanNegPip:
  val id = "negpip"

  def enabled(p: GenerationParams): Boolean =
    p.hooks.get(id).exists(!_.obj.get("enabled").contains(ujson.Bool(false)))

  private def gradient(p: GenerationParams): Boolean =
    p.hooks.get(VulkanWeightGradient.id).exists(!_.obj.get("enabled").contains(ujson.Bool(false)))

  private def steps(p: GenerationParams): Int =
    p.img2img.fold(p.steps)(i => math.min(p.steps,(i.denoisingStrength*p.steps).toInt+1))

  private def signed(span: PromptSpan): Boolean =
    span.weight < 0 && span.text.nonEmpty && !(span.text == "BREAK" && span.weight == -1)

  private def hasSign(text: String): Boolean = text.contains('-') && VulkanPromptAttention.parse(text).exists(signed)

  def applies(p: GenerationParams, positive: String, negative: String): Boolean =
    enabled(p) && Seq(positive,negative).exists { text =>
      hasSign(text) || gradient(p) && VulkanWeightGradient.hasGradient(text) &&
        (1 to steps(p)).exists(step => hasSign(VulkanWeightGradient.render(text,step,steps(p))))
    }

  private def normalized(text: String): String =
    val spans = VulkanPromptAttention.parse(text)
    if !spans.exists(signed) then text
    else spans.map { span =>
      if span.text == "BREAK" && span.weight == -1 then " BREAK "
      else
        val literal = span.text.flatMap(c => if "\\()[]".contains(c) then s"\\$c" else c.toString)
        if span.weight == 1 then literal
        else s"($literal:${BigDecimal(span.weight).bigDecimal.stripTrailingZeros.toPlainString})"
    }.mkString

  def prepare(p: GenerationParams, prompt: String, encoder: String, encoding: String,
      lines: mutable.Builder[String,Vector[String]]): VulkanPromptFusion.Schedule =
    val count = steps(p)
    val conditions = mutable.LinkedHashMap.empty[String,String]
    def branch(text: String): Vector[String] =
      VulkanFusionPrompt.schedule(text,count,fusionClock = false,gradient = gradient(p),requestedSteps = Some(p.steps)).map {
        case VulkanFusionPrompt.Plan.Leaf(value) =>
          conditions.getOrElseUpdate(value, {
            val name = s"signed_condition${conditions.size}"
            lines += s"$name = encode_prompt($encoder, ${ujson.write(ujson.Str(normalized(value)))}, $encoding, clip_negpip = true)"
            name
          })
        case _ => VulkanProtocol.unsupported("NegPiP and prompt interpolation")
      }
    new VulkanPromptFusion.Schedule(branch(prompt),branch(p.negativePrompt))
