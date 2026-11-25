package sdfx.vulkan

import org.apache.commons.math3.distribution.NormalDistribution
import ujson.Value

private[vulkan] object VulkanSemanticGuidance:
  def prepare(h: Value,text: String,enc: String,sdxl: Boolean,lines: scala.collection.mutable.Builder[String,Vector[String]]): String =
    def num(k: String,d: Double) =
      val value=h.obj.get(k).fold(d)(_.num)
      require(value.isFinite,s"Invalid Semantic Guidance $k")
      value
    val warmup=num("warmup",10).toInt
    val scale=num("edit_guidance_scale",1)
    val tail=num("tail_percentage_threshold",0.05)
    num("momentum_scale",0.3)
    num("momentum_beta",0.6)
    require(warmup>=0 && tail>=0 && tail<=1,"Invalid Semantic Guidance warmup or tail fraction")
    val concepts=for
      (key,sign)<-Vector("prompt"->1.0,"negative_prompt"-> -1.0)
      prompt<-h.obj.get(key).flatMap(_.strOpt).filter(_.nonEmpty).toVector
      part<-prompt.split(",",-1)
    yield
      val span=VulkanPromptAttention.parse(part.trim).head
      (span.text,span.weight*sign)
    if concepts.isEmpty || scale==0 || tail==0 then return "pos"
    val z=if tail==1 then 0.0 else new NormalDistribution().inverseCumulativeProbability(1-tail)
    var cross="latent_pos.crossattn"
    var vector="latent_pos.vector"
    for ((caption,strength),idx)<-concepts.zipWithIndex do
      val c=s"semantic$idx"
      lines+=s"$c = encode_prompt($text,${ujson.write(ujson.Str(caption))},$enc)"
      def edit(field: String,positive: String): String =
        val name=c+field
        val v=s"$c.$field"
        val negative=s"latent_neg.$field"
        lines+=s"${name}_delta = tensor::mul(tensor::add($v,tensor::mul($negative,-1)),$strength)"
        val delta=if tail==1 then s"${name}_delta" else
          lines+=s"${name}_mean = cads_mean($v)"
          lines+=s"${name}_centered = tensor::add($v,-${name}_mean)"
          lines+=s"${name}_variance = cads_mean(tensor::mul(${name}_centered,${name}_centered)) * tensor::numel($v) / (tensor::numel($v) - 1)"
          lines+=s"${name}_threshold = ${name}_mean + $z * cads_sqrt(${name}_variance)"
          lines+=s"${name}_abs = tensor::maximum(${name}_delta,tensor::mul(${name}_delta,-1))"
          lines+=s"${name}_mask = tensor::threshold(${name}_abs,threshold = ${name}_threshold)"
          lines+=s"${name}_edited = tensor::mul(${name}_delta,${name}_mask)"
          s"${name}_edited"
        lines+=s"$name = tensor::add($positive,tensor::mul($delta,$scale))"
        name
      cross=edit("crossattn",cross)
      if sdxl then vector=edit("vector",vector)
    val pooled=if sdxl then s",vector = $vector" else ""
    lines+=s"semantic_model = predictor(unet,{crossattn = $cross$pooled})"
    if warmup==0 then "semantic_model"
    else
      lines+=s"semantic_switch = switch_predictor(pos,semantic_model,after_steps = $warmup)"
      "semantic_switch"
