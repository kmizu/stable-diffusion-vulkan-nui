package sdfx.vulkan

import scala.collection.mutable
import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] object VulkanNeutralPrompt:
  val Keywords: Vector[String] = Vector("AND_PERP", "AND_SALT", "AND_TOPK", "AND")
  private val KeywordRegex = "\\b(AND_PERP|AND_SALT|AND_TOPK|AND)\\b".r

  sealed trait Node:
    def kind: String
    def weight: Double

  final case class Leaf(text: String, weight: Double = 1.0, kind: String = "AND") extends Node
  final case class Group(children: Vector[Node], weight: Double = 1.0, kind: String = "AND") extends Node

  def parse(prompt: String): Group = parse(prompt, identity)

  def parse(prompt: String, maskSyntax: String => String): Group =
    val parts = splitTopKeywords(prompt, maskSyntax)
    val children = parts.map { case (kind, rawSegment) =>
      parseSegment(rawSegment, kind, maskSyntax)
    }
    Group(children, 1.0, "AND")

  def leaves(group: Group): Vector[Leaf] =
    val buf = mutable.ArrayBuffer.empty[Leaf]
    def visit(n: Node): Unit = n match
      case l: Leaf => buf += l
      case g: Group => g.children.foreach(visit)
    visit(group)
    buf.toVector

  def isActive(prompt: String, cfgRescale: Double): Boolean =
    val tree = parse(prompt)
    val lvs = leaves(tree)
    cfgRescale != 0.0 || lvs.size != 1 || lvs.exists(n => n.weight != 1.0 || n.kind != "AND") || lvs.head.text != prompt

  private def splitTopKeywords(text: String, maskSyntax: String => String): Vector[(String, String)] =
    val masked = maskSyntax(text)
    val matches = KeywordRegex.findAllMatchIn(masked).filter { m =>
      isTopLevel(masked, m.start)
    }.toVector
    if matches.isEmpty then Vector(("AND", text))
    else
      val out = mutable.ArrayBuffer.empty[(String, String)]
      var curKind = "AND"
      var curStart = 0
      for m <- matches do
        out += ((curKind, text.substring(curStart, m.start)))
        curKind = m.group(1)
        curStart = m.end
      out += ((curKind, text.substring(curStart)))
      out.toVector

  private def isTopLevel(text: String, pos: Int): Boolean =
    var depth = 0
    var i = 0
    while i < pos do
      val c = text.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' || c == '(' || c == '<' then depth += 1
      else if (c == ']' || c == ')' || c == '>') && depth > 0 then depth -= 1
      i += 1
    depth == 0

  private def parseSegment(raw: String, kind: String, maskSyntax: String => String): Node =
    val trimmed = raw.trim
    val (body, weight) = splitTrailingWeight(trimmed, maskSyntax)
    val bTrim = body.trim
    if bTrim.startsWith("[") && bTrim.endsWith("]") && isMatchingOuterBracket(maskSyntax(bTrim)) then
      val inner = bTrim.substring(1, bTrim.length - 1)
      val subParts = splitTopKeywords(inner, maskSyntax)
      if subParts.size > 1 || splitTrailingWeight(inner.trim, maskSyntax)._2 != 1.0 then
        val children = subParts.map { case (k, seg) => parseSegment(seg, k, maskSyntax) }
        return Group(children, weight, kind)
    Leaf(if weight != 1.0 then body.trim else raw.trim, weight, kind)

  private def isMatchingOuterBracket(s: String): Boolean =
    var depth = 0
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' then depth += 1
      else if c == ']' then
        depth -= 1
        if depth == 0 && i < s.length - 1 then return false
      i += 1
    depth == 0

  private def splitTrailingWeight(s: String, maskSyntax: String => String): (String, Double) =
    val masked = maskSyntax(s)
    var depth = 0
    var lastColon = -1
    var i = 0
    while i < s.length do
      val c = masked.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' || c == '(' || c == '<' then depth += 1
      else if (c == ']' || c == ')' || c == '>') && depth > 0 then depth -= 1
      else if c == ':' && depth == 0 then lastColon = i
      i += 1
    if lastColon < 0 then (s, 1.0)
    else
      val rhs = s.substring(lastColon + 1).trim
      rhs.toDoubleOption match
        case Some(w) =>
          require(w.isFinite, "Neutral Prompt weight must be finite")
          (s.substring(0, lastColon), w)
        case None => (s, 1.0)

  val functions = """fn neutral_abs(x) {
    return tensor::maximum(x,tensor::mul(x,-1))
  }
  fn neutral_safe(v) {
    return tensor::to_list(tensor::clamp(tensor::vector([v],dtype = "f32"),min = 1.0e-30,max = 1.0e30),element = "number")[0]
  }
  fn neutral_perp(normal, v) {
    denom = tensor::stat_f32(tensor::mul(normal,normal),mode = "mean")
    dot = tensor::stat_f32(tensor::mul(normal,v),mode = "mean")
    return tensor::add(v,tensor::mul(normal,-dot / neutral_safe(denom)))
  }
  fn neutral_topk(x) {
    values = neutral_abs(x)
    rank = math::floor(tensor::numel(values) * 0.95)
    threshold = tensor::stat_f32(values,mode = "kth",rank = rank)
    return tensor::mul(x,tensor::threshold(values,threshold = threshold))
  }
  fn neutral_rescale(c,g,phi) {
    cm = tensor::stat_f32(c,mode = "mean")
    gm = tensor::stat_f32(g,mode = "mean")
    cs = tensor::stat_f32(c,mode = "std")
    gs = tensor::stat_f32(g,mode = "std")
    corrected = tensor::add(tensor::mul(tensor::add(g,-gm),cs / neutral_safe(gs)),cm)
    return tensor::lerp(g,corrected,phi)
  }
  """

  def sample(p: GenerationParams,prompt: String,h: Value,text: String,enc: String): Seq[String] =
    if VulkanProtocol.samplers(p.samplerName)!="euler" then VulkanProtocol.unsupported("Neutral Prompt currently requires Euler")
    val phi=h.obj.get("cfg_rescale").fold(0.0)(_.num)
    require(phi.isFinite && phi>=0 && phi<=1,"Neutral Prompt cfg_rescale must be in 0..1")
    val tree=parse(prompt)
    val leafNodes=leaves(tree)
    require(leafNodes.size<=64,"Neutral Prompt supports up to 64 leaves")
    val lines=Vector.newBuilder[String]
    for (leaf,i)<-leafNodes.zipWithIndex do
      lines+=s"neutral_leaf$i = predictor(unet,encode_prompt($text,${ujson.write(ujson.Str(leaf.text))},$enc))"
    val steps=p.img2img.fold(p.steps)(i=>math.min(p.steps,(p.steps*i.denoisingStrength).toInt+1))
    var input="start"
    var serial=0
    def put(expr: String): String =
      serial+=1
      val key=s"neutral_value$serial"
      lines+=s"$key = $expr"
      key
    for step<-0 until steps do
      val sigma=s"sigmas[$step]"
      val interval=put(s"tensor::vector([$sigma,0],dtype = \"f32\")")
      def prediction(model: String)=put(s"sample($input,$interval,predict = $model,method = \"euler\",rng = rng)")
      val uncond=prediction("neg")
      val branches=leafNodes.indices.map(i=>put(s"tensor::add(${prediction(s"neutral_leaf$i")},tensor::mul($uncond,-1))")).iterator
      def combine(n: Node): String = n match
        case _: Leaf => branches.next()
        case group: Group =>
          val children=group.children.map(node=>(node,combine(node)))
          val zero=put(s"tensor::zeros_like($uncond)")
          val normal=children.filter(_._1.kind=="AND").foldLeft(zero){ case (out,(node,v)) =>
            put(s"tensor::add($out,tensor::mul($v,${node.weight}))")
          }
          var out=normal
          for (node,v)<-children if Set("AND_PERP","AND_TOPK")(node.kind) do
            val select=if node.kind=="AND_PERP" then s"neutral_perp($normal,$v)" else s"neutral_topk($v)"
            out=put(s"tensor::add($out,tensor::mul($select,${node.weight}))")
          val salient=children.filter(_._1.kind=="AND_SALT")
          if salient.nonEmpty then
            var score=put(s"tensor::salience_f32($normal)")
            var picked=zero
            for (node,v)<-salient do
              val next=put(s"tensor::salience_f32($v)")
              val diff=put(s"tensor::add($next,tensor::mul($score,-1))")
              val wins=put(s"tensor::threshold(tensor::mul($diff,-1),threshold = 0,low = 1,high = 0)")
              val replacement=put(s"tensor::mul(tensor::add($v,tensor::mul($normal,-1)),${node.weight})")
              picked=put(s"tensor::lerp($picked,$replacement,$wins)")
              score=put(s"tensor::maximum($next,$score)")
            out=put(s"tensor::add($out,$picked)")
          out
      val combined=combine(tree)
      val guided=put(s"tensor::add($uncond,tensor::mul($combined,${p.cfgScale}))")
      val x0=if phi==0 then guided else
        val conditional=put(s"tensor::add($uncond,$combined)")
        put(s"neutral_rescale($conditional,$guided,$phi)")
      input=put(s"tensor::lerp($x0,$input,sigmas[${step+1}] / $sigma)")
    lines+=s"latent = $input"
    lines.result()
