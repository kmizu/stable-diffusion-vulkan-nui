package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

/** A one-step Euler integrator can be split at each supplied sigma without losing solver history. */
private[vulkan] object VulkanLatentGraph:
  val hooks = Set("latent_mirroring", "resharpen", "anti_burn", "color_grading", "vectorscope_cc", "detail_daemon", "cads", "cfg_cache", "mahiro", "rescale_cfg", "dynamic_thresholding")

  def sample(p: GenerationParams, method: String, eta: String, sdxl: Boolean, seed: Long): Seq[String] =
    if !Set("euler","euler_a")(method) then VulkanProtocol.unsupported("per-step latent hooks require Euler or Euler a")
    val lines = Vector.newBuilder[String]
    def config(k: String) = p.hooks.get(k).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
    def num(h: Value,k: String,default: Double,low: Double,high: Double): Double =
      val v = h.obj.get(k).fold(default)(_.num)
      require(v.isFinite && v >= low && v <= high,s"Invalid $k")
      v
    val total = p.img2img.fold(p.steps)(i => math.min(p.steps,(p.steps*i.denoisingStrength).toInt+1))
    val detail = config("detail_daemon").map(new VulkanDetailDaemon(_,total))
    val cads = config("cads").map(new VulkanCads(_,total,sdxl))
    if cads.nonEmpty then lines += s"cads_rng = make_rng(\"cpu\", seed = ${seed & 0xffffffffL})"
    if cads.nonEmpty && detail.nonEmpty then VulkanProtocol.unsupported("CADS with Detail Daemon")
    val guidance = new VulkanGuidance(p,total)
    if guidance.enabled && method != "euler" then VulkanProtocol.unsupported("CFG rescale and CFG cache require Euler")
    val threshold = config("dynamic_thresholding").filter(_ => p.cfgScale > 1).map(new VulkanDynamicThreshold(_, p))
    if threshold.nonEmpty && (method != "euler" || guidance.enabled) then
      VulkanProtocol.unsupported("Dynamic Thresholding requires Euler without other CFG modifiers")
    val burn = config("anti_burn")
    val skip = burn.fold(0)(h => num(h,"skip_last",2,0,50).toInt)
    val count = burn.fold(1)(h => num(h,"average",3,1,50).toInt)
    val stop = math.max(1,total-skip)
    val width = p.width/8
    val height = p.height/8
    def indices(n: Int) = (0 until n).reverse.mkString("[",",","]")
    val vectorscope = config("vectorscope_cc").map(new VulkanVectorscope(_,sdxl,width,height,total))
    if vectorscope.nonEmpty then lines += s"color_rng = make_rng(\"cpu\", seed = ${(seed+17) & 0xffffffffL})"
    val grading = config("color_grading").map(new VulkanColorGrading(_,sdxl,width,height,total))
    val mirroring = config("latent_mirroring").map { h =>
      val mode = h.obj.get("mode").fold("Alternate Steps")(_.str)
      val style = h.obj.get("style").fold("Vertical Mirroring")(_.str)
      require(Set("Alternate Steps","Blend Average")(mode),"Invalid mirroring mode")
      require(Set("Vertical Mirroring","Horizontal Mirroring","Horizontal+Vertical Mirroring","90 Degree Rotation","180 Degree Rotation")(style),"Invalid mirroring style")
      (mode,style,num(h,"strength",1,0,1),num(h,"max_step",0.25,0,1))
    }
    val resharpen = config("resharpen").map { h =>
      val curve = h.obj.get("scaling").fold("Flat")(_.str).replace(" ","").toLowerCase(java.util.Locale.ROOT)
      require(Set("flat","sin","cos","1-sin","1-cos")(curve),"Invalid ReSharpen scaling")
      (num(h,"sharpness",0.5,-4,4),curve)
    }
    var last = "start"
    val history = Vector.newBuilder[String]
    for n <- 1 to stop do
      var value = s"x$n"
      val branches = cads.fold(("pos","neg"))(_.branches(n,lines))
      val (pos,neg) = detail.fold(branches)(_.branches(n,lines,"pos","neg"))
      val guided = if pos == "pos" && neg == "neg" then "guided" else
        lines += s"guided$n = cfg($pos, $neg, scale = ${p.cfgScale})"
        s"guided$n"
      threshold match
        case Some(dynamic) => dynamic.step(lines, n, last, value, pos, neg)
        case None if guidance.enabled => guidance.step(lines,n,last,value,pos,neg)
        case None => lines += s"$value = sample($last, sigmas[${n-1}:${n+1}], predict = $guided, method = \"$method\", rng = rng$eta)"
      vectorscope.foreach(v => value = v.append(value,n,lines))
      for (mode,style,strength,maxStep) <- mirroring if n <= total*maxStep && (mode == "Blend Average" || n%2 == 1) do
        def flip(axis: Int,input: String) = s"tensor::index_select($input, axis = $axis, indices = ${indices(if axis==0 then width else height)})"
        val mirrored = style match
          case "Vertical Mirroring" => flip(0,value)
          case "Horizontal Mirroring" => flip(1,value)
          case "90 Degree Rotation" if width==height => flip(1,s"tensor::permute($value, [1,0,2,3])")
          case _ => flip(0,flip(1,value))
        val factor = if mode == "Blend Average" then strength/2 else strength
        lines += s"mirror$n = tensor::lerp($value, $mirrored, $factor)"
        value = s"mirror$n"
      for (sharpness,curve) <- resharpen if n > 1 && n < total do
        val angle = math.Pi*n/(2*total)
        val k = curve match
          case "sin" => math.sin(angle)
          case "cos" => math.cos(angle)
          case "1-sin" => 1-math.sin(angle)
          case "1-cos" => 1-math.cos(angle)
          case _ => 1.0
        lines += s"sharp$n = tensor::lerp($value, $last, ${-sharpness*0.1*k})"
        value = s"sharp$n"
      grading.foreach(g => value = g.append(lines,value,n))
      history += value
      last = value
    val used = history.result().takeRight(count)
    if used.size == 1 then lines += s"latent = $last"
    else
      var sum = used.head
      used.tail.zipWithIndex.foreach { (v,i) =>
        lines += s"sum$i = tensor::add($sum, $v)"
        sum = s"sum$i"
      }
      lines += s"latent = tensor::div($sum, ${used.size})"
    lines.result()
