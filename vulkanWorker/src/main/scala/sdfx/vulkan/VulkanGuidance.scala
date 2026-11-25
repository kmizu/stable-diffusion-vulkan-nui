package sdfx.vulkan

import sdfx.core.model.GenerationParams

/** Evaluates both standard model branches and applies CFG in native prediction space. */
private[vulkan] final class VulkanGuidance(p: GenerationParams, steps: Int):
  private type Lines = scala.collection.mutable.Builder[String,Vector[String]]
  private val rescale = p.hooks.get("rescale_cfg").filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
    .map(_.obj.get("multiplier").fold(0.7)(_.num))
  require(rescale.forall(n => n.isFinite && n >= 0 && n <= 1),"Invalid CFG rescale multiplier")
  private val cache = p.hooks.get("cfg_cache").filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
  private val start = cache.flatMap(_.obj.get("start")).fold(0.25)(_.num)
  private val interval = cache.flatMap(_.obj.get("interval")).fold(3)(_.num.toInt)
  require(start.isFinite && start >= 0 && start <= 1 && interval >= 2 && interval <= 20,"Invalid CFG cache schedule")
  if cache.exists(v => Seq("adaptive_threshold","hf_weight").exists(k => v.obj.get(k).exists(_.num != 0))) then
    VulkanProtocol.unsupported("adaptive CFG cache or high-frequency boosting")
  private val mahiro = p.hooks.get("mahiro").exists(!_.obj.get("enabled").contains(ujson.Bool(false)))
  val enabled = (mahiro || rescale.nonEmpty || cache.nonEmpty) && p.cfgScale > 1
  private var lastFull = -1
  private val warmup = math.max(1,math.ceil(steps*start).toInt)

  /** Reduce all four latent channels and spatial axes by pairs without materializing scalar lists. */
  private def sum(value: String, length: Int, label: String, lines: Lines): String =
    var size = length
    var tensor = s"${label}flat"
    lines += s"$tensor = tensor::reshape($value, [$size])"
    while size > 1 do
      val half = size/2
      val result = s"${label}_$size"
      val combined = s"tensor::add($tensor[:$half], $tensor[$half:${2*half}])"
      val expr = if size%2==0 then combined else s"tensor::concat([$combined,$tensor[${2*half}:]], axis = 0)"
      lines += s"$result = $expr"
      tensor = result
      size = (size+1)/2
    tensor

  def step(lines: Lines, n: Int, input: String, output: String, posPredictor: String, negPredictor: String): Unit =
    val step = n-1
    def put(label: String, expr: String): String =
      val name = s"cfg${n}_$label"
      lines += s"$name = $expr"
      name
    val sigma = s"sigmas[$step]"
    val toZero = put("range",s"tensor::vector([$sigma,0], dtype = \"f32\")")
    val scales = put("scales",s"denoiser_scalings(pos, $sigma)")
    val skip = put("skip",s"tensor::mul($input,$scales.skip_scale)")
    def raw(branch: String, pred: String): String =
      val denoised = put(branch+"x0",s"sample($input, $toZero, predict = $pred, method = \"euler\", rng = rng)")
      put(branch,s"tensor::div(tensor::add($denoised,tensor::mul($skip,-1)), $scales.output_scale)")
    val pos = raw("pos",posPredictor)
    if cache.isEmpty || lastFull < 0 || step < warmup || step-lastFull >= interval then
      val neg = raw("neg",negPredictor)
      put("delta",s"tensor::add($pos,tensor::mul($neg,-1))")
      lastFull = step
    val delta = s"cfg${lastFull+1}_delta"
    val cfg = put("guided",s"tensor::add($pos,tensor::mul($delta,${p.cfgScale-1}))")
    def sqrt(v: String,label: String): String =
      val safe = put(label+"safe",s"tensor::clamp($v,min = 0.0000000000000001,max = 1.0e30)")
      var root = put(label,s"tensor::add($safe,1)")
      for j <- 0 until 20 do root = put(s"$label$j",s"tensor::mul(tensor::add($root,tensor::div($safe,$root)),0.5)")
      root
    val guided = if !mahiro then cfg else
      def toX0(v: String) = s"tensor::add($skip,tensor::mul($v,$scales.output_scale))"
      def abs(v: String,label: String) = put(label,s"tensor::maximum($v,tensor::mul($v,-1))")
      def channels(v: String,label: String): String =
        def slice(k: Int) = s"tensor::crop($v, axes = [2], start = [$k], size = [1])"
        put(label,s"tensor::add(tensor::add(${slice(0)},${slice(1)}),tensor::add(${slice(2)},${slice(3)}))")
      val neg = put("mahNeg",s"tensor::add($pos,tensor::mul($delta,-1))")
      val leap = put("leap",s"tensor::mul(${toX0(pos)},${p.cfgScale})")
      val uLeap = put("uLeap",s"tensor::mul(${toX0(neg)},${p.cfgScale})")
      val cfgX0 = put("cfgX0",toX0(cfg))
      val merge = put("merge",s"tensor::lerp($leap,$cfgX0,0.5)")
      val absU = abs(uLeap,"absU")
      val absM = abs(merge,"absM")
      val u = put("normU",s"tensor::div($uLeap,${sqrt(absU,"sqrtU")})")
      val m = put("normM",s"tensor::div($merge,${sqrt(absM,"sqrtM")})")
      val dot = channels(s"tensor::mul($u,$m)","dot")
      val den = put("den",s"tensor::mul(${sqrt(channels(absU,"sqU"),"lenU")},${sqrt(channels(absM,"sqM"),"lenM")})")
      val cos = put("cos",s"tensor::div($dot,$den)")
      val plane = (p.width/8)*(p.height/8)
      val summed = sum(cos,plane,s"cfg${n}cos",lines)
      val weight = put("weight",s"tensor::add(tensor::mul($summed,${0.5/plane}),0.5)")
      val guidedX0 = put("mahiro",s"tensor::lerp($leap,$cfgX0,$weight)")
      put("mahiroRaw",s"tensor::div(tensor::add($guidedX0,tensor::mul($skip,-1)), $scales.output_scale)")
    val prediction = rescale.filter(_ > 0).fold(guided) { amount =>
      val length = (p.width/8)*(p.height/8)*4
      def variance(v: String, label: String): String =
        val total = sum(v,length,s"cfg${n}${label}mean",lines)
        val centered = put(label+"center",s"tensor::add($v,tensor::mul($total,${-1.0/length}))")
        val squared = put(label+"squared",s"tensor::mul($centered,$centered)")
        sum(squared,length,s"cfg${n}${label}var",lines)
      val c = variance(pos,"cond")
      val g = variance(guided,"cfg")
      // The denominator for unbiased variance cancels in the ratio.
      val ratio = put("ratio",s"tensor::div($c,tensor::clamp($g,min = 0.00000001, max = 1.0e30))")
      var root = put("root",s"tensor::add($ratio,1)")
      for j <- 0 until 16 do root = put(s"root$j",s"tensor::mul(tensor::add($root,tensor::div($ratio,$root)),0.5)")
      put("rescaled",s"tensor::mul($guided,tensor::add(tensor::mul($root,$amount),${1-amount}))")
    }
    val x0 = put("x0",s"tensor::add($skip,tensor::mul($prediction,$scales.output_scale))")
    lines += s"$output = tensor::lerp($x0,$input,sigmas[$n] / $sigma)"
