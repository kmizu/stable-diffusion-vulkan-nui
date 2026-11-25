package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] object VulkanLatentUpscale:
  case class Config(method: String, scheduler: String, fullSteps: Int)

  def parse(p: GenerationParams, h: Value, scheduler: String): Config =
    val strength = p.img2img.getOrElse(VulkanProtocol.invalid("Latent Upscale needs an img2img input image")).denoisingStrength
    require(strength.isFinite && strength >= p.steps/10000.0 && strength <= 1,"Invalid Latent Upscale denoising strength (schedule limit 10000)")
    val full = math.max(p.steps,math.ceil(p.steps/strength - 1e-9).toInt)
    val method = h.obj.get("method").fold("nearest-exact")(_.str)
    if Set("linear","trilinear")(method) then VulkanProtocol.unsupported(s"$method on 4-D image latents")
    require(Set("bilinear","bilinear-antialiased","bicubic","bicubic-antialiased","area","nearest","nearest-exact")(method),"Unknown latent interpolation method")
    val sched = h.obj.get("scheduler").fold("simple")(_.str) match
      case "simple" => "Simple"
      case "normal" => "Normal"
      case "karras" => "Karras"
      // The extension uses rho=1, which makes polyexponential the exponential schedule.
      case "exponential" | "polyexponential" => "Exponential"
      case "automatic" => scheduler
      case _ => VulkanProtocol.invalid("Unknown Latent Upscale scheduler")
    Config(method,sched,full)

  def resize(method: String, sw: Int, sh: Int, w: Int, h: Int): Vector[String] =
    if method != "area" then
      val mode = method.stripSuffix("-antialiased").replace('-','_')
      Vector(s"""base = tensor::interpolate(encoded, size = [$w,$h], mode = "$mode", antialias = ${method.endsWith("-antialiased")})""")
    else
      // PyTorch's area interpolation uses the mean of each adaptive pooling bin.
      val lines = Vector.newBuilder[String]
      var input = "encoded"
      for (source,target,axis) <- Seq((sw,w,0),(sh,h,1)) if source != target do
        val starts = Vector.tabulate(target)(i => i*source/target)
        val lengths = starts.indices.map(i => ((i+1)*source+target-1)/target - starts(i))
        val shape = if axis == 0 then s"[$w,1,1,1]" else s"[1,$h,1,1]"
        var sum = ""
        for j <- 0 until lengths.max do
          val indices = starts.indices.map(i => (starts(i)+j).min(source-1)).mkString("[",",","]")
          val weights = lengths.map(n => if j < n then 1.0/n else 0.0).mkString("[",",","]")
          val term = s"area${axis}t$j"
          lines += s"$term = tensor::mul(tensor::index_select($input, axis = $axis, indices = $indices), tensor::reshape(tensor::vector($weights), $shape))"
          if j == 0 then sum = term
          else
            val next = s"area${axis}s$j"
            lines += s"$next = tensor::add($sum, $term)"
            sum = next
        input = sum
      lines += s"base = $input"
      lines.result()
