package sdfx.vulkan

import ujson.Value

private[vulkan] final class VulkanVectorscope(config: Value, sdxl: Boolean, width: Int, height: Int, total: Int):
  private def number(key: String, default: Double, min: Double, max: Double): Double =
    val n = config.obj.get(key).fold(default)(_.num)
    require(n.isFinite && n >= min && n <= max,s"Invalid Vectorscope $key")
    n
  private val method = config.obj.get("method").fold("Straight")(_.str)
  require(Set("Straight","Cross","N.Random","U.Random")(method),"Invalid Vectorscope noise method")
  private val curve = config.obj.get("scaling").fold("Flat")(_.str).replace(" ","").toLowerCase(java.util.Locale.ROOT)
  require(Set("flat","cos","sin","1-cos","1-sin")(curve),"Invalid Vectorscope scaling")
  private val brightness = number("brightness",0,-5,5)
  private val rgb = Seq("r","g","b").map(k => brightness+number(k,0,-5,5))
  private val color = VulkanLatentMath.rgb(sdxl,rgb(0),rgb(1),rgb(2))
  private val luma = VulkanLatentMath.rgb(sdxl,1,1,1)
  private val normal = math.sqrt(luma.map(x => x.toDouble*x).sum)
  private val contrast = number("contrast",0,-5,5)
  private val saturation = number("saturation",0,-5,5)
  private val strength = number("strength",1,0,5)

  def append(input: String,n: Int,lines: scala.collection.mutable.Builder[String,Vector[String]]): String =
    def put(label: String,expr: String): String =
      val name = s"vec${n}_$label"
      lines += s"$name = $expr"
      name
    def vector(values: Seq[Double]): String =
      s"tensor::reshape(tensor::vector(${values.mkString("[",",","]")}, dtype = \"f32\"), [1,1,4,1])"
    val angle = n.toDouble/total*math.Pi/2
    val k = curve match
      case "cos" => math.cos(angle)
      case "sin" => math.sin(angle)
      case "1-cos" => 1-math.cos(angle)
      case "1-sin" => 1-math.sin(angle)
      case _ => 1.0
    val a = strength*k/total
    var out = input
    if color.exists(_!=0) then
      val base = put("offset",vector(color.toVector.map(_.toDouble)))
      val offset = method match
        case "Straight" => base
        case "Cross" =>
          val abs = put("abs",s"tensor::maximum($input,tensor::mul($input,-1))")
          val m = VulkanLatentMath.mean(abs,width,height,"cross",put)
          put("adjust",s"tensor::mul($base,$m)")
        case _ =>
          val z = put("noise",s"tensor::randn_like($input, rng = color_rng)")
          val f = if method == "U.Random" then s"tensor::maximum($z,tensor::mul($z,-1))" else s"tensor::add(1,tensor::mul($z,0.5))"
          put("adjust",s"tensor::mul($base,$f)")
      out = put("colored",s"tensor::add($out,tensor::mul($offset,${a*0.15}))")
    if contrast != 0 then
      val m = VulkanLatentMath.mean(out,width,height,"contrast",put)
      out = put("contrast",s"tensor::lerp($m,$out,${1+a*0.08*contrast})")
    if saturation != 0 then
      val l = put("luma",vector(luma.toVector.map(_/normal)))
      val projected = put("projected",s"tensor::mul($out,$l)")
      def slice(i: Int) = s"tensor::crop($projected, axes = [2], start = [$i], size = [1])"
      val dot = put("dot",s"tensor::add(tensor::add(${slice(0)},${slice(1)}),tensor::add(${slice(2)},${slice(3)}))")
      out = put("saturated",s"tensor::lerp(tensor::mul($dot,$l),$out,${1+a*0.08*saturation})")
    out
