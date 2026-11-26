package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanCdTuner:
  val id = "cd_tuner"
  private val promptTags = """<(?:cdt|cdtc):[^>]+>""".r
  def cleanPrompt(prompt: String): String = promptTags.replaceAllIn(prompt, "")
  private val sdColors = Vector(Vector(-1.0, 1.0 / 3, 2.0 / 3), Vector(1.0, 1.0, 0.0),
    Vector(0.0, -1.0, -1.0), Vector(1.0, 0.0, 1.0))
  private val xlColors = Vector(Vector(0.0, 0.0, 1.0), Vector(1.0, 0.0, 0.0),
    Vector(-1.0, -1.0, 0.0), Vector(-1.0, 1.0, 0.0))

  final case class Weights(detail: Double, detail2: Double, contrast: Double, stop: Double)
  final case class Region(left: Double, top: Double, right: Double, bottom: Double, offsets: Vector[Double])
  final case class ColorMap(regions: Vector[Region], steps: Int, strength: Double):
    def active: Boolean = steps > 0 && strength > 0 && regions.exists(_.offsets.exists(_ != 0))
    def forHires(firstOnly: Boolean): ColorMap =
      val attenuation = if steps < 5 then 1.15 - steps * .15 else .4
      if firstOnly then copy(steps = 1, strength = strength * .25 * attenuation)
      else copy(strength = strength * .25)

  final case class Options(base: Weights, highDetail: Double, highDetail2: Double, highStop: Double,
      highScale: Boolean, shifts: Vector[Double], shiftDelay: Double, saturation: Double, saturation2: Double,
      colorMap: Option[ColorMap], applyOnce: Boolean):
    private def detailOrTone: Boolean =
      Seq(base.detail, base.detail2, base.contrast, highDetail, highDetail2, saturation, saturation2)
        .exists(_ != 0) || shifts.exists(_ != 0)

    def active: Boolean = detailOrTone || colorMap.exists(_.active)

    def weights(areaRatio: Double, baseSteps: Int): Weights =
      if highStop == 0 || areaRatio <= 1 then
        if base.stop < 0 || baseSteps <= base.stop then base.copy(stop = -1)
        else Weights(0, 0, 0, 0)
      else
        val scale = if highScale then areaRatio else 1.0
        val d1 = if highDetail == 0 then base.detail else highDetail
        val d2 = if highDetail2 == 0 then base.detail2 else highDetail2
        Weights(d1 * scale, d2 * scale, base.contrast, highStop)

    def prepareMap(name: String): Vector[String] = colorMap.filter(_.active).toVector.map { c =>
      val values = c.regions.flatMap(r => Vector(r.left, r.top, r.right, r.bottom) ++ r.offsets)
      s"""$name = tensor::vector(${values.mkString("[", ",", "]")}, dtype = "f32")"""
    }

    def args(weights: Weights, name: String, negative: Boolean, colorShift: Boolean,
        high: Boolean = false, colorMapped: Boolean = true): String =
      val map = colorMap.filter(c => colorMapped && c.active).map(c => if high then c.forHires(!detailOrTone) else c).fold("") { c =>
        val sign = if negative then 1 else -1
        s", cdt_regions = $name, cdt_map_steps = ${c.steps}, cdt_map_strength = ${c.strength}, cdt_map_sign = $sign"
      }
      val delta = if !colorShift then "" else shifts.zipWithIndex.map((value, i) => s", cdt_shift$i = $value").mkString +
        s", cdt_shift_delay = $shiftDelay"
      s", cdt_detail = ${weights.detail}, cdt_detail2 = ${weights.detail2}, cdt_contrast = ${weights.contrast}, cdt_stop = ${weights.stop}$map$delta"

    def decoderArgs: String =
      if saturation == 0 && saturation2 == 0 then "" else s", cdt_sat = $saturation, cdt_sat2 = $saturation2"

    def followup(areaRatio: Double, baseSteps: Int): Option[Options] =
      if applyOnce then None
      else
        val grown = areaRatio > 1
        val detail = if grown then weights(areaRatio, baseSteps) else base
        val map = if grown then colorMap.filter(_.active).map(_.forHires(!detailOrTone)) else None
        val next = copy(base = detail, highDetail = 0, highDetail2 = 0, saturation = 0, saturation2 = 0, colorMap = map)
        Option.when(next.active)(next)

  def config(hooks: Map[String, Value], sdxl: Boolean): Option[Options] =
    hooks.get(id).filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).map(parse(_, sdxl)).filter(_.active)

  private def invalid(message: String): Nothing = VulkanProtocol.invalid(s"CD Tuner: $message")

  def parse(raw: Value, sdxl: Boolean): Options =
    val c = raw.obj
    def number(text: String, name: String): Double =
      text.trim.toDoubleOption.filter(_.isFinite).getOrElse(invalid(s"$name must be a finite number"))
    def n(name: String, default: Double = 0): Double = c.get(name).filterNot(_ == ujson.Null).fold(default) {
      case ujson.Num(x) if x.isFinite => x
      case ujson.Str(x) => number(x, name)
      case _ => invalid(s"$name must be a finite number")
    }
    def str(name: String, default: String): String =
      c.get(name).filterNot(_ == ujson.Null).fold(default)(_.strOpt.getOrElse(invalid(s"$name must be text")))
    def bool(name: String): Boolean = c.get(name).exists {
      case ujson.Bool(value) => value
      case ujson.Num(value) => value != 0
      case ujson.Str(value) => Set("true", "yes", "on", "1")(value.trim.toLowerCase(java.util.Locale.ROOT))
      case _ => false
    }
    val table = if sdxl then xlColors else sdColors
    def offsets(tones: Vector[Double]): Vector[Double] =
      Vector.tabulate(3)(channel => table.zip(tones).map((row, tone) => row(channel) * tone * .02).sum)
    def rows(text: String): Vector[String] =
      text.split(if text.contains('|') then "\\|" else ";", -1).toVector
    def spans(weights: Vector[Double]): Vector[(Double, Double)] =
      val total = weights.sum
      if total == 0 || !total.isFinite then invalid("Split Ratio must have a finite, nonzero sum")
      val ends = weights.scanLeft(0.0)((position, weight) => position + weight / total)
      ends.zip(ends.tail)

    val colors = str("colors", "")
    val map = Option.when(colors.nonEmpty) {
      val ratio = rows(str("split_ratio", "1,1").replace(',', ' '))
      val weighted = (if ratio.size == 1 then Vector("1 " + ratio.head) else ratio).map { row =>
        val values = row.split(" ", -1).toVector.map(number(_, "Split Ratio field"))
        if values.size == 1 then values ++ values else values
      }
      val outer = spans(weighted.map(_.head))
      val in = weighted.map(row => spans(row.tail))
      val shades = rows(colors).map { region =>
        val values = region.split(if region.contains(',') then "," else " ", -1).toVector
          .map(number(_, "Color Map tone"))
        if values.size == 3 then offsets(values.sum / 3 +: values)
        else if values.size == 4 then offsets(values)
        else invalid("Color Map regions need three or four tones")
      }
      val cells = outer.zip(in).flatMap { (out, inside) => inside.map(out -> _) }
      if shades.size < cells.size then invalid(s"Color Map needs ${cells.size} regions but only has ${shades.size} tones")
      val vertical = !str("split_by", "Horizontal").contains("Hor")
      val regions = cells.zip(shades).map { case (((a, b), (x, y)), tones) =>
        if vertical then Region(x, a, y, b, tones) else Region(a, x, b, y, tones)
      }
      val steps = c.get("map_stop_step") match
        case Some(ujson.Str(value)) => value.trim.toIntOption.getOrElse(invalid("Mapping Stop Step must be an integer"))
        case _ => n("map_stop_step", 2).toInt
      ColorMap(regions, steps, n("map_strength", 1))
    }
    val shifts = n("con2") * .02 +: offsets(Vector(n("bri"), n("col1"), n("col2"), n("col3")))
    Options(Weights(n("d1"), n("d2"), n("con1"), n("st1", -1)), n("hd1"), n("hd2"), n("st2", -1),
      bool("hrs"), shifts, n("dis"), n("sat"), n("sat2"), map, bool("apply_once"))
