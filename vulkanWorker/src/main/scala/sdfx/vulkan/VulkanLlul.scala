package sdfx.vulkan

import java.util.Locale
import ujson.Value

private[vulkan] object VulkanLlul:
  val id = "llul"
  private val layerBits = Map("resblock" -> 1, "transformer" -> 2, "s. attn." -> 4, "x. attn." -> 8, "out" -> 16)
  private val upModes = Map("nearest" -> 0, "bilinear" -> 1, "bicubic" -> 2)
  private val downModes = upModes ++ Map("pooling avg" -> 3, "pooling max" -> 4, "area" -> 5)
  // A superset of the labels shared by SD1.x, SD2.x and the SDXL U-Nets.
  private val blockLabels = (0 to 11).flatMap(i => Vector(f"IN$i%02d_00", f"OUT$i%02d_00")) ++
    Vector("M00_00", "M00_01")
  private val labelsByKind = Vector(
    1 -> blockLabels.map(_ + "_resblock"),
    2 -> blockLabels.map(_ + "_trans"),
    4 -> blockLabels.flatMap(label => (0 to 9).map(i => f"${label}_$i%02d_sattn")),
    8 -> blockLabels.flatMap(label => (0 to 9).map(i => f"${label}_$i%02d_xattn"))
  )

  final case class Config(weight: Double, factor: Int, x: Option[Double], y: Option[Double],
      filters: Vector[String], layerKinds: Int, start: Int, end: Int, up: Int, down: Int,
      upAA: Boolean, downAA: Boolean, spherical: Boolean, fp32: Boolean, mask: Option[String]):
    def active: Boolean = weight != 0 && factor > 1 && layerKinds != 0 && (end == 0 || start <= end)

    def endAt(scheduleSteps: Int): Int = if end == 0 then scheduleSteps else end

    def canApply(scheduleSteps: Int): Boolean =
      active && start <= endAt(scheduleSteps) &&
        ((layerKinds & 16) != 0 || filters.isEmpty || labelsByKind.exists { (bit, labels) =>
          (layerKinds & bit) != 0 && labels.exists(label => filters.exists(label.contains))
        })

    def origin(width: Int, height: Int): (Double, Double) =
      if width < 128 || height < 128 then VulkanProtocol.invalid("LLuL requires width and height of at least 128")
      def axis(explicit: Option[Double], size: Int): Double =
        explicit.getOrElse(((size - size / factor) / 2).toDouble) / size
      (axis(x, width), axis(y, height))

    def args(width: Int, height: Int, maskName: String, scheduleSteps: Int): String =
      val (xFraction, yFraction) = origin(width, height)
      val layerNames = ujson.write(ujson.Arr.from(filters))
      val masking = mask.fold("")(_ => s", llul_mask = $maskName")
      s", llul_kinds = $layerKinds, llul_factor = $factor, llul_weight = $weight, llul_x = $xFraction, llul_y = $yFraction" +
        s", llul_filters = $layerNames, llul_start = $start, llul_end = ${endAt(scheduleSteps)}, llul_up = $up, llul_down = $down" +
        s", llul_up_aa = $upAA, llul_down_aa = $downAA, llul_slerp = $spherical, llul_fp32 = $fp32$masking"

  def parse(value: Value): Config =
    import VulkanProtocol.invalid
    val fields = value.obj
    val fieldsAllowed = Set("enabled", "weight", "multiply", "x", "y", "layers", "apply_to", "start_steps",
      "max_steps", "up", "up_aa", "down", "down_aa", "intp", "force_float", "mask")
    if !fields.keysIterator.forall(fieldsAllowed) then invalid("Unknown LLuL option")
    def number(key: String, value: Value): Double =
      val result = value.numOpt.getOrElse(Double.NaN)
      if !result.isFinite || !result.toFloat.isFinite then invalid(s"LLuL $key must be a finite number")
      result
    def integer(key: String, default: Int, low: Int, high: Int = Int.MaxValue): Int =
      val result = fields.get(key).fold(default.toDouble)(number(key, _))
      if !result.isWhole || result < low || result > high then invalid(s"Invalid LLuL $key")
      result.toInt
    def text(key: String, default: String): String =
      fields.get(key).fold(default)(_.strOpt.getOrElse(invalid(s"LLuL $key must be text"))).trim
    def flag(key: String): Boolean =
      fields.get(key).fold(false) {
        case ujson.Bool(enabled) => enabled
        case ujson.Str(enabled) if enabled.equalsIgnoreCase("true") => true
        case ujson.Str(enabled) if enabled.equalsIgnoreCase("false") => false
        case _ => invalid(s"LLuL $key must be true or false")
      }
    def list(key: String, fallback: Vector[String]): Vector[String] =
      fields.get(key).fold(fallback) {
        case ujson.Null if key == "layers" => Vector.empty
        case ujson.Str(s) if s.trim.isEmpty && key == "layers" => Vector.empty
        case ujson.Str(s) => s.split(",", -1).map(_.trim).toVector
        case a: ujson.Arr => a.value.map(_.strOpt.getOrElse(invalid(s"LLuL $key needs text entries")).trim).toVector
        case _ => invalid(s"Invalid LLuL $key")
      }
    def position(key: String): Option[Double] = fields.get(key).flatMap {
      case ujson.Null | ujson.Str("") => None
      case p => Some(number(key, p))
    }
    def mode(key: String, choices: Map[String, Int], aa: Boolean): Int =
      val name = text(key, "bilinear").toLowerCase(Locale.ROOT)
      if aa && name != "bilinear" && name != "bicubic" then invalid(s"LLuL $key antialiasing requires bilinear or bicubic")
      choices.getOrElse(name, invalid(s"Unknown LLuL $key scaler"))
    val kinds = list("apply_to", Vector("out")).map(k =>
      layerBits.getOrElse(k.toLowerCase(Locale.ROOT), invalid(s"Unknown LLuL layer kind $k"))).foldLeft(0)(_ | _)
    val upAA = flag("up_aa")
    val downAA = flag("down_aa")
    val interpolation = text("intp", "lerp").toLowerCase(Locale.ROOT)
    if interpolation != "lerp" && interpolation != "slerp" then invalid("Unknown LLuL interpolation")
    val mask = Option(text("mask", "")).filter(_.nonEmpty)
    Config(fields.get("weight").fold(0.15)(number("weight", _)), 1 << integer("multiply", 1, 0, 5),
      position("x"), position("y"), list("layers", Vector("OUT")), kinds, integer("start_steps", 5, 1),
      integer("max_steps", 0, 0), mode("up", upModes, upAA), mode("down", downModes, downAA),
      upAA, downAA, interpolation == "slerp" && mask.isEmpty, flag("force_float"), mask)
