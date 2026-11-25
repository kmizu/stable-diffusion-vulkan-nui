package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object CfgCachePlugin:
  val Id = "cfg_cache"
  val Hook = Id
  val StartKey = "CFG-Cache start"
  val IntervalKey = "CFG-Cache interval"
  val AdaptiveKey = "CFG-Cache adaptive threshold"
  val HfWeightKey = "CFG-Cache HF weight"
  val InfotextKeys: Seq[String] = Seq(StartKey, IntervalKey, AdaptiveKey, HfWeightKey)

  private def parseDouble(raw: String, low: Double, high: Double): Option[Double] =
    raw.trim.toDoubleOption.filter(v => java.lang.Double.isFinite(v) && v >= low && v <= high)

  def replay(fields: Map[String, String]): Option[Values] =
    fields.get(IntervalKey) match
      case None => Some(Values(Map("enabled" -> "false")))
      case Some(rawInterval) =>
        val interval = rawInterval.trim.toIntOption.filter(v => v >= 2 && v <= 20)
        val start = fields.get(StartKey).map(parseDouble(_, 0.0, 1.0))
        val adaptive = fields.get(AdaptiveKey).map(parseDouble(_, 0.0, 1.0))
        val hfWeight = fields.get(HfWeightKey).map(parseDouble(_, 0.0, 0.5))
        if interval.isEmpty || start.contains(None) || adaptive.contains(None) || hfWeight.contains(None) then None
        else
          val base = Values.defaults(new CfgCachePlugin().controls).map
          Some(Values(base ++ Map("enabled" -> "true", "interval" -> interval.get.toString) ++
            start.flatten.map(v => "start" -> Values.fmt(v)) ++
            adaptive.flatten.map(v => "adaptive_threshold" -> Values.fmt(v)) ++
            hfWeight.flatten.map(v => "hf_weight" -> Values.fmt(v))))

/** FasterCache CFG-Cache (Lv et al., ICLR 2025) + Adaptive Guidance (Castillo et al.):
  * caches the CFG noise delta after warmup steps and skips the unconditional U-Net pass for `interval - 1` steps.
  */
class CfgCachePlugin extends HookPlugin:
  import CfgCachePlugin.*
  def id = Id
  def name = "CFG-Cache (FasterCache)"
  def hook = Hook
  override def description =
    "Training-free acceleration (FasterCache + Adaptive Guidance): caches the unconditional CFG delta after warmup steps " +
    "and skips redundant negative U-Net passes while keeping full positive conditioning active."

  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("start", "Warmup fraction (full CFG steps at start)", 0.0, 1.0, 0.05, 0.25),
    IntField("interval", "Reuse interval (full CFG every N steps)", 2, 20, 3),
    Slider("adaptive_threshold", "Adaptive cosine threshold (0 = fixed interval)", 0.0, 1.0, 0.02, 0.0),
    Slider("hf_weight", "High-frequency boost weight late in sampling", 0.0, 0.5, 0.05, 0.0)
  )

  def hookFields(v: Values): ujson.Obj =
    val adaptive = v.double("adaptive_threshold", 0.0)
    val hfWeight = v.double("hf_weight", 0.0)
    val base = ujson.Obj("start" -> v.double("start", 0.25), "interval" -> v.int("interval", 3))
    if adaptive > 0.0 then base("adaptive_threshold") = adaptive
    if hfWeight > 0.0 then base("hf_weight") = hfWeight
    base

  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    val base = Seq(
      StartKey -> Values.fmt(o.value.get("start").map(_.num).getOrElse(0.25)),
      IntervalKey -> o.value.get("interval").map(_.num.toInt.toString).getOrElse("3")
    )
    val adaptiveOpt = o.value.get("adaptive_threshold").flatMap(_.numOpt).filter(_ > 0.0).map(v => AdaptiveKey -> Values.fmt(v))
    val hfOpt = o.value.get("hf_weight").flatMap(_.numOpt).filter(_ > 0.0).map(v => HfWeightKey -> Values.fmt(v))
    base ++ adaptiveOpt ++ hfOpt

  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("CFG-Cache interval", "interval"), numAxis("CFG-Cache start", "start"))
