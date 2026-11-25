package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object DeepCachePlugin:
  val Id = "deepcache"
  val IntervalKey = "DeepCache interval"
  val BranchKey = "DeepCache branch"
  val FirstKey = "DeepCache full first"
  val OrderKey = "DeepCache order"
  val ContractionKey = "DeepCache contraction"
  val InfotextKeys: Seq[String] = Seq(IntervalKey, BranchKey, FirstKey, OrderKey, ContractionKey)

  private def inRange(raw: String, low: Int, high: Int): Option[String] =
    raw.trim.toIntOption.filter(n => n >= low && n <= high).map(_.toString)

  private def inDoubleRange(raw: String, low: Double, high: Double): Option[String] =
    raw.trim.toDoubleOption.filter(v => java.lang.Double.isFinite(v) && v >= low && v <= high).map(Values.fmt)

  /** PNG restore: `DeepCache interval` enables the plugin; any other PNG switches it off. Out-of-range values give `None`. */
  def replay(fields: Map[String, String]): Option[Values] =
    if !fields.contains(IntervalKey) then Some(Values(Map("enabled" -> "false")))
    else
      val interval = inRange(fields(IntervalKey), 2, 10)
      val branch = fields.get(BranchKey).map(inRange(_, 0, 3))
      val first = fields.get(FirstKey).map(inRange(_, 0, 50))
      val order = fields.get(OrderKey).map(inRange(_, 0, 2))
      val contraction = fields.get(ContractionKey).map(inDoubleRange(_, 0.1, 1.0))
      if interval.isEmpty || branch.contains(None) || first.contains(None) || order.contains(None) || contraction.contains(None) then None
      else
        val base = Values.defaults(new DeepCachePlugin().controls).map
        Some(Values(base ++ Map("enabled" -> "true", "interval" -> interval.get) ++ branch.flatten.map("branch" -> _) ++
          first.flatten.map("full_first" -> _) ++ order.flatten.map("order" -> _) ++
          contraction.flatten.map("contraction" -> _)))

/** DeepCache for the U-Net: recompute only the shallow blocks on most steps and reuse the deep features recorded by the
  * last full step. Built into the engine (low-VRAM / speed options, #143), not an A1111 extension port.
  */
class DeepCachePlugin extends HookPlugin:
  import DeepCachePlugin.*
  def id = Id
  def name = "DeepCache"
  def hook = Id
  override def description =
    "Faster sampling by reusing the U-Net's deep features between neighbouring steps (a full step every N steps). " +
    "An approximation: the image differs a little from the exact one (see docs/optimization-verification.md). " +
    "Needs the GPU VRAM mode; ControlNet, Composable LoRA and hooks that patch the U-Net are refused."

  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    IntField("interval", "Full step every N steps", 2, 10, 3),
    IntField("branch", "Cache depth (0 = shallowest, fastest)", 0, 3, 0),
    IntField("full_first", "Full steps at the start (they decide the composition)", 0, 50, 5),
    IntField("order", "TaylorSeer forecast order (0 = reuse, 1 = linear, 2 = quadratic)", 0, 2, 0),
    Slider("contraction", "HiCache Hermite contraction (1 = exact TaylorSeer)", 0.1, 1.0, 0.05, 1.0))

  def hookFields(v: Values): ujson.Obj =
    val order = v.int("order", 0)
    val contraction = v.double("contraction", 1.0)
    val base = ujson.Obj("interval" -> v.int("interval", 3), "branch" -> v.int("branch", 0), "full_first" -> v.int("full_first", 5))
    if order > 0 then base("order") = order
    if contraction < 1.0 then base("contraction") = contraction
    base

  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    val base = Seq(IntervalKey -> field(o, "interval"), BranchKey -> field(o, "branch"), FirstKey -> field(o, "full_first"))
    val orderOpt = o.value.get("order").flatMap(_.numOpt).map(_.toInt).filter(_ > 0)
    val contractionOpt = o.value.get("contraction").flatMap(_.numOpt).filter(_ < 1.0).map(v => ContractionKey -> Values.fmt(v))
    base ++ orderOpt.map(ord => OrderKey -> ord.toString) ++ contractionOpt
