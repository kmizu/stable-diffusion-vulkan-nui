package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object PostDiffPlugin:
  val Id = "postdiff"
  val Hook = Id
  val ScaleKey = "PostDiff scale"
  val PortionKey = "PostDiff portion"
  val InfotextKeys: Seq[String] = Seq(ScaleKey, PortionKey)

  private def parseDouble(raw: String, low: Double, high: Double): Option[Double] =
    raw.trim.toDoubleOption.filter(v => java.lang.Double.isFinite(v) && v >= low && v <= high)

  def replay(fields: Map[String, String]): Option[Values] =
    fields.get(ScaleKey) match
      case None => Some(Values(Map("enabled" -> "false")))
      case Some(rawScale) =>
        val scale = parseDouble(rawScale, 0.25, 0.95)
        val portion = fields.get(PortionKey).map(parseDouble(_, 0.05, 0.85))
        if scale.isEmpty || portion.contains(None) then None
        else
          val base = Values.defaults(new PostDiffPlugin().controls).map
          Some(Values(base ++ Map("enabled" -> "true", "scale" -> Values.fmt(scale.get)) ++
            portion.flatten.map(v => "portion" -> Values.fmt(v))))

/** PostDiff progressive resolution diffusion (arXiv:2508.06160):
  * denoises early coarse steps at a lower spatial scale before upscaling and transitioning to full resolution.
  */
class PostDiffPlugin extends HookPlugin:
  import PostDiffPlugin.*
  def id = Id
  def name = "PostDiff Progressive Resolution"
  def hook = Hook
  override def description =
    "Training-free acceleration (PostDiff): runs early composition steps at reduced spatial resolution and transitions to full resolution."

  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("scale", "First-stage resolution scale", 0.25, 0.95, 0.05, 0.5),
    Slider("portion", "First-stage step portion", 0.05, 0.85, 0.05, 0.4)
  )

  def hookFields(v: Values): ujson.Obj =
    ujson.Obj("scale" -> v.double("scale", 0.5), "portion" -> v.double("portion", 0.4))

  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq(
      ScaleKey -> Values.fmt(o.value.get("scale").map(_.num).getOrElse(0.5)),
      PortionKey -> Values.fmt(o.value.get("portion").map(_.num).getOrElse(0.4))
    )

  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("PostDiff scale", "scale"), numAxis("PostDiff portion", "portion"))
