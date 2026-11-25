package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object FastModePlugin:
  val Id = "fast_mode"
  val InfotextKey = "Fast mode"

  /** PNG restore: `Fast mode` enables the plugin; any other PNG switches it off. Malformed or out-of-range values give `None`. */
  def replay(fields: Map[String, String]): Option[Values] =
    fields.get(InfotextKey) match
      case None => Some(Values(Map("enabled" -> "false")))
      case Some(raw) =>
        val parts = raw.split(",").toSeq.map(_.trim).filter(_.nonEmpty)
        if parts.isEmpty then None
        else
          var from: Option[Double] = None
          var until: Option[Double] = None
          var interval: Option[Int] = None
          var order: Option[Int] = None
          var contraction: Option[Double] = None
          var cfgCacheInterval: Option[Int] = None
          var gateStep: Option[Int] = None
          var valid = true
          parts.foreach { part =>
            if part.startsWith("CFG from ") && from.isEmpty then
              from = part.stripPrefix("CFG from ").trim.toDoubleOption.filter(v => java.lang.Double.isFinite(v) && v >= 0.0 && v <= 0.4)
              if from.isEmpty then valid = false
            else if part.startsWith("CFG until ") && until.isEmpty then
              until = part.stripPrefix("CFG until ").trim.toDoubleOption.filter(v => java.lang.Double.isFinite(v) && v >= 0.3 && v <= 1.0)
              if until.isEmpty then valid = false
            else if part.startsWith("DeepCache interval ") && interval.isEmpty then
              interval = part.stripPrefix("DeepCache interval ").trim.toIntOption.filter(v => v >= 0 && v <= 10)
              if interval.isEmpty then valid = false
            else if part.startsWith("TaylorSeer order ") && order.isEmpty then
              order = part.stripPrefix("TaylorSeer order ").trim.toIntOption.filter(v => v >= 0 && v <= 2)
              if order.isEmpty then valid = false
            else if part.startsWith("HiCache contraction ") && contraction.isEmpty then
              contraction = part.stripPrefix("HiCache contraction ").trim.toDoubleOption.filter(v => java.lang.Double.isFinite(v) && v >= 0.1 && v <= 1.0)
              if contraction.isEmpty then valid = false
            else if part.startsWith("CFG-Cache interval ") && cfgCacheInterval.isEmpty then
              cfgCacheInterval = part.stripPrefix("CFG-Cache interval ").trim.toIntOption.filter(v => v >= 0 && v <= 10)
              if cfgCacheInterval.isEmpty then valid = false
            else if part.startsWith("T-GATE step ") && gateStep.isEmpty then
              gateStep = part.stripPrefix("T-GATE step ").trim.toIntOption.filter(v => v >= 0 && v <= 50)
              if gateStep.isEmpty then valid = false
            else valid = false
          }
          if !valid || until.isEmpty || interval.isEmpty then None
          else
            val base = Values.defaults(new FastModePlugin().controls).map
            Some(Values(base ++ Map(
              "enabled" -> "true",
              "guidance_from" -> from.fold("0")(Values.fmt),
              "guidance_until" -> Values.fmt(until.get),
              "interval" -> interval.get.toString,
              "order" -> order.getOrElse(0).toString,
              "contraction" -> Values.fmt(contraction.getOrElse(1.0)),
              "cfg_cache_interval" -> cfgCacheInterval.getOrElse(0).toString,
              "gate_step" -> gateStep.getOrElse(0).toString
            )))

/** One switch for training-free speed-ups verified on the scala-torch engine (docs/v0.32-speedup-verification.md):
  * DeepCache + HiCache (scaled-Hermite TaylorSeer forecasting), FasterCache CFG-Cache, T-GATE cross-attention caching,
  * and CFG for the first part of the steps only.
  *
  * It only fills in what the user has not set: a DeepCache, CFG-Cache, T-GATE, or CFG Schedule plugin that is switched on
  * keeps its own values.
  */
class FastModePlugin extends Plugin:
  import FastModePlugin.*
  def id = Id
  def name = "Fast mode"
  def kind: PluginKind = PluginKind.AlwaysOn
  override def requiresEngineFeatures: Seq[String] = Seq("deepcache", "tgate", "cfg_schedule", "cfg_cache")
  override def description =
    "Faster sampling combining training-free approximations: DeepCache + HiCache forecasting, CFG-Cache, T-GATE cross-attention caching, " +
    "and CFG only for the first part of the steps. DeepCache, CFG-Cache, T-GATE and CFG Schedule keep their own values when switched on."

  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("guidance_from", "CFG from (Limited Interval CFG; 0 = start)", 0.0, 0.4, 0.05, 0.10),
    Slider("guidance_until", "CFG until (fraction of the steps; 1 = all steps)", 0.3, 1, 0.05, 0.7),
    IntField("interval", "DeepCache: full step every N steps (0 = no DeepCache)", 0, 10, 3),
    IntField("order", "TaylorSeer forecast order (0 = reuse, 1 = linear, 2 = quadratic)", 0, 2, 1),
    Slider("contraction", "HiCache contraction (1 = exact TaylorSeer)", 0.1, 1.0, 0.05, 0.7),
    IntField("cfg_cache_interval", "CFG-Cache: full CFG every N steps (0 = off)", 0, 10, 3),
    IntField("gate_step", "T-GATE: cache cross-attention after step (0 = off)", 0, 50, 10))

  override def process(p: GenerationParams, v: Values): GenerationParams =
    val until = math.max(0.0, math.min(1.0, v.double("guidance_until", 0.7)))
    val from = math.max(0.0, math.min(until - 0.05, v.double("guidance_from", 0.10)))
    val interval = v.int("interval", 3)
    val order = math.max(0, math.min(2, v.int("order", 1)))
    val contraction = math.max(0.1, math.min(1.0, v.double("contraction", 0.7)))
    val cfgCacheInterval = math.max(0, math.min(10, v.int("cfg_cache_interval", 3)))
    val gateStep = math.max(0, math.min(50, v.int("gate_step", 10)))
    val withCache =
      if interval < 2 || p.hooks.contains(DeepCachePlugin.Id) then p
      else
        val dc = ujson.Obj("interval" -> interval, "branch" -> 0, "full_first" -> 5)
        if order > 0 then
          dc("order") = order
          if contraction < 1.0 then dc("contraction") = contraction
        Hooks(p, DeepCachePlugin.Id, dc)._1
    val withTGate =
      if gateStep < 1 || withCache.hooks.contains(TGatePlugin.Id) then withCache
      else Hooks(withCache, TGatePlugin.Id, ujson.Obj("gate_step" -> gateStep))._1
    val withCfgCache =
      if cfgCacheInterval < 2 || p.cfgScale <= 1.0 || withTGate.hooks.contains(CfgCachePlugin.Id) then withTGate
      else Hooks(withTGate, CfgCachePlugin.Id, ujson.Obj("start" -> 0.25, "interval" -> cfgCacheInterval))._1
    // The engine skips the negative pass exactly when the scheduled CFG is 1, so the schedule steps from the request's CFG to 1.
    val withGuidance =
      if (until >= 1.0 && from <= 0.0) || p.cfgScale <= 1.0 || withCfgCache.hooks.contains("cfg_schedule") then withCfgCache
      else
        val schedule = if until < 1.0 then ujson.Obj(
          "start_cfg" -> p.cfgScale, "end_cfg" -> 1.0, "curve" -> "linear",
          "start" -> math.max(0.0, until - 0.01), "end" -> until)
        else ujson.Obj(
          "start_cfg" -> p.cfgScale, "end_cfg" -> p.cfgScale, "curve" -> "linear",
          "start" -> 0.0, "end" -> 1.0)
        if from > 0.0 then schedule("active_start") = from
        Hooks(withCfgCache, "cfg_schedule", schedule)._1
    val info = (if from > 0.0 then s"CFG from ${Values.fmt(from)}, " else "") +
      s"CFG until $until, DeepCache interval $interval" +
      (if interval >= 2 && order > 0 then s", TaylorSeer order $order" else "") +
      (if interval >= 2 && order > 0 && contraction < 1.0 then s", HiCache contraction ${Values.fmt(contraction)}" else "") +
      (if cfgCacheInterval >= 2 && p.cfgScale > 1.0 then s", CFG-Cache interval $cfgCacheInterval" else "") +
      (if gateStep >= 1 then s", T-GATE step $gateStep" else "")
    withGuidance.copy(extra = withGuidance.extra + (InfotextKey -> info))
