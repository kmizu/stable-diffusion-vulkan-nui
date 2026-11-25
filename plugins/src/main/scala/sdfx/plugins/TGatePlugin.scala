package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object TGatePlugin:
  val Id = "tgate"
  val Hook = Id
  val GateStepKey = "T-GATE gate step"
  val InfotextKeys: Seq[String] = Seq(GateStepKey)

  def replay(fields: Map[String, String]): Option[Values] =
    fields.get(GateStepKey) match
      case None => Some(Values(Map("enabled" -> "false")))
      case Some(raw) =>
        raw.trim.toIntOption.filter(v => v >= 1 && v <= 50).map { step =>
          Values(Map("enabled" -> "true", "gate_step" -> step.toString))
        }

class TGatePlugin extends HookPlugin:
  import TGatePlugin.*
  def id = Id
  def name = "T-GATE Cross-Attention Cache"
  def hook = Hook
  override def description =
    "Training-free acceleration (Zhang et al., TMLR 2024/2025): caches and reuses cross-attention outputs after gate_step."

  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    IntField("gate_step", "Gate step (cache cross-attention after step)", 1, 50, 8)
  )

  def hookFields(v: Values): ujson.Obj =
    ujson.Obj("gate_step" -> v.int("gate_step", 8))

  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq(GateStepKey -> o.value.get("gate_step").map(_.num.toInt.toString).getOrElse("8"))

  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("T-GATE gate step", "gate_step"))
