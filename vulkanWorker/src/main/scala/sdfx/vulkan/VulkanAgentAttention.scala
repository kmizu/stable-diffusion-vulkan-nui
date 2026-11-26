package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanAgentAttention:
  val id = "agent_attention"

  private val stages = Seq("", "sp_")
  private val stageFields = Set("sx", "sy", "ratio", "agent_ratio")
  private val allowed = Set("enabled", "use_sp", "sp_step", "use_fp32", "max_downsample", "hires_fix_only") ++
    stages.flatMap(prefix => stageFields.map(prefix + _))

  final case class Stage(sx: Int, sy: Int, ratio: Double, agents: Double):
    def args(prefix: String): String =
      s", ${prefix}sx = $sx, ${prefix}sy = $sy, ${prefix}ratio = $ratio, ${prefix}ratio_agents = $agents"

  final case class Config(first: Option[Stage], second: Option[Stage], switchStep: Int,
      depth: Int, fp32: Boolean, hiresOnly: Boolean):
    def applicable(model: String, hiresEnabled: Boolean): Boolean =
      val minimumDepth = if Set("sdxl", "SDXL Inpaint")(model) then 2 else 1
      depth >= minimumDepth && (!hiresOnly || hiresEnabled) &&
        (first.exists(s => s.sx > 1 || s.sy > 1) || second.exists(s => s.sx > 1 || s.sy > 1))

    def args(seed: Long, hires: Boolean): String =
      if hires != hiresOnly || first.isEmpty && second.isEmpty then ""
      else s", agent_depth = $depth, agent_switch = $switchStep, agent_seed = $seed, agent_fp32 = $fp32" +
        first.fold("")(_.args("agent_")) + second.fold("")(_.args("agent_second_"))

  def parse(value: Value): Config =
    import VulkanProtocol.invalid
    val fields = value.obj
    if fields.keysIterator.exists(key => !allowed(key)) then invalid("Unknown Agent Attention option")

    def bool(key: String): Boolean =
      fields.get(key).fold(false) {
        case ujson.Bool(result) => result
        case ujson.Str(text) if text.trim.equalsIgnoreCase("true") => true
        case ujson.Str(text) if text.trim.equalsIgnoreCase("false") => false
        case _ => invalid(s"Agent Attention $key must be true or false")
      }

    def real(key: String, default: Double): Double =
      fields.get(key).fold(default)(_.numOpt.getOrElse(Double.NaN))

    def int(key: String, default: Int, minimum: Int = 0): Int =
      val n = real(key, default)
      if !n.isWhole || n < minimum || n > Int.MaxValue then
        invalid(s"Agent Attention $key must be an integer of at least $minimum")
      n.toInt

    def ratio(key: String, default: Double): Double =
      val n = real(key, default)
      if !n.isFinite || n < 0 || n > 1 then invalid(s"Agent Attention $key must be in [0, 1]")
      n

    def stage(prefix: String, stride: Int, agents: Double, used: Boolean): Option[Stage] =
      val s = Stage(int(prefix + "sx", stride), int(prefix + "sy", stride),
        ratio(prefix + "ratio", 0.4), ratio(prefix + "agent_ratio", agents))
      if used && (s.sx == 0 || s.sy == 0 || s.ratio == 0) then
        invalid(s"Agent Attention ${if prefix.isEmpty then "first" else "second"} stage needs nonzero strides and ratio")
      Option.when(used)(s)

    val switchStep = int("sp_step", 20)
    Config(stage("", 4, 0.95, switchStep > 0), stage("sp_", 2, 0.5, bool("use_sp")),
      switchStep, int("max_downsample", 1, 1), bool("use_fp32"), bool("hires_fix_only"))
