package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanCharacteristic:
  val id = "characteristic_guidance"

  final case class Config(regularization: Double, range: Double, iterations: Int, bins: Int,
      reuse: Double, tolerance: Double, step: Double, speed: Double, strength: Double, memory: Int,
      start: Double, stop: Double):
    def window(steps: Int): (Int, Int) =
      def bound(fraction: Double): Int = math.rint(steps * fraction).max(0).min(steps).toInt
      (bound(start), bound(stop))

    def applicable(steps: Int, cfg: Double): Boolean =
      if cfg < 1 then VulkanProtocol.unsupported("Characteristic Guidance with CFG below 1")
      val (first, end) = window(steps)
      cfg > 1 && first < end

    def args(steps: Int, cfg: Double): String =
      if !applicable(steps, cfg) then ""
      else
        val (first, end) = window(steps)
        s", characteristic = {regularization = $regularization, regularization_range = $range, reuse = $reuse, " +
          s"tolerance = $tolerance, step_size = $step, annealing_speed = $speed, annealing_strength = $strength, " +
          s"iterations = $iterations, bins = $bins, memory = $memory, start = $first, stop = $end}"

  def parse(value: Value): Config =
    val fields = value.obj
    def invalid(message: String): Nothing = VulkanProtocol.invalid(s"Characteristic Guidance $message")
    def number(key: String, default: Double): Double =
      val result = fields.get(key).filterNot(_ == ujson.Null).fold(default) { v =>
        v.numOpt.orElse(v.strOpt.flatMap(_.toDoubleOption)).getOrElse(invalid(s"$key must be a number"))
      }
      if !result.isFinite then invalid(s"$key must be finite")
      result
    def whole(key: String, default: Int, minimum: Int): Int =
      val result = number(key, default)
      if !result.isWhole || result < minimum || result > Int.MaxValue then invalid(s"$key must be an integer of at least $minimum")
      result.toInt
    def stretch(n: Double): Double =
      if n <= 5 then n else 5 + math.expm1(.8898 * (n - 5)) / .8898
    val initial = stretch(number("reg_ini", 1))
    val range = stretch(number("reg_range", 1))
    val strength = number("reg_w", .5)
    val exponent = number("res", -4)
    if initial < 0 || !initial.isFinite then invalid("regularization strength is out of range")
    if range <= 0 || !range.isFinite || strength <= 0 then invalid("needs positive regularization range and annealing strength")
    if exponent > 300 then invalid("log tolerance must be 300 or less")
    fields.get("controlnet_mode").filterNot(_ == ujson.Null).foreach { mode =>
      if mode.strOpt.forall(s => s != "More Prompt" && s != "More ControlNet") then invalid("has an unknown ControlNet compatibility mode")
    }
    Config(initial, range, whole("ite", 50, 1), whole("noise_base", 0, 0), number("chara_decay", 1),
      math.pow(10, exponent), number("lr", 1), number("reg_size", .4), strength, whole("aa_dim", 2, 1),
      number("start_step", 0), number("stop_step", 1))
