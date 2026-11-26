package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanLLLite:
  val id = "controllllite"

  final case class Config(model: String, image: Option[String], strength: Double, start: Double, end: Double):
    def active: Boolean = strength > 0 && start <= end

    def source(init: Option[String]): String =
      image.orElse(init).getOrElse(VulkanProtocol.invalid("ControlNet-LLLite requires a hint or init image"))

    def args(path: String, hint: String): String =
      s", lllite_path = ${ujson.write(ujson.Str(path))}, lllite_hint = $hint" +
        s", lllite_strength = $strength, lllite_start = $start, lllite_end = $end"

  def parse(value: Value): Config =
    val fields = value.obj
    def text(key: String): Option[String] =
      fields.get(key).flatMap(_.strOpt).map(_.trim).filter(_.nonEmpty)
    def number(key: String, fallback: Double, limit: Double): Double =
      val value = fields.get(key).fold(fallback)(_.numOpt.getOrElse(Double.NaN))
      if !value.isFinite || value < 0 || value > limit then
        VulkanProtocol.invalid(s"ControlNet-LLLite $key must be in [0, $limit]")
      value
    val model = text("model").filterNot(_.equalsIgnoreCase("none"))
      .getOrElse(VulkanProtocol.invalid("ControlNet-LLLite needs a model"))
    val start = number("start_percent", 0, 1)
    val end = number("end_percent", 1, 1)
    Config(model, text("image"), number("strength", 1, 10), start, end)
