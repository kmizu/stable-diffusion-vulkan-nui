package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanReference:
  val id = "reference"

  final case class Config(image: String, style: Double, weight: Double):
    def args(prefix: String, positive: Boolean): String =
      if weight == 0 then ""
      else s", reference_latent = ${prefix}_latent, reference_noise = ${prefix}_noise" +
        s", reference_weight = $weight, reference_style = $style, reference_positive = $positive"

    def prepare(prefix: String, tiled: String): Vector[String] =
      if weight == 0 then Vector.empty
      else Vector(
        s"""${prefix}_rng = make_rng("cpu", seed = 0)""",
        s"""${prefix}_latent = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(${prefix}_image), rng = ${prefix}_rng, posterior = "mode"$tiled))""",
        s"${prefix}_noise = tensor::randn_like(${prefix}_latent, rng = ${prefix}_rng)"
      )

  def parse(raw: Value): Config =
    val fields = raw.obj
    val image = fields.get("image").flatMap(_.strOpt).map(_.trim).filter(_.nonEmpty)
      .getOrElse(VulkanProtocol.invalid("Reference only needs an image"))
    def number(key: String, default: Double, max: Double): Double =
      val value = fields.get(key).fold(default)(_.numOpt.getOrElse(Double.NaN))
      if !value.isFinite || value < 0 || value > max then
        VulkanProtocol.invalid(s"Reference only $key must be in [0, $max]")
      value
    Config(image, number("style_fidelity", 0.5, 1), number("attention_auto_machine_weight", 1, 2))
