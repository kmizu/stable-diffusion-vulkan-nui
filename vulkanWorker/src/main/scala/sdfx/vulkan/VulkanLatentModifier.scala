package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] object VulkanLatentModifier:
  val id = "latent_modifier"
  private val effects = Vector("sharpness_multiplier", "tonemap_multiplier", "contrast_multiplier",
    "combat_cfg_drift", "extra_noise_multiplier", "divisive_norm_multiplier", "spectral_mod_multiplier")

  def config(p: GenerationParams): Option[Value] =
    if p.cfgScale <= 1 then None
    else p.hooks.get(id).filter { value =>
      !value.obj.get("enabled").contains(ujson.Bool(false)) &&
        effects.exists(key => value.obj.get(key).exists(_.num != 0))
    }

  def arguments(options: Value, seed: Long): String =
    val numeric = effects.map(_ -> 0.0) ++ Vector("tonemap_percentile" -> 100.0,
      "spectral_mod_percentile" -> 5.0, "extra_noise_lowpass" -> 100.0, "divisive_norm_size" -> 127.0)
    val numbers = numeric.map { (key, default) =>
      val value = options.obj.get(key).fold(default)(_.num)
      require(value.isFinite, s"Invalid Latent Modifier option $key")
      s"$key=$value"
    }
    val choices = Vector(
      "sharpness_method" -> Vector("anisotropic", "gaussian"),
      "tonemap_method" -> Vector("reinhard", "arctan", "clamp"),
      "combat_method" -> Vector("subtract", "median"),
      "extra_noise_type" -> Vector("gaussian", "uniform"),
      "spectral_mod_mode" -> Vector("hard_clamp", "soft_clamp")
    ).map { (key, values) =>
      val value = options.obj.get(key).fold(values.head)(_.str)
      require(values.contains(value), s"Invalid Latent Modifier option $key")
      s"$key=${ujson.write(ujson.Str(value))}"
    }
    val fields = numbers ++ choices :+ s"seed=${seed & 0xffffffffL}"
    s", latent_modifier={${fields.mkString(",")}}"
