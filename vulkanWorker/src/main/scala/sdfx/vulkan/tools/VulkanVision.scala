package sdfx.vulkan.tools

import java.nio.file.{Files, Path}
import scala.util.Using
import sdfx.vulkan.VulkanProtocol.{invalid,unsupported}

private[vulkan] object VulkanVision:
  case class Adapter(plus: Boolean, encoder: Path)

  private def cachedRepo(repo: String): Option[Path] =
    val hub = sys.env.get("HF_HUB_CACHE").map(Path.of(_)).getOrElse(
      sys.env.get("HF_HOME").map(Path.of(_)).getOrElse(Path.of(sys.props("user.home"),".cache","huggingface")).resolve("hub"))
    if !repo.matches("[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+") then return None
    val root = hub.resolve("models--" + repo.replace("/","--"))
    val ref = root.resolve("refs/main")
    Option.when(Files.isRegularFile(ref))(Files.readString(ref).trim).filter(_.matches("[0-9a-f]{40}"))
      .map(root.resolve("snapshots").resolve(_)).filter(Files.isDirectory(_))

  private def findWeights(dir: Path): Option[Path] =
    Seq("model.safetensors","pytorch_model.bin","clip_vision_h.safetensors").map(dir.resolve(_)).find(Files.isRegularFile(_))

  def inspect(model: Path, requested: String, family: String): Adapter =
    if !model.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      unsupported("IP-Adapter checkpoint format (a .safetensors adapter is required)")
    val header = Using.resource(Files.newInputStream(model))(in => NativeMetadataUtils.parseHeader(in,Files.size(model))._1)
    val plus = header.keys.exists(k => k == "image_proj.latents" || k.endsWith(".image_proj.latents"))
    val proj = header.iterator.collectFirst { case (k, NativeJson.Obj(fields)) if
      k.endsWith("image_proj.proj.weight") || k.endsWith("image_proj.proj_in.weight") =>
        fields.get("shape").collect { case NativeJson.Arr(xs) if xs.size == 2 => xs(1).asInstanceOf[NativeJson.Number].value.toInt }
    }.flatten
    val bigG = proj.fold(family == "sdxl" && !model.getFileName.toString.contains("vit-h"))(n => if plus then n >= 1664 else n >= 1280)
    if bigG then unsupported("CLIP ViT-bigG IP-Adapter (native vision loader requires ViT-H)")
    val selected = Option(requested.trim).filter(_.nonEmpty)
    val explicit = selected.map(Path.of(_)).filter(Files.isRegularFile(_))
    val directory = selected.map(Path.of(_)).filter(Files.isDirectory(_)).flatMap(findWeights)
    val adjacent = if selected.nonEmpty then None else findWeights(model.getParent.resolve("image_encoder"))
    val cached = cachedRepo(selected.getOrElse("h94/IP-Adapter")).flatMap { root =>
      if selected.nonEmpty then findWeights(root) else findWeights(root.resolve(if bigG then "sdxl_models/image_encoder" else "models/image_encoder"))
    }
    val encoder = explicit.orElse(directory).orElse(adjacent).orElse(cached).getOrElse(
      invalid("IP-Adapter needs a CLIP vision encoder. Set image_encoder to its weights, local directory, or cached Hugging Face repository."))
    Adapter(plus,encoder.toAbsolutePath.normalize())
