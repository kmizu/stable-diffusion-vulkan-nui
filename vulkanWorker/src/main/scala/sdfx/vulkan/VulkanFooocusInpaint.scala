package sdfx.vulkan

import java.nio.file.Path
import sdfx.core.protocol.{GenerationRequest, LoadedModel}
import sdfx.vulkan.tools.VulkanFooocusFiles

private[vulkan] object VulkanFooocusInpaint:
  val id = "fooocus_inpaint"

  final case class Assets(unet: Path, head: Path, start: Double, end: Double):
    def prepareInput(width: Int, height: Int, rng: String, tiled: String): Vector[String] =
      def replicated(size: Int): String = (Vector(0) ++ (0 until size) :+ (size - 1)).mkString("[", ",", "]")
      val headFile = ujson.write(ujson.Str(head.toString))
      Vector(
        s"fooocus_head = tensor::load($headFile)",
        s"""fooocus_latent = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(init), rng = $rng, posterior = "mode"$tiled))""",
        "fooocus_feed = tensor::concat([latent_mask, fooocus_latent], axis = 2)",
        s"fooocus_wide = tensor::index_select(fooocus_feed, axis = 0, indices = ${replicated(width / 8)})",
        s"fooocus_input = tensor::index_select(fooocus_wide, axis = 1, indices = ${replicated(height / 8)})")

    def predictorArgs: String =
      s", fooocus_head = fooocus_head, fooocus_input = fooocus_input, fooocus_start = $start, fooocus_end = $end"

  def prepareAssets(engine: VulkanInference, model: LoadedModel, req: GenerationRequest, hasDifference: Boolean,
      scratch: Path, cancelled: () => Boolean): Assets =
    import VulkanProtocol.{invalid, unsupported}
    if !engine.supportsFooocusInpaint() then unsupported("This Vulkan runtime lacks Fooocus Inpaint")
    if !Set("sdxl", "SDXL Inpaint")(model.kind) then unsupported("Fooocus Inpaint requires an SDXL checkpoint")
    if !req.params.img2img.exists(i => i.mask.nonEmpty || hasDifference) then invalid("Fooocus Inpaint requires an input image and mask")
    val config = req.params.hooks(id).obj
    val modelName = config.get("model").flatMap(_.strOpt).filter(_.nonEmpty)
      .getOrElse(invalid("Fooocus Inpaint requires a patch model"))
    val patch = Path.of(VulkanConditioning.file(req.controlnetPaths.getOrElse(modelName, modelName), "Fooocus Inpaint patch"))
    val headPath = config.get("head").flatMap(_.strOpt).filter(_.nonEmpty)
      .getOrElse(patch.resolveSibling("fooocus_inpaint_head.pth").toString)
    val head = Path.of(VulkanConditioning.file(headPath, "Fooocus Inpaint head"))
    val start = config.get("start_percent").fold(0.0)(_.num)
    val end = config.get("end_percent").fold(1.0)(_.num)
    if !start.isFinite || !end.isFinite || start < 0 || end > 1 || start > end then
      invalid("Fooocus Inpaint requires 0 ≤ start ≤ end ≤ 1")
    val headFile = scratch.resolve("head.npy")
    val unet = scratch.resolve("unet.safetensors")
    VulkanFooocusFiles.writeHead(head, headFile)
    VulkanFooocusFiles.mergeUnet(Path.of(model.path), patch, unet, cancelled)
    Assets(unet, headFile, start, end)
