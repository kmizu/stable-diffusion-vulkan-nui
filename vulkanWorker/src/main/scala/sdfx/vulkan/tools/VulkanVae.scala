// Latent RGB projection factors from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import sdfx.core.protocol.{EngineError, LoadedModel}
import sdfx.vulkan.{ApproxVaeDecoder, VulkanInference, VulkanProtocol}
import sdfx.vulkan.VulkanInference.Latent

private[vulkan] final class NativeVaeToolsCodec(engine: VulkanInference, device: String, model: LoadedModel, approximation: String,
    tiled: Boolean, tiny: Option[Path], approx: Option[ApproxVaeDecoder]):
  private val family = model.kind match
    case "sd15" => "sd1"
    case "sdxl" => "sdxl"
    case other => VulkanProtocol.unsupported(s"VAE tools on $other")
  private val tiling = if tiled then ", tiling = tiling(tile_size_w = 32, tile_size_h = 32)" else ""
  private val vae = tiny match
    case Some(file) => s"load_tae(${ujson.write(ujson.Str(file.toString))}, family = \"$family\", backend = ${ujson.write(ujson.Str(device))})"
    case None =>
      val scale = if model.kind == "sdxl" && model.vaePath.isEmpty then ", conv_scale = 0.03125" else ""
      s"load_vae(${ujson.write(ujson.Str(model.vaePath.getOrElse(model.path)))}, family = \"$family\", backend = ${ujson.write(ujson.Str(device))}$scale)"

  def encodeSeeded(image: BufferedImage, seed: Long, check: () => Unit): Latent =
    check()
    val code = s"""fn main(input) {
      vae = $vae
      rng = make_rng("cpu", seed = $seed)
      raw = vae_encode(vae, image::to_f32(input, channels = 3), rng = rng$tiling)
      return {value = vae_to_unet_latents(vae, raw)}
    }"""
    engine.vaeEncode(code, image)

  def decode(latent: Latent, check: () => Unit): BufferedImage =
    check()
    if approximation == "Approx cheap" then VulkanVae.rgb(latent, family == "sdxl")
    else approx.fold(decoded(latent, false))(_.decode(latent, () => check()))
  def decodeRaw(latent: Latent, check: () => Unit): BufferedImage =
    check()
    decoded(latent, true)
  private def decoded(latent: Latent, raw: Boolean): BufferedImage =
    val expression = if raw then "input" else "unet_to_vae_latents(vae, input)"
    engine.vaeDecode(s"fn main(input) { vae = $vae\n return {value = vae_decode(vae, $expression$tiling)} }", latent)

private[vulkan] object VulkanVae:
  def execute(raw: ujson.Value, engine: VulkanInference, device: String, model: LoadedModel, cancelled: () => Boolean,
      publish: () => Unit, event: (String,ujson.Value) => Unit, password: Option[String]): ujson.Value =
    val context = new NativeWorkContext(cancelled,publish,(kind,fields) => event(kind,ujson.read(NativeJson.render(NativeJson.Obj(fields)))))
    val mode = raw.obj.get("approximation").map(_.str).getOrElse("Full")
    val cache = raw.obj.get("cache_dir").map(v => Path.of(v.str)).getOrElse(Path.of("models/vae-tools"))
    val tiny = Option.when(mode == "TAESD")(tinyPath(model, Some(cache))
      .orElse(Option.when(raw.obj.get("allow_download").exists(_.bool))(downloadTiny(model,cache,cancelled)))
      .getOrElse(VulkanProtocol.unsupported("TAESD without compatible local tiny autoencoder weights")))
    val approx = Option.when(mode == "Approx NN") {
      val name = if model.kind == "sdxl" then "vaeapprox-sdxl.pt" else "model.pt"
      val explicit = raw.obj.get("approx_nn_path").flatMap(_.strOpt).filter(_.nonEmpty).map(Path.of(_))
      val cached = cache.resolve("vae-approx").resolve(name)
      val sibling = Option(Path.of(model.path).toAbsolutePath.getParent).flatMap(p => Option(p.getParent)).map(_.resolve("VAE-approx").resolve(name))
      val file = explicit.orElse((Seq(cached) ++ sibling).find(Files.isRegularFile(_))).getOrElse {
        if !raw.obj.get("allow_download").exists(_.bool) then VulkanProtocol.unsupported("Approx NN requires local A1111 weights")
        try NativeApproxDownload.official(cached,name,() => context.checkCancelled())
        catch case _: java.nio.file.FileAlreadyExistsException if Files.isRegularFile(cached) => ()
        cached
      }
      new ApproxVaeDecoder(file)
    }
    if !Set("Full","Approx cheap","Approx NN","TAESD")(mode) then VulkanProtocol.unsupported(s"VAE Tools approximation $mode")
    val tiled = raw.obj.get("tiled").exists(_.bool)
    val codec = new NativeVaeToolsCodec(engine,device,model,mode,tiled,tiny,approx)
    val params = NativeJson.parse(ujson.write(raw)).asInstanceOf[NativeJson.Obj].fields
    val provenance = Map("model_name" -> NativeJson.Str(model.name), "model_hash" -> NativeJson.Str(model.hash),
      "model_kind" -> NativeJson.Str(model.kind), "vae" -> NativeJson.Str(model.vaeName.getOrElse("Automatic")))
    try ujson.read(NativeJson.render(NativeJson.Obj(NativeVaeTools.execute(params,codec,context,provenance,password))))
    catch case e: NativeBackendError => throw EngineError(e.code, e.getMessage)

  def tinyPath(model: LoadedModel, cache: Option[Path]): Option[Path] =
    val stem = if model.kind == "sdxl" then "taesdxl" else "taesd"
    val modelDir = Path.of(model.path).toAbsolutePath.getParent
    val roots = cache.toSeq ++ Seq(Path.of("models/vae-tools")) ++ Option(modelDir) ++ Option(modelDir).flatMap(p => Option(p.getParent)).map(_.resolve("VAE-approx"))
    val overrides = sys.env.get(if model.kind == "sdxl" then "SDFX_TAESDXL_PATH" else "SDFX_TAESD_PATH").map(Path.of(_))
    (overrides.toSeq ++ roots.map(_.resolve(stem+".safetensors"))).find(Files.isRegularFile(_))

  private def downloadTiny(model: LoadedModel, dir: Path, cancelled: () => Boolean): Path =
    val (name, revision, digest) = if model.kind == "sdxl" then
      ("taesdxl","b20258aaef75ef61e659c1e0f14f251cf0ad153e","ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f")
    else ("taesd","614f76814bbe30edbe2e627ace1c2234c81a2c0e","db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549")
    Files.createDirectories(dir)
    val dest = dir.resolve(name+".safetensors")
    val temp = Files.createTempFile(dir,".tiny-",".part")
    try
      val client = java.net.http.HttpClient.newBuilder().followRedirects(java.net.http.HttpClient.Redirect.NORMAL)
        .connectTimeout(java.time.Duration.ofSeconds(15)).build()
      val uri = java.net.URI.create(s"https://huggingface.co/madebyollin/$name/resolve/$revision/diffusion_pytorch_model.safetensors")
      val request = java.net.http.HttpRequest.newBuilder(uri).timeout(java.time.Duration.ofMinutes(2)).build()
      val response = client.send(request,java.net.http.HttpResponse.BodyHandlers.ofInputStream())
      scala.util.Using.resource(response.body()) { input =>
        require(response.statusCode() == 200,"Tiny autoencoder download failed")
        scala.util.Using.resource(Files.newOutputStream(temp)) { output =>
          val block = new Array[Byte](65536)
          var count = 0L
          var size = input.read(block)
          while size >= 0 do
            if cancelled() then throw new java.util.concurrent.CancellationException()
            count += size
            require(count <= 9793292L,"Tiny autoencoder exceeds expected size")
            output.write(block,0,size)
            size = input.read(block)
        }
      }
      require(sdfx.core.imagebrowser.BrowserFiles.sha256(temp) == digest,"Tiny autoencoder digest mismatch")
      try Files.move(temp,dest)
      catch case _: java.nio.file.FileAlreadyExistsException => require(sdfx.core.imagebrowser.BrowserFiles.sha256(dest) == digest)
      dest
    finally Files.deleteIfExists(temp)

  private val sd15Factors=Array(Array(.298f,.207f,.208f),Array(.187f,.286f,.173f),
    Array(-.158f,.189f,.264f),Array(-.184f,-.271f,-.473f))
  private val sdxlFactors=Array(Array(.3448f,.4168f,.4395f),Array(-.1953f,-.0290f,.0250f),
    Array(.1074f,.0886f,-.0163f),Array(-.3730f,-.2499f,-.2088f))
  def rgb(latent: Latent,sdxl: Boolean = false): BufferedImage=
    val factors=if sdxl then sdxlFactors else sd15Factors
    val values = latent.values
    require(values.forall(_.isFinite),"nonfinite latent")
    val width = latent.width
    val height = latent.height
    val plane = width * height
    val image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    for i <- 0 until plane do
      var packed=0
      for channel <- 0 until 3 do
        var sum=0f
        for l <- 0 until 4 do sum+=values(l*plane+i)*factors(l)(channel)
        val value=(math.max(0f,math.min(1f,(sum+1f)/2f))*255f+.5f).toInt
        packed=(packed<<8)|value
      image.setRGB(i%width,i/width,packed)
    image
