package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.vulkan.VulkanInference
import ujson.Value

private[vulkan] object VulkanNovelView:
  private def weights(model: Path, dir: String, stem: String): Path =
    Vector(".fp16.safetensors",".safetensors").map(suffix => model.resolve(dir).resolve(stem+suffix))
      .find(Files.isRegularFile(_)).getOrElse(throw IllegalArgumentException(s"Missing $dir/$stem safetensors"))

  def render(raw: Value, engine: VulkanInference, device: String, check: () => Unit,
      progress: (Int,Int) => Unit): BufferedImage =
    val model = Path.of(raw("model").str).toAbsolutePath
    require(Files.isRegularFile(model.resolve("model_index.json")), "Choose a Zero123 Diffusers model directory")
    def q(s: String) = ujson.write(ujson.Str(s))
    def scalar(k: String, default: Double) = raw.obj.get(k).fold(default)(_.num)
    val elevation = math.toRadians(scalar("elevation",0))
    val azimuth = math.toRadians(scalar("azimuth",30))
    val cfg = scalar("cfg_scale",3)
    val stepsValue = scalar("steps",50)
    val steps = stepsValue.toInt
    require(stepsValue == steps && steps >= 1 && steps <= 500 && cfg.isFinite && cfg >= 0 &&
      elevation.isFinite && azimuth.isFinite, "Invalid Zero123 sampling or camera settings")
    val seed = raw.obj.get("seed").fold(0L)(_.num.toLong) & 0xffffffffL
    val image = DiffusionImageResize.lanczos(DiffusionImageInput.readRgb(Path.of(raw("image_path").str)),256,256)
    val enc = weights(model,"image_encoder","model")
    val adapter = q(device)
    val code = s"""fn main(input) {
                   |vision=load_vision_encoder(${q(enc.toString)},backend=$adapter)
                   |return {embedding=encode_vision(vision,input,pooled=true)}
                   |}""".stripMargin
    check()
    val clip = engine.clipVision(code,DiffusionImageResize.bicubic(image,224,224),false,768)
    val input = clip.map(_.toDouble) ++ Array(elevation,math.sin(azimuth),math.cos(azimuth),0d)
    val projectionDir = if Files.isDirectory(model.resolve("clip_camera_projection")) then "clip_camera_projection" else "cc_projection"
    val projection = weights(model,projectionDir,"diffusion_pytorch_model")
    val token = Using.resource(CheckpointIO.open(projection)) { file =>
      val matrix = file.tensors.values.find(_.shape == Vector(768L,772L)).getOrElse(
        throw IllegalArgumentException("Invalid Zero123 camera projection"))
      val w = file.floats(matrix)
      val bias = file.tensors.values.find(_.shape == Vector(768L)).map(file.floats).getOrElse(new Array[Float](768))
      Array.tabulate(768)(r => (0 until 772).foldLeft(bias(r).toDouble)((sum,c) => sum + w(r*772+c)*input(c)))
    }
    // Diffusers DDIM leading timesteps with scaled-linear betas and set_alpha_to_one=false.
    val alpha = (0 until 1000).scanLeft(1.0) { (a,i) =>
      val root = math.sqrt(.00085) + (math.sqrt(.012)-math.sqrt(.00085))*i/999.0
      a * (1-root*root)
    }.tail
    def sigma(t: Int) = math.sqrt((1-alpha(t))/alpha(t))
    val schedule = (steps-1 to 0 by -1).map(i=>sigma(1+i*(1000/steps))) :+ sigma(0)
    val unet = weights(model,"unet","diffusion_pytorch_model")
    val vae = weights(model,"vae","diffusion_pytorch_model")
    val graph = s"""fn main(input) {
                    |unet=load_unet(${q(unet.toString)},family="sd1",backend=$adapter)
                    |vae=load_vae(${q(vae.toString)},family="sd1",backend=$adapter)
                    |rng=make_rng("cpu",seed=$seed)
                    |z=vae_encode(vae,image::to_f32(input),rng=rng,posterior="mode")
                    |token=tensor::reshape(tensor::vector(${token.mkString("[",",","]")}),[768,1,1])
                    |pos=predictor(unet,{crossattn=token},prediction="eps",concat=z)
                    |neg=predictor(unet,{crossattn=tensor::zeros_like(token)},prediction="eps",concat=tensor::zeros_like(z))
                    |guided=cfg(pos,neg,scale=$cfg)
                    |base=tensor::mul(tensor::randn_like(z,rng=rng),${math.hypot(1,schedule.head)})
                    |last=sample(base,tensor::vector(${schedule.mkString("[",",","]")}),predict=guided,method="euler",rng=rng)
                    |return {image=vae_decode(vae,unet_to_vae_latents(vae,tensor::mul(last,${1/math.hypot(1,schedule.last)})))}
                    |}""".stripMargin
    check()
    val callback: VulkanInference.Progress = (stage,done,total) => if stage == "sampling" then progress(Math.toIntExact(done),Math.toIntExact(total))
    val images = engine.dglFrames(graph,Map("input"->image).asJava,"image",callback)
    check()
    images.getFirst()
