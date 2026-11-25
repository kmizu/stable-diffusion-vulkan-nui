package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import sdfx.vulkan.tools.{CheckpointIO, DiffusionImageInput, DiffusionImageResize}
import scala.util.Using
import ujson.Value

/** Stable Video Diffusion uses two VAEs from the single-file SGM checkpoint. */
private[vulkan] object VulkanSvd:
  def render(raw: Value, engine: VulkanInference, device: String, check: () => Unit,
      progress: (Int,Int) => Unit): Vector[BufferedImage] =
    def integer(key: String, default: Int, minimum: Int, maximum: Int): Int =
      val n = raw.obj.get(key).fold(default.toDouble)(_.num)
      require(n.isWhole && n >= minimum && n <= maximum, s"Invalid SVD $key")
      n.toInt
    def number(key: String, default: Double, minimum: Double, maximum: Double): Double =
      val n = raw.obj.get(key).fold(default)(_.num)
      require(n >= minimum && n <= maximum, s"Invalid SVD $key")
      n
    val path = Path.of(raw("model").str).toAbsolutePath.normalize
    require(Files.isRegularFile(path) && path.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors"),
      "Select a single-file SVD or SVD-XT safetensors checkpoint")
    Using.resource(CheckpointIO.open(path)) { checkpoint =>
      val required = Seq("conditioner.embedders.0.open_clip.model.visual.proj",
        "conditioner.embedders.3.encoder.quant_conv.weight", "first_stage_model.decoder.conv_out.time_mix_conv.weight",
        "model.diffusion_model.input_blocks.1.0.time_stack.in_layers.2.weight")
      require(required.forall(checkpoint.tensors.contains), "Checkpoint lacks the SVD temporal model or conditioning weights")
    }
    val w = integer("width",1024,64,2048)
    val h = integer("height",576,64,2048)
    require(w % 64 == 0 && h % 64 == 0, "SVD dimensions must be multiples of 64")
    val frames = integer("frames",14,2,64)
    val fps = integer("fps",6,1,60)
    val bucket = integer("motion_bucket_id",127,1,1023)
    val chunk = integer("decode_chunk_size",4,1,64)
    val steps = integer("steps",25,1,100)
    val noiseAug = number("noise_aug_strength",0.02,0,10)
    val low = number("min_cfg",1,0,15)
    val high = number("max_cfg",3,0,15)
    val requestedSeed = raw.obj.get("seed") match
      case None => 0L
      case Some(ujson.Str(s)) => s.toLongOption.getOrElse(VulkanProtocol.invalid("Invalid SVD seed"))
      case Some(ujson.Num(n)) if n.isWhole && n.abs <= 9007199254740991L => n.toLong
      case _ => VulkanProtocol.invalid("Provide an exact integer SVD seed, using a string for large values")
    val seed = requestedSeed & 0xffffffffL
    val source = DiffusionImageResize(DiffusionImageInput.readRgb(Path.of(raw("image_path").str)), w, h, 1)
    val model = ujson.write(ujson.Str(path.toString))
    val backend = ujson.write(ujson.Str(device))
    val ids = for
      value <- Vector(fps.toDouble-1,bucket.toDouble,noiseAug)
      cosine <- Vector(true,false)
      i <- 0 until 128
    yield
      val angle = value * math.exp(-math.log(10000.0) * i / 128)
      if cosine then math.cos(angle) else math.sin(angle)
    def vector(values: Seq[Double]) = values.mkString("tensor::vector([",",","])")
    val guidance = vector(Vector.tabulate(frames)(i => low + (high-low)*i/(frames-1)))
    // The SVD Euler scheduler uses continuous EDM time and Karras rho 7.
    val start = math.pow(700.0,1.0/7)
    val end = math.pow(0.002,1.0/7)
    val sigmas = Vector.tabulate(steps)(i => math.pow(start+(end-start)*i/math.max(1,steps-1),7)) :+ 0.0
    val shape = s"[${w/8},${h/8},4,$frames]"
    val script = new StringBuilder
    script ++= s"""fn main(input) {
                  |unet=load_unet($model,family="svd",backend=$backend)
                  |clip=load_vision_encoder($model,family="svd",backend=$backend)
                  |encoder=load_vae($model,family="sd1",backend=$backend)
                  |decoder=load_vae($model,family="svd",backend=$backend)
                  |rng=make_rng("cpu",seed=$seed)
                  |image_tokens=tensor::reshape(encode_vision(clip,input,pooled=true),[1024,1,1])
                  |tokens=tensor::broadcast_to(image_tokens,[1024,1,$frames])
                  |pixels=image::to_f32(input)
                  |noisy=tensor::add(pixels,tensor::mul(tensor::randn_like(pixels,rng=rng),${noiseAug/2}))
                  |cond=tensor::broadcast_to(vae_encode(encoder,noisy,rng=rng,posterior="mode"),$shape)
                  |ids=tensor::broadcast_to(tensor::reshape(${vector(ids)},[768,1]),[768,$frames])
                  |pos=predictor(unet,{crossattn=tokens,vector=ids},prediction="edm_v",concat=cond)
                  |neg=predictor(unet,{crossattn=tensor::zeros_like(tokens),vector=ids},prediction="edm_v",concat=tensor::zeros_like(cond))
                  |ramp=tensor::reshape($guidance,[1,1,1,$frames])
                  |base=tensor::zeros($shape)
                  |state0=add_noise(pos,base,tensor::randn_like(base,rng=rng),sigma=${sigmas.head})
                  |""".stripMargin
    for i <- 0 until steps do
      val hi = sigmas(i)
      script ++= s"""positive$i=sample(state$i,tensor::vector([$hi,0]),predict=pos,method="euler",rng=rng)
                    |negative$i=sample(state$i,tensor::vector([$hi,0]),predict=neg,method="euler",rng=rng)
                    |guided$i=tensor::lerp(negative$i,positive$i,ramp)
                    |state${i+1}=tensor::lerp(state$i,guided$i,${1-sigmas(i+1)/hi})
                    |""".stripMargin
    val chunks = (0 until frames by chunk).map { i =>
      val stop = math.min(frames,i+chunk)
      script ++= s"decoded$i=vae_decode(decoder,unet_to_vae_latents(decoder,state$steps[:,:,:,$i:$stop]))\n"
      script ++= s"chunk$i=tensor::reshape(decoded$i,[3,$w,$h,${stop-i}])\n"
      s"chunk$i"
    }
    val output = if chunks.size == 1 then chunks.head else s"tensor::concat(${chunks.mkString("[",",","]")},axis=3)"
    script ++= s"return {frames=$output}\n}"
    check()
    var finishedBranches = 0
    val callback: VulkanInference.Progress = (stage, done, total) =>
      if stage == "sampling" && done == 1 && total == 1 then
        finishedBranches += 1
        if finishedBranches % 2 == 0 then progress(finishedBranches / 2, steps)
    val result = engine.dglFrames(script.result(),Map("input" -> source).asJava,"frames",callback).asScala.toVector
    check()
    require(result.size == frames, "SVD decoder returned an incomplete video")
    result