package sdfx.vulkan.tools

import java.nio.file.Path
import sdfx.core.protocol.*
import sdfx.core.model.{GenerationParams, Img2ImgParams, HiresParams}
import sdfx.vulkan.VulkanInference.DeviceMemory
import ujson.Value

private[vulkan] final class VulkanBenchmarks:
  import NativeJson.*
  private var lastProfile = Option.empty[NativeJson]

  def run(kind: String, raw: Value, status: Value, load: Option[LoadModelRequest], memory: () => DeviceMemory,
      cancelled: () => Boolean, event: (String, Value) => Unit,
      generate: (GenerationParams, Path) => GenerationResult): Value =
    def native(value: Value) = NativeJson.parse(ujson.write(value)).asInstanceOf[Obj].fields
    val params = native(raw)
    val sys = native(status)
    val ctx = NativeWorkContext(cancelled, () => (), (name, fields) => event(name, ujson.read(NativeJson.render(Obj(fields)))))
    def requireModel(): Unit = if load.isEmpty then throw EngineError("no_model", "Load a checkpoint before benchmarking")
    val result = if kind == "system_benchmark" then
      requireModel()
      NativeSystemBenchmark.run(params, sys, ctx) { (batch, w, h, steps, out, _) =>
        val p = GenerationParams(prompt = NativeSystemBenchmark.prompt, negativePrompt = NativeSystemBenchmark.negative,
          width = w, height = h, steps = steps, seed = 42, samplerName = "Euler a", nIter = batch)
        generate(p, out).images.size
      }
    else
      val current = memory()
      val profile = params.get("profile").filterNot(_ == Null).orElse(lastProfile)
      val runtime = ujson.Obj("backend" -> "vulkan", "batch_execution" -> "sequential")
      load.foreach { r =>
        runtime("dtype") = r.dtype
        runtime("vram_mode") = r.vramMode
        runtime("fp8_storage") = r.fp8Storage
        runtime("vae_tiling") = r.vaeTiling
        runtime("vae_fp16") = r.vaeFp16
      }
      val info = NativeVramEstimator.profile(sys("model").asInstanceOf[Obj].fields,
        status("device").str, current.total, Obj(native(runtime)), profile, load.nonEmpty && current.total > 0)
      def integer(key: String, default: Int): Int = params.get(key) match
        case None => default
        case Some(Number(n)) if n.isValidInt => n.toInt
        case _ => throw IllegalArgumentException(s"Invalid $key")
      raw.obj.get("operation").fold("benchmark")(_.str) match
        case "profile" => info
        case "estimate" => profile match
          case None => Map("available" -> Bool(false), "reason" -> Str("no_profile"))
          case Some(_) if info("compatible") != Bool(true) => Map("available" -> Bool(false), "reason" -> Str("profile_identity_mismatch"))
          case Some(record) =>
            NativeVramEstimator.estimate(record, raw.obj.get("mode").fold("txt2img")(_.str), integer("width",512),
              integer("height",512), integer("batch",1), (current.total-current.free)/1048576.0, current.total/1048576.0) ++
              Map("fingerprint" -> info("current_fingerprint"), "profile_status" -> record.asInstanceOf[Obj].fields("status"))
        case "benchmark" =>
          requireModel()
          require(current.total > 0, "Vulkan memory budget unavailable")
          NativeVramEstimator.benchmark(params,info("current_identity"),info("current_fingerprint").asInstanceOf[Str].value,
            memory,ctx,v => lastProfile=Some(v)) { (c, _, input, out) =>
            val p = GenerationParams(prompt="an architectural city scene, daylight, detailed",negativePrompt="blurry, monochrome",
              width=c.width,height=c.height,seed=42,steps=integer("steps",1),batchSize=c.batch,samplerName="Euler",scheduler="Uniform",
              img2img=Option.when(c.mode=="img2img")(Img2ImgParams(initImage=input.toString)),
              hires=c.upscaler.fold(HiresParams())(name => HiresParams(enabled=true, upscaler=name,resizeX=c.target,resizeY=c.target)))
            val images=generate(p,out).images
            require(images.size==c.batch && images.forall(i => i.width==c.target && i.height==c.target),"Unexpected calibration image dimensions")
          }.asInstanceOf[Obj].fields
        case other => throw IllegalArgumentException(s"Unknown VRAM operation: $other")
    ujson.read(NativeJson.render(Obj(result)))
