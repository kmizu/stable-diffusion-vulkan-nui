package sdfx.engine

import java.awt.image.BufferedImage
import java.io.{BufferedReader, InputStreamReader, OutputStream, OutputStreamWriter}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.UUID
import javax.imageio.ImageIO

/** Lightweight pure-JVM JSONL protocol v1 dry-run worker so NUI `--mcp --dry-run` works with zero Python or CUDA/checkpoint dependencies. */
object JvmDryRunWorker:
  val Features: Seq[String] = Seq(
    "img2img_hires_fix", "loopback_scaler", "tokenizer", "latent_upscale", "inpaint_difference",
    "gpu_temperature_guard", "rescale_cfg", "cfg_schedule", "cfg_cache", "postdiff", "sada", "tgate", "resharpen",
    "latent_mirroring", "color_grading", "anti_burn", "vectorscope_cc", "detail_daemon", "vae_tools",
    "model_keyword", "freeu", "deepcache", "dynamic_thresholding", "tiling", "cads", "hypertile",
    "metadata_utils", "model_sampling", "apg", "mahiro", "pag", "weight_gradient", "negpip",
    "kohya_hrfix", "token_downsampling", "conditioning_highres_fix", "multidiffusion", "latent_modifier",
    "soft_inpainting", "neveroom", "semantic_guidance", "sonar", "hires_fix_tweaks", "pnginfo_injection",
    "prevent_interrupt", "controlnet", "system_info", "save_intermediate", "steps_animation",
    "img2imgalt", "cd_tuner", "llul", "agent_attention", "incantations", "characteristic_guidance",
    "clip_io", "prompt_fusion", "neutral_prompt", "composable_lora", "embedding_merge",
    "aesthetic_gradients", "sag", "reference", "revision", "ip_adapter", "controllllite",
    "fooocus_inpaint", "compile", "taesd", "spandrel", "vram_estimator", "adetailer",
    "batch_face_swap", "face_restore", "nsfw_checker", "aesthetic_score", "animatediff", "photomaker",
    "caption", "img2vid", "novel_view", "custom_code", "ldsr", "marigold"
  )

  private val SamplersResult: ujson.Obj =
    def sampler(name: String, schedule: String) = ujson.Obj("name" -> name, "supported" -> true, "default_scheduler" -> schedule)
    ujson.Obj(
      "samplers" -> ujson.Arr(
        sampler("DPM++ 2M", "Karras"), sampler("DPM++ SDE", "Karras"), sampler("DPM++ 2M SDE", "Exponential"),
        sampler("DPM++ 2M SDE Heun", "Exponential"), sampler("DPM++ 2S a", "Karras"), sampler("DPM++ 3M SDE", "Exponential"),
        sampler("Euler a", "Uniform"), sampler("Euler", "Uniform"), sampler("LMS", "Uniform"), sampler("Heun", "Uniform"),
        sampler("DPM2", "Karras"), sampler("DPM2 a", "Karras"), sampler("DDIM", "Uniform"), sampler("UniPC", "Uniform"), sampler("LCM", "Uniform")
      ),
      "schedulers" -> ujson.Arr("Automatic", "Uniform", "Karras", "Exponential", "Polyexponential", "Normal", "Simple", "SGM Uniform", "Align Your Steps", "KL Optimal", "DDIM", "Beta"),
      "upscalers" -> ujson.Arr("Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (bicubic antialiased)", "Latent (nearest)", "Latent (nearest-exact)", "Nearest", "Lanczos", "R-ESRGAN 4x+"),
      "preprocessors" -> ujson.Arr("none", "invert", "inpaint", "canny")
    )

  def helloObject(version: String): ujson.Obj = ujson.Obj(
    "id" -> ujson.Null,
    "type" -> "hello",
    "protocol" -> 1,
    "engine_version" -> version,
    "python" -> "",
    "cuda" -> false,
    "device" -> "cpu",
    "dry_run" -> true,
    "features" -> ujson.Arr.from(Features)
  )

  def main(args: Array[String]): Unit =
    serve(new BufferedReader(new InputStreamReader(System.in, UTF_8)), System.out)

  def serve(in: BufferedReader, out: OutputStream, version: String = "vulkan-jvm-dryrun"): Unit =
    val writer = new OutputStreamWriter(out, UTF_8)
    def emit(js: ujson.Value): Unit = synchronized {
      writer.write(ujson.write(js))
      writer.write('\n')
      writer.flush()
    }
    def emitResult(id: String, res: ujson.Value): Unit =
      emit(ujson.Obj("id" -> id, "type" -> "result", "result" -> res))
    def emitEvent(id: String, name: String, fields: Seq[(String, ujson.Value)]): Unit =
      emit(ujson.Obj.from(fields ++ Seq("id" -> ujson.Str(id), "type" -> ujson.Str("event"), "event" -> ujson.Str(name))))
    def emitError(id: Option[String], code: String, msg: String): Unit =
      emit(ujson.Obj("id" -> id.fold[ujson.Value](ujson.Null)(ujson.Str(_)), "type" -> "error", "code" -> code, "message" -> msg, "detail" -> ""))

    emit(helloObject(version))
    var modelSummary: Option[ujson.Obj] = None
    var protectedRev = 0L
    var running = true
    while running do
      val line = in.readLine()
      if line == null then running = false
      else if line.trim.nonEmpty then
        var reqId: Option[String] = None
        try
          val msg = ujson.read(line).obj
          val id = msg("id").str
          reqId = Some(id)
          val kind = msg("type").str
          val params = msg.get("params").collect { case o: ujson.Obj => o }.getOrElse(ujson.Obj())
          kind match
            case "ping" =>
              val h = helloObject(version)
              h.value.remove("id")
              h.value.remove("type")
              emitResult(id, h)
            case "shutdown" =>
              emitResult(id, ujson.Obj())
              running = false
            case "system_status" =>
              emitResult(id, ujson.Obj(
                "loaded" -> modelSummary.nonEmpty,
                "device" -> "cpu",
                "vram_mode" -> "cpu",
                "dry_run" -> true
              ))
            case "list_samplers" =>
              emitResult(id, SamplersResult)
            case "list_models" =>
              emitResult(id, ujson.Obj("models" -> ujson.Arr()))
            case "list_files" =>
              emitResult(id, ujson.Obj("files" -> ujson.Arr()))
            case "memory" =>
              emitResult(id, ujson.Obj("allocated_mb" -> 0, "reserved_mb" -> 0, "ram_used_mb" -> 0, "total_mb" -> 0))
            case "interruption_protection" =>
              protectedRev += 1
              emitResult(id, ujson.Obj("enabled" -> false, "revision" -> protectedRev))
            case "restart_authorization" | "interrupt" =>
              emitResult(id, ujson.Obj("accepted" -> true, "blocked" -> false, "affected" -> false, "too_late" -> false,
                "protection" -> ujson.Obj("enabled" -> false, "revision" -> protectedRev), "active_id" -> ujson.Null))
            case "load_model" =>
              val p = params.value.get("path").collect { case ujson.Str(s) => s }.filter(_.nonEmpty).getOrElse("dryrun-model")
              val name = Option(Path.of(p).getFileName).map(_.toString).getOrElse(p)
              val loaded = ujson.Obj("name" -> name, "path" -> p, "kind" -> "safetensors", "dtype" -> "fp16",
                "hash" -> "00000000", "family" -> "sd", "model_token" -> UUID.randomUUID().toString)
              modelSummary = Some(loaded)
              emitResult(id, loaded)
            case "unload" =>
              modelSummary = None
              emitResult(id, ujson.Obj())
            case "set_vae" =>
              emitResult(id, modelSummary.getOrElse(ujson.Obj()))
            case "txt2img" | "img2img" =>
              val g = params.value.get("params").collect { case o: ujson.Obj => o }.getOrElse(ujson.Obj())
              val w = g.value.get("width").flatMap(_.numOpt).map(_.toInt).getOrElse(512).max(8).min(4096)
              val h = g.value.get("height").flatMap(_.numOpt).map(_.toInt).getOrElse(512).max(8).min(4096)
              val steps = g.value.get("steps").flatMap(_.numOpt).map(_.toInt).getOrElse(20).max(1)
              val seed = g.value.get("seed").flatMap(_.numOpt).map(_.toLong).getOrElse(42L)
              val batchSize = g.value.get("batch_size").flatMap(_.numOpt).map(_.toInt).getOrElse(1).max(1)
              val nIter = g.value.get("n_iter").flatMap(_.numOpt).map(_.toInt).getOrElse(1).max(1)
              val outDir = Path.of(params.value.get("output_dir").collect { case ujson.Str(s) => s }.filter(_.nonEmpty).getOrElse("outputs"))
              Files.createDirectories(outDir)
              val images = (0 until batchSize * nIter).map { idx =>
                emitEvent(id, "progress", Seq("step" -> steps, "total" -> steps, "pass" -> "base", "eta_s" -> 0,
                  "image" -> idx, "images" -> (batchSize * nIter), "batch_size" -> batchSize))
                val file = outDir.resolve(f"dryrun-$idx%04d-${UUID.randomUUID().toString.take(8)}.png")
                val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
                val g2 = img.createGraphics()
                try
                  g2.setPaint(new java.awt.GradientPaint(0f, 0f, new java.awt.Color(28, 36, 54), w.toFloat, h.toFloat, new java.awt.Color(44, 68, 108)))
                  g2.fillRect(0, 0, w, h)
                  g2.setColor(new java.awt.Color(255, 255, 255, 28))
                  val step = math.max(32, w / 8)
                  var x = step
                  while x < w do { g2.drawLine(x, 0, x, h); x += step }
                  var y = step
                  while y < h do { g2.drawLine(0, y, w, y); y += step }
                  g2.setColor(new java.awt.Color(230, 236, 245, 200))
                  g2.drawString(s"Dry-run ${w}x${h} (seed ${seed + idx})", 16, 28)
                finally g2.dispose()
                ImageIO.write(img, "png", file.toFile)
                ujson.Obj("path" -> file.toAbsolutePath.toString, "width" -> w, "height" -> h, "seed" -> (seed + idx), "infotext" -> "Steps: 20")
              }
              emitResult(id, ujson.Obj(
                "images" -> ujson.Arr.from(images),
                "timing_s" -> 0.01,
                "model_name" -> modelSummary.flatMap(_.value.get("name").flatMap(_.strOpt)).getOrElse("dryrun"),
                "scheduler" -> g.value.get("scheduler").flatMap(_.strOpt).getOrElse("Uniform"),
                "warnings" -> ujson.Arr(),
                "intermediate_artifacts" -> ujson.Arr()
              ))
            case other =>
              emitError(Some(id), "unsupported", s"unsupported dry-run request: $other")
        catch case e: Throwable =>
          emitError(reqId, "invalid_params", Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
