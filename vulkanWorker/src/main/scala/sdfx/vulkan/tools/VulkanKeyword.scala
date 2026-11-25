package sdfx.vulkan.tools

import java.nio.file.Path
import sdfx.core.model.LoraRef
import sdfx.core.protocol.{GenerationRequest, LoadedModel}
import sdfx.vulkan.VulkanProtocol
import ujson.{Arr, Obj, Value}

private[vulkan] object VulkanKeyword:
  case class Image(prompt: String, replay: String, source: String, loras: Seq[LoraRef])
  case class Plan(images: IndexedSeq[Image], commit: () => Unit)

  def plan(req: GenerationRequest, loaded: LoadedModel, session: VulkanTools.Session, seed: Long,
      cancelled: () => Boolean): Option[Plan] =
    val p = req.params
    p.hooks.get("model_keyword").filterNot(_.obj.get("enabled").contains(ujson.Bool(false))).map { config =>
      val options = config.obj
      val prompt = options.get("_base_prompt").fold(p.prompt)(_.str)
      val allowed = Set("enabled", "placement", "model_choice", "embedding", "order", "lora", "lora_weight",
        "multiplier", "lora_choice", "lora_name", "_base_prompt", "_mapping_dir", "_lora_path")
      require(options.keySet.forall(allowed), "unknown model_keyword option")
      require(!options.get("lora").flatMap(_.strOpt).exists(s => s.nonEmpty && s != "None") ||
        options.contains("_lora_path"), "selected model keyword LoRA requires a resolved path")
      val request = Obj("checkpoint" -> loaded.path, "mapping_dir" -> options.getOrElse("_mapping_dir",
        VulkanProtocol.invalid("model_keyword requires _mapping_dir")), "preview" -> prompt,
        "prompts" -> Arr.from(Vector.fill(p.imagesPerJob)(prompt)), "batch_size" -> p.batchSize,
        "choice_seed" -> ujson.Str(seed.toString), "options" -> config)
      options.get("_lora_path").foreach(path => request("lora") = path)
      val args = NativeJson.parse(request.render()).asInstanceOf[NativeJson.Obj].fields
        .updated("choice_seed", NativeJson.Number(BigDecimal(seed)))
      val context = new NativeWorkContext(cancelled, () => (), (_,_) => ())
      val (result, state) = NativeKeywordMappings.planRequest(args, Path.of(loaded.path),
        () => VulkanProtocol.unsupported("directory-format checkpoint"), session.keywordState, context)
      val value = ujson.read(NativeJson.render(NativeJson.Obj(result)))
      val prompts = value("prompts").arr
      val images = prompts.indices.map { i =>
        val start = i - i % p.batchSize
        val batch = value("batches")(i / p.batchSize)
        val clean = batch("clean_prompts")(i % p.batchSize).str
        val adapters = batch("adapters").arr.map { a =>
          def num(k: String): Option[Double] = a.obj.get(k).flatMap(_.numOpt)
          LoraRef(a("name").str, a("te").num, num("unet"), a.obj.get("lbw").flatMap(_.strOpt),
            num("start"), num("stop"), num("dyn").map(_.toInt))
        }
        val manual = req.manualLoras.getOrElse(p.loras)
        Image(clean, NativeLoraPrompt.replay(prompts(start).str,clean), prompts(i).str, manual ++ adapters)
      }
      Plan(images, () => session.keywordState = state)
    }
