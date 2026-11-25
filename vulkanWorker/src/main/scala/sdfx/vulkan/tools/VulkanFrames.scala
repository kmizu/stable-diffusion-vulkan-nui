package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.nio.file.Path
import sdfx.core.protocol.{GenerationRequest, IntermediateArtifact}
import ujson.Obj

private[vulkan] final class VulkanFrames(req: GenerationRequest, val config: SaveIntermediateConfig, cancelled: () => Boolean, steps: Option[StepsAnimationConfig]):
  private val p = req.params
  private val metadata = Obj("prompt" -> p.prompt,"negative_prompt" -> p.negativePrompt,"steps" -> p.steps,
    "cfg_scale" -> p.cfgScale,"sampler" -> p.samplerName,"mode" -> (if p.isImg2Img then "img2img" else "txt2img"))
  private val session = new NativeIntermediateRecorder.Session(Path.of(req.outputDir),config,
    NativeJson.parse(metadata.render()).asInstanceOf[NativeJson.Obj].fields,cancelled,steps)
  private var seed = 0L
  private var index = 0
  private var pass = -1
  private var offset = 0
  private var total = 0
  @volatile private var failure = Option.empty[Throwable]
  def begin(i: Int, s: Long): Unit =
    index = i
    seed = s
    pass = -1
    offset = 0
    total = 0
  def frame(step: Int, samplePass: Int, passSteps: Int, image: BufferedImage): Unit =
    if failure.nonEmpty || cancelled() then return
    try
      if samplePass != pass then
        offset += total
        pass = samplePass
        total = passSteps
      session.observeImage(image,if offset==0 then "base" else "hires",step,offset,seed,index/p.batchSize,index%p.batchSize)
    catch case scala.util.control.NonFatal(e) => failure = Some(e)
  def finalImage(image: BufferedImage): Unit =
    failure.foreach(e => throw e)
    session.recordFinal(image,seed,index/p.batchSize,index%p.batchSize)
  def finish(): Seq[IntermediateArtifact] =
    failure.foreach(e => throw e)
    session.finish().map { value =>
      val fields = value.asInstanceOf[NativeJson.Obj].fields
      def text(k: String) = fields(k).asInstanceOf[NativeJson.Str].value
      IntermediateArtifact(text("path"),text("type"))
    }
  def incomplete(secondInterrupt: () => Boolean): Unit = session.closeIncomplete(secondInterrupt)

private[vulkan] object VulkanFrames:
  def start(req: GenerationRequest,cancelled: () => Boolean): Option[VulkanFrames] =
    val hooks = Some(NativeJson.Obj(req.params.hooks.map((k,v) => k -> NativeJson.parse(v.render()))))
    val save = NativeIntermediateRecorder.parseSaveIntermediate(hooks)
    val animation = NativeIntermediateRecorder.parseStepsAnimation(hooks, req.params.batchSize, req.params.nIter, save.isDefined)
    val config = save.orElse(animation.map(s => SaveIntermediateConfig(everyN = 1, startAt = s.skipInitial + 1)))
    config.map(new VulkanFrames(req,_,cancelled,animation))
