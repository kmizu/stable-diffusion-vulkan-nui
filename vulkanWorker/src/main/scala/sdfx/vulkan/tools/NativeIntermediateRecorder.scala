// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Base64
import javax.imageio.ImageIO

object NativeIntermediateRecorder:
  import NativeJson.*

  private def hookFields(hooksOpt: Option[NativeJson], name: String): Option[Map[String, NativeJson]] =
    hooksOpt match
      case Some(Obj(hooks)) =>
        hooks.get(name).flatMap {
          case Null | Bool(false) => None
          case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
          case Bool(true) => Some(Map.empty[String, NativeJson])
          case Obj(fields) => Some(fields)
          case _ => throw IllegalArgumentException(s"invalid $name hook")
        }
      case _ => None

  private def str(fields: Map[String, NativeJson], key: String, default: String): String =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Str(v)) => v
      case _ => throw IllegalArgumentException(s"$key must be a string")

  private def num(fields: Map[String, NativeJson], key: String, default: Double): Double =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Number(v)) => v.toDouble
      case Some(Str(s)) =>
        try s.trim.toDouble
        catch case _: NumberFormatException => throw IllegalArgumentException(s"$key must be numeric")
      case _ => throw IllegalArgumentException(s"$key must be numeric")

  private def intVal(fields: Map[String, NativeJson], key: String, default: Int): Int =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Number(v)) if v.isValidInt => v.toInt
      case Some(Str(s)) =>
        try s.trim.toInt
        catch case _: NumberFormatException => throw IllegalArgumentException(s"$key must be an integer")
      case _ => throw IllegalArgumentException(s"$key must be an integer")

  private def boolVal(fields: Map[String, NativeJson], key: String, default: Boolean): Boolean =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Bool(v)) => v
      case Some(Number(v)) => v != 0
      case Some(Str(s)) => Set("1", "true", "yes", "on").contains(s.trim.toLowerCase(java.util.Locale.ROOT))
      case _ => default

  def parseSaveIntermediate(hooksOpt: Option[NativeJson]): Option[SaveIntermediateConfig] =
    hookFields(hooksOpt, "save_intermediate").map { fields =>
      val allowed = Set(
        "enabled", "every_n", "start_at", "stop_at", "image_type", "mode", "final_save",
        "format", "codec", "fps", "add_first_frames", "add_last_frames", "smooth", "seconds",
        "lores", "hires", "ffmpeg_bat", "bat_only", "output_dir"
      )
      require(fields.keySet.subsetOf(allowed), "unknown save_intermediate option")
      val raw = SaveIntermediateConfig(
        everyN = intVal(fields, "every_n", 5),
        startAt = intVal(fields, "start_at", 0),
        stopAt = intVal(fields, "stop_at", 0),
        imageType = str(fields, "image_type", "Denoised"),
        mode = str(fields, "mode", "Standard"),
        finalSave = boolVal(fields, "final_save", false),
        format = str(fields, "format", "mp4"),
        codec = str(fields, "codec", "h264"),
        fps = num(fields, "fps", 2.0),
        addFirstFrames = intVal(fields, "add_first_frames", 0),
        addLastFrames = intVal(fields, "add_last_frames", 0),
        smooth = boolVal(fields, "smooth", false),
        seconds = num(fields, "seconds", 0.0),
        lores = boolVal(fields, "lores", true),
        hires = boolVal(fields, "hires", true),
        ffmpegBat = boolVal(fields, "ffmpeg_bat", false),
        batOnly = boolVal(fields, "bat_only", false),
        outputDirOverride = fields.get("output_dir").collect { case Str(s) if s.trim.nonEmpty => s.trim }
      )
      SaveIntermediateConfig.validate(raw)
    }

  def parseStepsAnimation(
      hooksOpt: Option[NativeJson],
      batchSize: Int,
      nIter: Int,
      hasSaveIntermediate: Boolean
  ): Option[StepsAnimationConfig] =
    hookFields(hooksOpt, "steps_animation").map { fields =>
      require(batchSize == 1 && nIter == 1, "steps animation requires batch size and count equal to one")
      require(!hasSaveIntermediate, "steps animation cannot run together with Save intermediate images")
      val allowed = Set(
        "enabled", "duration", "fps", "codec", "interpolation", "skip_initial",
        "debug", "run_incomplete", "delete_intermediate", "create_output", "frames_dir", "output_dir"
      )
      require(fields.keySet.subsetOf(allowed), "unknown steps_animation option")
      val raw = StepsAnimationConfig(
        duration = num(fields, "duration", 5.0),
        fps = intVal(fields, "fps", 30),
        codec = str(fields, "codec", "h264"),
        interpolation = str(fields, "interpolation", "blend"),
        skipInitial = intVal(fields, "skip_initial", 0),
        debug = boolVal(fields, "debug", false),
        runIncomplete = boolVal(fields, "run_incomplete", false),
        deleteIntermediate = boolVal(fields, "delete_intermediate", false),
        createOutput = boolVal(fields, "create_output", false),
        framesDir = str(fields, "frames_dir", "steps-animation/frames"),
        outputDir = str(fields, "output_dir", "steps-animation/videos")
      )
      StepsAnimationConfig.validate(raw)
    }

  final class Session(
      outputDir: Path,
      val effectiveSave: SaveIntermediateConfig,
      metadata: Map[String, NativeJson],
      cancelled: () => Boolean,
      steps: Option[StepsAnimationConfig] = None
  ):
    val directory: Path =
      val base = effectiveSave.outputDirOverride.map(Path.of(_).toAbsolutePath.normalize()).getOrElse(outputDir)
      val root = steps.fold(Files.createDirectories(base.resolve("intermediates")))(s => StepVideo.directory(base,s.framesDir))
      Files.createTempDirectory(root, "generation-")

    private var frameCounter = 0
    private val frames = scala.collection.mutable.ArrayBuffer.empty[IntermediateFrameRecord]
    private val artifacts = scala.collection.mutable.ArrayBuffer.empty[IntermediateArtifact]
    private var finished = false

    private def nextOutputPath(suffix: String): Path =
      val idx = frameCounter
      frameCounter += 1
      directory.resolve(f"$idx%05d-$suffix.png")

    private def frameToJson(f: IntermediateFrameRecord): Map[String, NativeJson] =
      Map(
        "path" -> Str(f.path),
        "seed" -> Number(BigDecimal(f.seed)),
        "pass_name" -> Str(f.passName),
        "step" -> f.step.fold[NativeJson](Null)(s => Number(s)),
        "pass_step" -> f.passStep.fold[NativeJson](Null)(s => Number(s)),
        "iteration" -> Number(f.iteration),
        "batch_index" -> Number(f.batchIndex),
        "width" -> Number(f.width),
        "height" -> Number(f.height),
        "image_type" -> Str(f.imageType)
      )

    private def artifactToJson(a: IntermediateArtifact): NativeJson =
      val base = scala.collection.mutable.LinkedHashMap[String, NativeJson](
        "type" -> Str(a.kind),
        "path" -> Str(a.path)
      )
      a.seed.foreach(v => base("seed") = Number(BigDecimal(v)))
      a.passName.foreach(v => base("pass_name") = Str(v))
      if a.kind == "image" then
        base("step") = a.step.fold[NativeJson](Null)(s => Number(s))
        base("pass_step") = a.passStep.fold[NativeJson](Null)(s => Number(s))
      a.iteration.foreach(v => base("iteration") = Number(v))
      a.batchIndex.foreach(v => base("batch_index") = Number(v))
      a.width.foreach(v => base("width") = Number(v))
      a.height.foreach(v => base("height") = Number(v))
      a.imageType.foreach(v => base("image_type") = Str(v))
      a.frames.foreach(v => base("frames") = Number(v))
      a.format.foreach(v => base("format") = Str(v))
      Obj(base.toMap)

    private def savePng(
        image: BufferedImage,
        seed: Long,
        passName: String,
        passStep: Option[Int],
        absStep: Option[Int],
        iteration: Int,
        batchIndex: Int
    ): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("intermediate capture cancelled")
      val stepLabel = absStep.map(_.toString).getOrElse("final")
      val suffix =
        if effectiveSave.mode == "Suffix" then s"$seed-$passName-$stepLabel"
        else s"$passName-$stepLabel-$seed"
      val path = nextOutputPath(suffix)
      val imgType = if passName == "final" then "Final" else effectiveSave.imageType
      val record = IntermediateFrameRecord(
        path = path.toString,
        seed = seed,
        passName = passName,
        step = absStep,
        passStep = passStep,
        iteration = iteration,
        batchIndex = batchIndex,
        width = image.getWidth,
        height = image.getHeight,
        imageType = imgType
      )
      val sdfxJson = NativeJson.render(Obj(metadata ++ frameToJson(record)))
      val promptStr = metadata.get("prompt").collect { case Str(s) => s }.getOrElse("")
      val negStr = metadata.get("negative_prompt").collect { case Str(s) => s }.getOrElse("")
      val stepsVal = metadata.get("steps").collect { case Number(n) => n.toInt }.getOrElse(0)
      val samplerStr = metadata.get("sampler").collect { case Str(s) => s }.getOrElse("")
      val cfgVal = metadata.get("cfg_scale").collect { case Number(n) => n.toString }.getOrElse("7")
      val paramLines =
        promptStr + (if negStr.nonEmpty then s"\nNegative prompt: $negStr" else "") + "\n" +
          Vector(
            Some(s"Steps: $stepsVal"),
            Some(s"Sampler: $samplerStr"),
            Some(s"CFG scale: $cfgVal"),
            Some(s"Seed: $seed"),
            Some(s"Size: ${image.getWidth}x${image.getHeight}"),
            Some(s"Intermediate pass: $passName"),
            absStep.map(s => s"Intermediate step: $s"),
            Some(s"Intermediate subject: $imgType")
          ).flatten.mkString(", ")
      IntermediatePngWriter.writePngWithMetadata(
        image,
        path,
        Vector("sdfx_intermediate" -> sdfxJson, "parameters" -> paramLines)
      )
      if steps.exists(_.debug) then System.err.println(s"Steps Animation captured $passName step ${absStep.getOrElse(0)}")
      frames += record
      artifacts += IntermediateArtifact(
        kind = "image",
        path = record.path,
        seed = Some(record.seed),
        passName = Some(record.passName),
        step = record.step,
        passStep = record.passStep,
        iteration = Some(record.iteration),
        batchIndex = Some(record.batchIndex),
        width = Some(record.width),
        height = Some(record.height),
        imageType = Some(record.imageType)
      )

    def observeImage(image: BufferedImage, passName: String, step: Int, offset: Int, seed: Long, iteration: Int, batchIndex: Int): Unit =
      if effectiveSave.isSelected(step + offset) then
        savePng(image,seed,passName,Some(step),Some(step+offset),iteration,batchIndex)

    def recordFinal(image: BufferedImage, seed: Long, iteration: Int, batchIndex: Int): Unit =
      if effectiveSave.finalSave then
        savePng(image, seed, "final", None, None, iteration, batchIndex)

    private def timeline(group: Vector[IntermediateFrameRecord]): Vector[IntermediateFrameRecord] =
      val selected = group.filter { f =>
        f.passName == "final" ||
        (f.passName == "hires" && effectiveSave.hires) ||
        (f.passName == "base" && effectiveSave.lores)
      }
      if selected.isEmpty then Vector.empty
      else
        Vector.fill(effectiveSave.addFirstFrames)(selected.last) ++ selected ++
          Vector.fill(effectiveSave.addLastFrames)(selected.last)

    private def videoTiming(count: Int): (Double, Double) =
      val duration =
        if effectiveSave.smooth then math.max(count.toDouble, effectiveSave.seconds)
        else if effectiveSave.seconds > 0.0 then effectiveSave.seconds
        else count.toDouble / effectiveSave.fps
      (count.toDouble / duration, duration)

    private def writeRecipe(part: Vector[IntermediateFrameRecord], name: String): Unit =
      val listing = directory.resolve(s"$name.ffconcat")
      val (rate, duration) = videoTiming(part.size)
      val lines = scala.collection.mutable.ArrayBuffer[String]("ffconcat version 1.0")
      part.foreach { f =>
        val fname = Path.of(f.path).getFileName.toString
        lines += s"file '$fname'"
        lines += f"duration ${1.0 / rate}%.9f"
      }
      lines += s"file '${Path.of(part.last.path).getFileName}'"
      Files.writeString(listing, lines.mkString("\n") + "\n", StandardCharsets.UTF_8)
      val bat = directory.resolve(s"$name.bat")
      val target = s"$name.${effectiveSave.format}"
      val batContent =
        s"@echo off\ncd /d \"%~dp0\"\nffmpeg -n -f concat -safe 1 -i ${listing.getFileName} -t $duration $target\n"
      Files.writeString(bat, batContent, StandardCharsets.UTF_8)
      artifacts += IntermediateArtifact(kind = "ffmpeg_recipe", path = bat.toString)
      artifacts += IntermediateArtifact(kind = "ffmpeg_playlist", path = listing.toString)

    private def encodeVideo(sequence: Vector[IntermediateFrameRecord], name: String): Path =
      val (rate,duration) = videoTiming(sequence.size)
      val input = directory.resolve(s"$name.ffconcat")
      val lines = sequence.flatMap { frame =>
        val filename = Path.of(frame.path).getFileName
        Seq(s"file '$filename'",f"duration ${1.0/rate}%.9f")
      }
      Files.writeString(input, (Seq("ffconcat version 1.0") ++ lines :+ s"file '${Path.of(sequence.last.path).getFileName}'").mkString("\n")+"\n")
      val dest = directory.resolve(s"$name.${effectiveSave.format}")
      val width = ((sequence.map(_.width).max+1)/2)*2
      val height = ((sequence.map(_.height).max+1)/2)*2
      val resize = s"scale=$width:$height:force_original_aspect_ratio=decrease,pad=$width:$height:(ow-iw)/2:(oh-ih)/2"
      val timing = if effectiveSave.smooth then s"minterpolate=fps=${effectiveSave.fps}" else s"fps=${effectiveSave.fps}"
      val codec = effectiveSave.format match
        case "gif" => Seq.empty
        case "webm" => Seq("-c:v","libvpx-vp9","-pix_fmt","yuv420p")
        case _ => Seq("-c:v",Map("h264"->"libx264","h265/hevc"->"libx265","av1"->"libaom-av1")(effectiveSave.codec),"-pix_fmt","yuv420p")
      val command = Seq("ffmpeg","-nostdin","-v","error","-n","-f","concat","-safe","1","-i",input.toString,
        "-t",duration.toString,"-vf",s"$resize,$timing") ++ codec :+ dest.toString
      val log = directory.resolve(s"$name-ffmpeg.log")
      val process = new ProcessBuilder(command*).redirectErrorStream(true).redirectOutput(log.toFile).start()
      try
        while !process.waitFor(150,java.util.concurrent.TimeUnit.MILLISECONDS) do
          if cancelled() then throw java.util.concurrent.CancellationException("video export interrupted")
        require(process.exitValue()==0,s"ffmpeg failed; see $log")
        dest
      catch case e: Throwable =>
        process.destroyForcibly().waitFor()
        Files.deleteIfExists(dest)
        throw e

    def finish(): Vector[NativeJson] = finish("complete", cancelled)

    private def finish(status: String, stop: () => Boolean): Vector[NativeJson] =
      if finished then return artifacts.toVector.map(artifactToJson)
      if frames.isEmpty && steps.isEmpty then
        Files.deleteIfExists(directory)
        finished = true
        return Vector.empty
      if effectiveSave.mode == "Video" && (effectiveSave.ffmpegBat || effectiveSave.batOnly) then
        val groups = frames.toVector.map(f => (f.iteration, f.batchIndex, f.seed)).distinct.sorted
        for (iter, bIdx, sd) <- groups do
          val grp = frames.toVector.filter(f => (f.iteration, f.batchIndex, f.seed) == (iter, bIdx, sd))
          val name = s"iteration-$iter-image-$bIdx-seed-$sd"
          for passName <- Vector("base", "hires") do
            if grp.exists(_.passName == passName) then
              val part = timeline(grp.filter(f => f.passName == passName || f.passName == "final"))
              if part.nonEmpty then writeRecipe(part, s"$name-$passName")
      if effectiveSave.mode == "Video" && !effectiveSave.batOnly then
        val groups = frames.toVector.groupBy(f => (f.iteration, f.batchIndex, f.seed)).toVector.sortBy(_._1)
        groups.foreach { case ((iteration, index, seed), group) =>
          val sequence = timeline(group)
          if sequence.nonEmpty then
            val name = s"iteration-$iteration-image-$index-seed-$seed"
            val video = encodeVideo(sequence, name)
            artifacts += IntermediateArtifact("video",video.toString,Some(seed),frames=Some(sequence.size),format=Some(effectiveSave.format))
        }
      val errors = scala.collection.mutable.ArrayBuffer.empty[String]
      for settings <- steps if settings.createOutput && frames.nonEmpty do
        try
          val output = StepVideo.encode(outputDir, directory, frames.toVector, settings, stop)
          artifacts += IntermediateArtifact("video", output.toString, frames = Some(frames.size), format = Some(StepVideo.extension(settings.codec)))
          if settings.deleteIntermediate then
            frames.foreach(f => Files.delete(Path.of(f.path)))
            artifacts.filterInPlace(_.kind != "image")
        catch
          case e: java.util.concurrent.CancellationException if status == "complete" => throw e
          case scala.util.control.NonFatal(e) =>
            val message = s"Steps Animation retained its frames after export failed: ${e.getMessage}"
            errors += message
            System.err.println(message)
      val manifest = directory.resolve("manifest.json")
      val manifestObj = Obj(
        Map(
          "status" -> Str(if errors.isEmpty then status else status + "_with_export_errors"),
          "metadata" -> Obj(metadata),
          "frames" -> Arr(frames.toVector.map(f => Obj(frameToJson(f)))),
          "artifacts" -> Arr(artifacts.toVector.map(artifactToJson)),
          "export_errors" -> Arr(errors.toVector.map(Str.apply))
        )
      )
      Files.writeString(manifest, NativeJson.render(manifestObj) + "\n", StandardCharsets.UTF_8)
      artifacts += IntermediateArtifact(kind = "manifest", path = manifest.toString)
      finished = true
      artifacts.toVector.map(artifactToJson)

    def closeIncomplete(secondInterrupt: () => Boolean): Unit =
      if !finished && steps.exists(_.runIncomplete) then
        val deadline = System.nanoTime() + 60000000000L
        try finish("incomplete", () => secondInterrupt() || System.nanoTime() - deadline >= 0)
        catch case scala.util.control.NonFatal(e) => System.err.println(s"Steps Animation partial export failed: ${e.getMessage}")
      if !finished then
        if frames.isEmpty then
          try Files.deleteIfExists(directory) catch case _: Throwable => ()
        else
          try
            val manifest = directory.resolve("manifest.json")
            val manifestObj = Obj(
              Map(
                "status" -> Str("incomplete"),
                "metadata" -> Obj(metadata),
                "frames" -> Arr(frames.toVector.map(f => Obj(frameToJson(f)))),
                "artifacts" -> Arr(artifacts.toVector.map(artifactToJson))
              )
            )
            Files.writeString(manifest, NativeJson.render(manifestObj) + "\n", StandardCharsets.UTF_8)
          catch case _: Throwable => ()
        finished = true
