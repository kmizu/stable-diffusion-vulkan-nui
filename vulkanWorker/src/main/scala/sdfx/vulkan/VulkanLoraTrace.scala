package sdfx.vulkan

import java.awt.{BasicStroke, Color, Font, RenderingHints}
import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.Base64
import scala.collection.mutable
import scala.util.Using
import sdfx.core.png.ScrambledPng
import sdfx.core.protocol.{ComposableLoraTrace, GeneratedImage}
import ujson.Obj

private[vulkan] object VulkanLoraTrace:
  def label(pass: String, branch: String, prompt: String, negative: Boolean,
      weights: Seq[(String, Double)], bindings: Seq[String]): String =
    val payload = Obj("pass_name" -> pass, "branch" -> branch, "prompt" -> prompt,
      "is_negative" -> negative, "weights" -> Obj.from(weights.map((name, value) => name -> ujson.Num(value))),
      "bindings" -> ujson.Arr.from(bindings.map(ujson.Str(_))))
    Base64.getUrlEncoder.withoutPadding().encodeToString(payload.render().getBytes(UTF_8))

  private case class Series(image: Int, pass: String, branch: String, prompt: String, negative: Boolean, lora: String)
  private case class Prediction(pass: String, branch: String, prompt: String, negative: Boolean,
      weights: Vector[(String, Double)], bindings: Vector[String])
  private case class Evaluations(models: Int, layers: Option[Vector[Long]]):
    def add(other: Evaluations): Evaluations =
      Evaluations(models + other.models, layers.zip(other.layers).map((a, b) => a.zip(b).map(_ + _)))
  private case class Samples(sum: Double, minimum: Double, maximum: Double, count: Int, layers: Option[Long]):
    def add(value: Double, times: Int, extraLayers: Option[Long]): Samples =
      Samples(sum + value * times, minimum.min(value), maximum.max(value), count + times,
        layers.zip(extraLayers).map(_ + _))
    def mean: Double = sum / count

  final class Run(config: ujson.Value):
    private val labels = mutable.HashMap.empty[String, Prediction]
    private val events = mutable.LinkedHashMap.empty[(Int, Prediction, Int), Evaluations]
    var imageIndex: Int = 0
    def record(message: String): Unit =
      val parts = message.split(" ", 3)
      val encoded = parts(1)
      val prediction = labels.getOrElseUpdate(encoded, {
        val value = ujson.read(new String(Base64.getUrlDecoder.decode(encoded), UTF_8))
        Prediction(value("pass_name").str, value("branch").str, value("prompt").str,
          value("is_negative").bool, value("weights").obj.iterator.map((name, weight) => name -> weight.num).toVector,
          value("bindings").arr.iterator.map(_.str).toVector)
      })
      val layers = parts.lift(2).map { raw =>
        val counts = if raw == "-" then Vector.empty else raw.split(',').iterator.map(_.toLong).toVector
        require(counts.size <= prediction.bindings.size, "Unexpected LoRA layer count")
        counts.padTo(prediction.bindings.size, 0L)
      }
      val key = (imageIndex, prediction, parts(0).toInt)
      val current = Evaluations(1, layers)
      events.updateWith(key)(prior => Some(prior.fold(current)(_.add(current))))

    private def summarize(): Vector[((Series, Int), Samples)] =
      val stats = mutable.LinkedHashMap.empty[(Series, Int), Samples]
      for ((image, prediction, step), evaluated) <- events do
        val byAdapter = evaluated.layers.map(counts => prediction.bindings.zip(counts).toMap)
        val times = evaluated.models
        for (name, weight) <- prediction.weights do
          val series = Series(image, prediction.pass, prediction.branch, prediction.prompt, prediction.negative, name)
          val key = (series, step)
          val layers = byAdapter.map(_.getOrElse(name, 0L))
          stats.updateWith(key)(prior =>
            Some(prior.fold(Samples(weight * times, weight, weight, times, layers))(_.add(weight, times, layers))))
      stats.toVector

    private def writeTrace(trace: Path, stats: Vector[((Series, Int), Samples)]): Unit =
      Using.resource(Files.newBufferedWriter(trace)) { writer =>
        writer.write("{\"description\":\"Effective U-Net weights on evaluated Vulkan predictors. Steps restart for each pass.\",\"records\":[")
        for (((s, step), v), index) <- stats.iterator.zipWithIndex do
          if index > 0 then writer.write(",")
          writer.write("\n")
          val row = Obj("pass_name" -> s.pass, "image_index" -> s.image, "branch" -> s.branch, "step" -> step,
            "prompt" -> s.prompt, "is_negative" -> s.negative, "lora" -> s.lora, "mean_weight" -> v.mean,
            "min_weight" -> v.minimum, "max_weight" -> v.maximum, "model_evaluations" -> v.count)
          v.layers.foreach(layers => row("layer_evaluations") = ujson.Num(layers.toDouble))
          writer.write(row.render())
        writer.write("\n]}\n")
      }

    def finish(directory: Path, first: GeneratedImage, password: Option[String],
        register: Path => Unit): (GeneratedImage, ComposableLoraTrace) =
      val stats = summarize()
      val plot = drawPlot(stats)
      val image = Files.createTempFile(directory, "composable-lora-weights-", ".png")
      register(image)
      ScrambledPng.write(plot, image, Nil, password)
      val root = config.obj.get("output_dir").flatMap(_.strOpt).filter(_.nonEmpty).fold(directory)(Path.of(_))
      val traceDirectory = Files.createDirectories(root.resolve("intermediates").resolve("composable-lora"))
      val trace = Files.createTempFile(traceDirectory, "weights-", ".json")
      register(trace)
      writeTrace(trace, stats)
      (GeneratedImage(image.toString, first.seed, subseed = first.subseed, width = plot.getWidth,
        height = plot.getHeight), ComposableLoraTrace(trace.toString))

  private def drawPlot(stats: Vector[((Series, Int), Samples)]): BufferedImage =
    val image = new BufferedImage(1200, 680, BufferedImage.TYPE_INT_RGB)
    val g = image.createGraphics()
    try
      g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
      g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 15))
      g.setColor(new Color(0x17212b))
      g.fillRect(0, 0, image.getWidth, image.getHeight)
      g.setColor(Color.WHITE)
      g.drawString("Composable LoRA · Effective U-Net weights per sampling step", 24, 28)
      val labels = new Color(0xaebdcc)
      g.setColor(labels)
      g.drawString("Mean weight across model evaluations. Prompts and weight ranges are in the trace JSON.", 24, 54)
      val groups = stats.groupMap(_._1._1)(row => row._1._2 -> row._2.mean)
      val values = stats.map(_._2.mean)
      val low = values.minOption.fold(0.0)(_.min(0.0))
      val high = values.maxOption.fold(1.0)(_.max(1.0))
      val last = stats.map(_._1._2).maxOption.getOrElse(1)
      def x(step: Int): Int = 84 + ((step - 1) * 1070.0 / (last - 1).max(1)).round.toInt
      def y(value: Double): Int = 468 - ((value - low) / (high - low) * 380).round.toInt
      g.drawLine(84, 88, 84, 468)
      g.drawLine(84, y(0.0), 1154, y(0.0))
      for i <- 0 to 5 do
        val value = low + (high - low) * i / 5
        g.drawString(f"$value%.2f", 16, y(value) + 4)
      val ticks = (0 to (last - 1).min(7)).map(i => 1 + i * (last - 1) / (last - 1).min(7).max(1))
      ticks.foreach(step => g.drawString(step.toString, x(step) - 5, 493))
      g.drawString("Sampling step", 536, 517)
      val palette = Vector(0x80c7ff, 0xefad72, 0x9ed792, 0xdba8eb, 0xed819a, 0xf3dc83)
      g.setStroke(new BasicStroke(2))
      if groups.isEmpty then g.drawString("No U-Net adapter predictions were recorded.", 100, 144)
      for ((series, data), index) <- groups.toVector.sortBy((s, _) => (s.image, s.pass, s.branch, s.lora)).zipWithIndex do
        g.setColor(new Color(palette(index % palette.size)))
        val points = data.sortBy(_._1).map((step, value) => x(step) -> y(value))
        points.zip(points.drop(1)).foreach { case ((x0, y0), (x1, y1)) => g.drawLine(x0, y0, x1, y1) }
        points.foreach((px, py) => g.fillOval(px - 3, py - 3, 6, 6))
        if index < 8 then
          val polarity = if series.negative then "negative" else "positive"
          val prompt = series.prompt.replaceAll("\\s+", " ").take(70)
          val label = s"${series.image + 1} / ${series.pass} / ${series.lora} / $polarity / $prompt"
          val clipped = label.inits.find(s => g.getFontMetrics.stringWidth(s + "…") <= 550).getOrElse("")
          g.drawString(if clipped == label then label else clipped + "…", 24 + (index % 2) * 585, 549 + (index / 2) * 28)
      if groups.size > 8 then
        g.setColor(labels)
        g.drawString("See the trace JSON for all branch labels.", 24, 658)
    finally g.dispose()
    image
