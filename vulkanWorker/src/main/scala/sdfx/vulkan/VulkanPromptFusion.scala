package sdfx.vulkan

import java.nio.file.{Files, Path}
import scala.collection.mutable
import sdfx.core.model.GenerationParams
import sdfx.vulkan.VulkanClipIo.{Matrix, readNpy, writeNpy}
import sdfx.vulkan.VulkanFusionPrompt.Plan
import ujson.Value

/** Encodes prompt leaves on the selected Vulkan adapter and schedules their evaluated conditioning. */
private[vulkan] object VulkanPromptFusion:
  private def invalid(detail: String): Nothing = VulkanProtocol.invalid(s"Prompt Fusion: $detail")
  private def file(path: Path): String = ujson.write(ujson.Str(path.toString))

  private final case class Options(scale: Double, epsilon: Double, negativeOrigin: Boolean)
  private final case class Condition(cross: Array[Float], pool: Array[Float]):
    def zip(other: Condition)(f: (Array[Float], Array[Float]) => Array[Float]): Condition =
      Condition(f(cross,other.cross),f(pool,other.pool))

  private def elementwise(a: Array[Float], b: Array[Float])(f: (Float,Float) => Float): Array[Float] =
    Array.tabulate(a.length)(index => f(a(index),b(index)))

  private def add(a: Condition,b: Condition): Condition = a.zip(b)((x,y) => elementwise(x,y)(_+_))
  private def subtract(a: Condition,b: Condition): Condition = a.zip(b)((x,y) => elementwise(x,y)(_-_))
  private def scale(a: Condition,s: Double): Condition =
    val factor = s.toFloat
    Condition(a.cross.map(_*factor),a.pool.map(_*factor))

  private def interpolate(left: Condition, right: Condition, t: Double, options: Options): Condition =
    if t == 0d then left
    else if t == 1d then right
    else left.zip(right) { (a,b) =>
      val ratio = t.toFloat
      def linear = elementwise(a,b)((x,y) => x+(y-x)*ratio)
      if options.scale == 0 || a.isEmpty then linear
      else
        def dot(a: Array[Float], b: Array[Float]): Double =
          var result = 0d
          var i = 0
          while i < a.length do
            result += a(i).toDouble*b(i)
            i += 1
          result
        def length(vector: Array[Float]): Double = math.sqrt(dot(vector,vector))
        val na = length(a)
        val nb = length(b)
        if na == 0 || nb == 0 then linear
        else
          val cosine = (dot(a,b)/(na*nb)).max(-1).min(1)
          if cosine <= options.epsilon-1 || cosine >= 1-options.epsilon || math.abs(cosine) == 1 then linear
          else
            val halfAngle = math.acos(cosine)/2
            val sphericalRatio = ((math.tan(halfAngle*(2*t-1))/math.tan(halfAngle)+1)/2).toFloat
            val normalize = (na/nb).toFloat
            val point = elementwise(a,b)((x,y) => x+(y*normalize-x)*sphericalRatio)
            val sphericalScale = ((na+(nb-na)*t)/length(point)).toFloat
            val result = linear
            val blend = options.scale.toFloat
            for i <- result.indices do result(i) += (point(i)*sphericalScale-result(i))*blend
            result
    }

  private def evaluate(plan: Plan, lookup: String => Condition, options: Options): Condition = plan match
    case Plan.Leaf(text) => lookup(text)
    case Plan.Blend(method,t,children,weights) =>
      val points = children.map(evaluate(_,lookup,options))
      def bezier(start: Vector[Condition], ratio: Double): Condition =
        var remaining = start
        while remaining.size > 1 do
          remaining = remaining.zip(remaining.tail).map((a,b) => interpolate(a,b,ratio,options))
        remaining.head
      method match
        case "mean" => points.zip(weights).map(scale(_, _)).reduce(add)
        case "bezier" => bezier(points,t)
        case _ =>
          val along = t*(points.size-1)
          val index = along.toInt.min(points.size-1)
          if index == points.size-1 then points.last
          else
            val ratio = along-index
            val a = points(index)
            val b = points(index+1)
            if method != "catmull" || points.size < 3 then interpolate(a,b,ratio,options)
            else
              val previous = points.lift(index-1).getOrElse(subtract(scale(a,2),b))
              val next = points.lift(index+2).getOrElse(subtract(scale(b,2),a))
              val innerA = add(a,scale(subtract(b,previous),1d/6d))
              val innerB = add(b,scale(subtract(a,next),1d/6d))
              bezier(Vector(a,innerA,innerB,b),ratio)

  final class Schedule private[VulkanPromptFusion](positive: Vector[String], negative: Vector[String]):
    def predictors(config: String, lines: mutable.Builder[String,Vector[String]]): Unit =
      def build(name: String, conditions: Vector[String]): Unit =
        val unique = conditions.distinct.zipWithIndex.toMap
        unique.toVector.sortBy(_._2).foreach { (cond,i) =>
          lines += s"${name}_leaf$i = predictor(unet, $cond$config)"
        }
        def range(start: Int, stop: Int): String =
          if conditions(start) == conditions(stop-1) && (start until stop).forall(i => conditions(i) == conditions(start)) then
            s"${name}_leaf${unique(conditions(start))}"
          else
            val middle = (start+stop)/2
            val first = range(start,middle)
            val last = range(middle,stop)
            val node = s"${name}_${start}_$stop"
            lines += s"$node = switch_predictor($first, $last, after_steps = $middle)"
            node
        val predictor = range(0,conditions.size)
        lines += s"$name = $predictor"
      build("pos",positive)
      build("neg",negative)

  def prepare(engine: VulkanInference, scratch: Path, config: Value,
      params: GenerationParams, prompt: String, sdxl: Boolean, width: Int,
      textSetup: Vector[String], encoder: String, encoding: String, check: () => Unit,
      lines: mutable.Builder[String,Vector[String]]): Schedule =
    def option(name: String, default: Double) = config.obj.get(name).fold(default)(_.num)
    val options = Options(option("slerp_scale",0),option("slerp_epsilon",.0001),
      config.obj.get("negative_origin").forall(_.bool))
    if !options.scale.isFinite || !options.epsilon.isFinite || options.epsilon < 0 || options.epsilon > 1 then
      invalid("invalid spherical geometry")
    val steps = params.img2img.fold(params.steps)(i => math.min(params.steps,(i.denoisingStrength*params.steps).toInt+1))
    val positive = VulkanFusionPrompt.schedule(prompt,steps)
    val negative = VulkanFusionPrompt.schedule(params.negativePrompt,steps)
    val prompts = mutable.LinkedHashSet("")
    (positive ++ negative).foreach(_.visit(text => prompts += text))
    if prompts.size > 513 then invalid("the two prompts exceed 512 conditioning leaves")
    val exports = prompts.toVector.zipWithIndex.map { (text,index) =>
      val prepass = Vector.newBuilder[String]
      val output = scratch.resolve(s"fusion-leaf-$index.npy")
      val pool = scratch.resolve(s"fusion-pool-$index.npy")
      val variable = s"fusion_leaf$index"
      prepass += s"$variable = encode_prompt($encoder, ${ujson.write(ujson.Str(text))}, $encoding)"
      prepass += s"tensor::save(${file(output)}, $variable.crossattn)"
      if sdxl then
        prepass += s"${variable}_vector = tensor::reshape($variable.vector,[2816,1])"
        prepass += s"tensor::save(${file(pool)},${variable}_vector)"
      (text,output,pool,prepass.result())
    }
    var materialized = 0L
    for batch <- exports.grouped(16) do
      val prepass = textSetup ++ batch.flatMap(_._4) :+ "return {value = tensor::zeros([1])}"
      check()
      engine.writeDglTensors(s"fn main() {\n${prepass.mkString("\n")}\n}")
      materialized += batch.iterator.map(item => Files.size(item._2)).sum
      if materialized > 384L*1024L*1024L then invalid("conditioning leaves exceed the memory limit")
    val paths = exports.map((text,cross,pool,_) => text -> (cross,pool)).toMap
    val records = paths.map { (key, paths) =>
      check()
      val (cross, pool) = paths
      val matrix = readNpy(cross,width)
      val vector = if sdxl then readNpy(pool,2816).values else Array.emptyFloatArray
      key -> (matrix,vector)
    }
    val (empty,emptyVector) = records("")
    val tokens = records.valuesIterator.map(_._1.tokens).max
    if tokens.toLong*width*records.size*4L > 384L*1024L*1024L then
      invalid("padded conditioning exceeds the memory limit")
    val encoded = records.map { (key, value) =>
      val (matrix,vec) = value
      val padded = Array.tabulate(tokens*width) { index =>
        if index < matrix.values.length then matrix.values(index)
        else empty.values((index-matrix.values.length)%empty.values.length)
      }
      key -> Condition(padded,vec.take(1280))
    }
    val blank = encoded("")
    val zero = Condition(new Array[Float](tokens*width),new Array[Float](if sdxl then 1280 else 0))
    val metadata = emptyVector.drop(1280)
    val cached = mutable.Map.empty[(Plan,Boolean,Option[Plan]),String]
    def evaluateFrom(plan: Plan, origin: Condition, neg: Boolean): Condition =
      def lookup(text: String): Condition = if neg && sdxl && text.isEmpty then zero else encoded(text)
      if options.scale == 0 then evaluate(plan,lookup,options)
      else add(evaluate(plan,text => subtract(lookup(text),origin),options),origin)
    def save(key: (Plan,Boolean,Option[Plan]), calculated: => Condition): String =
      cached.getOrElseUpdate(key, {
        if (cached.size.toLong+1)*tokens*width*4L > 768L*1024L*1024L then
          invalid("scheduled conditioning exceeds the memory limit")
        val condition = calculated
        if condition.cross.exists(!_.isFinite) || condition.pool.exists(!_.isFinite) then
          invalid("interpolation produced non-finite conditioning")
        val name = s"fusion_condition${cached.size}"
        val cross = scratch.resolve(s"$name.npy")
        writeNpy(cross,Matrix(tokens,width,condition.cross))
        val pool = if !sdxl then "" else
          val target = scratch.resolve(s"${name}_vector.npy")
          writeNpy(target,Matrix(1,2816,condition.pool ++ metadata))
          s", vector = tensor::reshape(tensor::load(${file(target)}),[2816])"
        lines += s"$name = {crossattn = tensor::load(${file(cross)})$pool}"
        name
      })
    val frames = positive.zip(negative).map { (pos,neg) =>
      check()
      lazy val negativePoint = evaluateFrom(neg,blank,neg = true)
      val negativeCond = save((neg,true,None),negativePoint)
      val usesOrigin = options.scale != 0 && options.negativeOrigin
      val positiveCond = save((pos,false,Option.when(usesOrigin)(neg)),
        evaluateFrom(pos,if usesOrigin then negativePoint else blank,neg = false))
      positiveCond -> negativeCond
    }
    new Schedule(frames.map(_._1),frames.map(_._2))
