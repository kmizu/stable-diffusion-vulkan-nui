// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan

import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer

private[vulkan] object VulkanWeightGradient:
  val id = "weight_gradient"
  val Modes: Set[String] = Set("a", "e", "ei", "eo", "c", "ci", "co")
  private val numPat = """[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"""
  private val weightsRe = Pattern.compile(s"""^\\s*($numPat)\\s*-\\s*($numPat)(?:\\s*-\\s*($numPat))?\\s*$$""")
  private val intervalRe = Pattern.compile(s"""^\\s*($numPat%?)\\s*-\\s*($numPat%?)\\s*$$""")

  def curve(mode: String, rawX: Double): Double =
    val x = math.max(0.0, math.min(1.0, rawX))
    if x == 0.0 || x == 1.0 then x
    else mode match
      case "a" => x
      case "ei" => math.pow(2.0, 10.0 * x - 10.0)
      case "eo" => 1.0 - math.pow(2.0, -10.0 * x)
      case "e" =>
        if x < 0.5 then math.pow(2.0, 20.0 * x - 10.0) / 2.0
        else 1.0 - math.pow(2.0, -20.0 * x + 10.0) / 2.0
      case "ci" => 1.0 - math.sqrt(math.max(0.0, 1.0 - x * x))
      case "co" => math.sqrt(math.max(0.0, 1.0 - (x - 1.0) * (x - 1.0)))
      case "c" =>
        if x < 0.5 then (1.0 - math.sqrt(math.max(0.0, 1.0 - (2.0 * x) * (2.0 * x)))) / 2.0
        else (1.0 + math.sqrt(math.max(0.0, 1.0 - (2.0 - 2.0 * x) * (2.0 - 2.0 * x)))) / 2.0
      case other => throw IllegalArgumentException(s"unknown weight gradient mode: $other")

  final case class Gradient(
      tokens: String,
      interval: Option[((Double, Boolean), (Double, Boolean))],
      weights: Vector[Double],
      mode: String
  ):
    def weight(step: Int, total: Int): Option[Double] =
      val (start, end) = interval match
        case None => (1, total)
        case Some(((v1, f1), (v2, f2))) =>
          val s = math.max(1, math.floor(if f1 then v1 * total else v1).toInt)
          val e = math.max(1, math.floor(if f2 then v2 * total else v2).toInt)
          (s, e)
      if end < start then None
      else
        val rawX = if start == end && step >= end then 1.0 else (step - start).toDouble / math.max(1, end - start).toDouble
        var x = math.max(0.0, math.min(1.0, rawX))
        var a = weights(0)
        var b = weights(1)
        if weights.size == 3 then
          if x <= 0.5 then x *= 2.0
          else
            a = b
            b = weights(2)
            x = (x - 0.5) * 2.0
        Some(a + (b - a) * curve(mode, x))

  private def splitFields(body: String): Option[Vector[String]] =
    val fields = ArrayBuffer.empty[String]
    val stack = ArrayBuffer.empty[Char]
    var start = 0
    var i = 0
    while i < body.length do
      val c = body.charAt(i)
      if c == '\\' then i += 2
      else
        if c == '(' || c == '[' then
          stack += c
          i += 1
        else if c == ')' || c == ']' then
          val expected = if c == ')' then '(' else '['
          if stack.isEmpty || stack.remove(stack.size - 1) != expected then return None
          i += 1
        else if c == '{' || c == '}' then return None
        else if c == ':' && stack.isEmpty then
          fields += body.substring(start, i).trim
          start = i + 1
          i += 1
        else i += 1
    if stack.nonEmpty then None
    else
      fields += body.substring(start).trim
      Some(fields.toVector)

  private def parsePoint(raw: String): Option[(Double, Boolean)] =
    if raw.endsWith("%") then
      raw.dropRight(1).toDoubleOption.filter(_.isFinite).map(v => (v / 100.0, true))
    else
      raw.toDoubleOption.filter(_.isFinite).map { v =>
        val fractional = v >= 0.0 && v <= 1.0 && raw.toLowerCase(java.util.Locale.ROOT).exists(c => c == '.' || c == 'e')
        (v, fractional)
      }

  private def parseGradient(body: String): Option[Gradient] =
    splitFields(body).flatMap { fields =>
      if (fields.size != 3 && fields.size != 4) || fields(0).isEmpty then None
      else
        val rawMode = if fields.size == 4 then fields(3) else "a"
        val mode = if rawMode.isEmpty then "a" else rawMode
        val wm = weightsRe.matcher(fields(2))
        if !Modes.contains(mode) || !wm.matches() then None
        else
          val w1 = wm.group(1).toDoubleOption
          val w2 = wm.group(2).toDoubleOption
          val w3 = Option(wm.group(3)).flatMap(_.toDoubleOption)
          if w1.isEmpty || w2.isEmpty || (wm.group(3) != null && w3.isEmpty) then None
          else
            val weights = Vector(w1.get, w2.get) ++ w3.toVector
            if !weights.forall(_.isFinite) then None
            else if fields(1).isEmpty then Some(Gradient(fields(0), None, weights, mode))
            else
              val im = intervalRe.matcher(fields(1))
              if !im.matches() then None
              else
                for
                  p1 <- parsePoint(im.group(1))
                  p2 <- parsePoint(im.group(2))
                  if !(p1._2 == p2._2 && p1._1 > p2._1)
                yield Gradient(fields(0), Some((p1, p2)), weights, mode)
    }

  private def findTags(text: String): Vector[(Int, Int, Gradient)] =
    val out = ArrayBuffer.empty[(Int, Int, Gradient)]
    var i = 0
    while i < text.length do
      if text.charAt(i) == '\\' then i += 2
      else if text.charAt(i) != '{' then i += 1
      else
        var end = i + 1
        var depth = 1
        while end < text.length && depth > 0 do
          if text.charAt(end) == '\\' then end += 2
          else
            if text.charAt(end) == '{' then depth += 1
            else if text.charAt(end) == '}' then depth -= 1
            end += 1
        if depth > 0 then i = text.length
        else
          parseGradient(text.substring(i + 1, end - 1)).foreach { g =>
            out += ((i, end, g))
          }
          i = end
    out.toVector

  def hasGradient(text: String): Boolean = findTags(text).nonEmpty

  private def formatWeight(w: Double): String =
    val formatted = String.format(java.util.Locale.ROOT, "%.4f", java.lang.Double.valueOf(w))
      .replaceAll("0+$", "")
      .replaceAll("\\.$", "")
    if formatted == "-0" || formatted.isEmpty then "0" else formatted

  def render(text: String, step: Int, total: Int): String =
    val tags = findTags(text)
    if tags.isEmpty then text
    else
      val sb = new java.lang.StringBuilder()
      var at = 0
      tags.foreach { case (start, end, gradient) =>
        sb.append(text.substring(at, start))
        gradient.weight(step, total) match
          case None => sb.append(text.substring(start, end))
          case Some(w) => sb.append("(").append(gradient.tokens).append(":").append(formatWeight(w)).append(")")
        at = end
      }
      sb.append(text.substring(at))
      sb.toString
