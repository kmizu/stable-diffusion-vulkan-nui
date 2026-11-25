// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import scala.collection.mutable
import java.util.Locale

private[tools] object EmbeddingMerge:
  val MaxExpression = 65536
  val MaxRows = 4096
  val MaxActions = 4096
  private val NumberRegex = "^([+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+))".r

  final case class Action(
      kind: String,
      textVal: String = "",
      numVal: Double = 0.0,
      intVal: Int = 0,
      whole: Boolean = false,
      stream: Option[String] = None
  )

  final case class Span(start: Int, end: Int, original: String, expression: String)

  def hasExpressions(prompt: String): Boolean =
    prompt != null && (prompt.contains("<'") || prompt.contains("{'"))

  def extractSpans(prompt: String): Vector[Span] =
    if !hasExpressions(prompt) then return Vector.empty
    require(prompt.length <= MaxExpression, "Embedding Merge: prompt input limit exceeded")
    val out = mutable.ArrayBuffer.empty[Span]
    var pos = 0
    while pos < prompt.length - 1 do
      val idx = findNextOpen(prompt, pos)
      if idx < 0 then pos = prompt.length
      else
        val span = readSpan(prompt, idx)
        out += span
        pos = span.end
    out.toVector

  private def findNextOpen(s: String, from: Int): Int =
    var i = from
    while i < s.length - 1 do
      val c = s.charAt(i)
      if (c == '<' || c == '{') && s.charAt(i + 1) == '\'' then return i
      i += 1
    -1

  private def readSpan(text: String, start: Int): Span =
    val close = if text.charAt(start) == '<' then '>' else '}'
    var quote = false
    var pos = start + 1
    while pos < text.length do
      val ch = text.charAt(pos)
      if ch == '\'' then
        if quote && pos + 1 < text.length && text.charAt(pos + 1) == '\'' then pos += 2
        else { quote = !quote; pos += 1 }
      else if ch == close && !quote then
        val orig = text.substring(start, pos + 1)
        val expr = text.substring(start + 1, pos)
        parseExpression(expr)
        return Span(start, pos + 1, orig, expr)
      else pos += 1
    throw IllegalArgumentException(s"Embedding Merge: unclosed merge expression at ${start}")

  def parseExpression(raw: String): Vector[Action] =
    require(raw != null && raw.length <= MaxExpression, "Embedding Merge: expression input limit exceeded")
    val text = raw.replace('\u0000', ' ')
    val actions = mutable.ArrayBuffer.empty[Action]
    var pos = 0
    def skipSpaces(): Unit =
      while pos < text.length && text.charAt(pos).isWhitespace do pos += 1
    def readQuoted(): String =
      skipSpaces()
      if pos >= text.length || text.charAt(pos) != '\'' then
        throw IllegalArgumentException(s"Embedding Merge: expected quoted text at ${pos}")
      pos += 1
      val sb = new StringBuilder
      while pos < text.length do
        val ch = text.charAt(pos)
        pos += 1
        if ch == '\'' then
          if pos < text.length && text.charAt(pos) == '\'' then { sb.append('\''); pos += 1 }
          else return sb.toString
        else sb.append(ch)
      throw IllegalArgumentException("Embedding Merge: last quote not closed")

    actions += Action("text", textVal = readQuoted())
    var done = false
    while !done do
      skipSpaces()
      if pos >= text.length then done = true
      else
        if actions.size >= MaxActions then throw IllegalArgumentException("Embedding Merge: expression action limit exceeded")
        val whole = text.charAt(pos) == '='
        if whole then { pos += 1; skipSpaces() }
        if pos >= text.length then throw IllegalArgumentException("Embedding Merge: unfinished result operator")
        val op = text.charAt(pos)
        pos += 1
        skipSpaces()
        if (op == '+' || op == '-') && !whole then
          actions += Action(if op == '+' then "add" else "sub")
          actions += Action("text", textVal = readQuoted())
        else if op == '*' || op == '/' then
          val sub = text.substring(pos)
          val m = NumberRegex.findFirstMatchIn(sub).getOrElse(
            throw IllegalArgumentException("Embedding Merge: expected numeric multiplier"))
          pos += m.end
          val v = m.group(1).toDouble
          skipSpaces()
          val stream = if pos < text.length && (text.charAt(pos) == 'L' || text.charAt(pos) == 'G') then
            val s = if text.charAt(pos) == 'L' then "clip_l" else "clip_g"
            pos += 1
            Some(s)
          else None
          actions += Action(if op == '*' then "mul" else "div", numVal = v, whole = whole, stream = stream)
        else if op == ':' then
          var shift = 0
          if pos < text.length && (text.charAt(pos) == '+' || text.charAt(pos) == '-') then
            shift = if text.charAt(pos) == '+' then 1 else -1
            pos += 1
            skipSpaces()
          val intPat = if shift != 0 then "^(-?[0-9]+)".r else "^([0-9]+)".r
          val m = intPat.findFirstMatchIn(text.substring(pos)).getOrElse(
            throw IllegalArgumentException("Embedding Merge: expected integer vector count or rotation"))
          pos += m.end
          val v = m.group(1).toInt
          if shift != 0 then actions += Action("roll", intVal = v * shift, whole = whole)
          else actions += Action("resize", intVal = v, whole = whole)
        else if op == ',' && !whole then
          if pos < text.length && text.charAt(pos) == '\'' then
            actions += Action("append", textVal = readQuoted())
          else
            val m = "^([0-9]+)".r.findFirstMatchIn(text.substring(pos)).getOrElse(
              throw IllegalArgumentException("Embedding Merge: expected quoted text or token ID after comma"))
            val rawTok = m.group(1)
            pos += rawTok.length
            if rawTok.startsWith("0") then
              rawTok match
                case "0" => actions += Action("token_id", intVal = 0)
                case "00" => actions += Action("token_special", textVal = "pad")
                case "000" => actions += Action("token_special", textVal = "end")
                case "0000" => actions += Action("token_special", textVal = "start")
                case _ => throw IllegalArgumentException("Embedding Merge: invalid zero-prefixed token ID")
            else
              actions += Action("token_id", intVal = rawTok.toInt)
        else
          throw IllegalArgumentException(s"Embedding Merge: invalid operator '${op}' at ${pos - 1}")
    actions.toVector

  /** StreamVectors: one `Vector[Array[Float]]` (rows x width) per stream (`clip_l`, optionally `clip_g`). */
  type StreamVectors = Vector[Vector[Array[Float]]]

  def evaluate(
      actions: Vector[Action],
      streamNames: Vector[String],
      streamWidths: Vector[Int],
      resolveText: String => StreamVectors,
      resolveToken: Either[Int, String] => StreamVectors
  ): StreamVectors =
    def checked(sv: StreamVectors): StreamVectors =
      require(sv.size == streamNames.size, "Embedding Merge: mismatched CLIP streams")
      for s <- sv.indices do
        val rows = sv(s)
        val w = streamWidths(s)
        require(rows.size <= MaxRows, "Embedding Merge: vector row limit exceeded")
        require(rows.forall(r => r.length == w && r.forall(_.isFinite)), "Embedding Merge: nonfinite or mismatched vector")
      require(sv.map(_.size).distinct.size <= 1, "Embedding Merge: CLIP stream rows must match")
      sv

    def resizeRows(rows: Vector[Array[Float]], count: Int, width: Int): Vector[Array[Float]] =
      require(count >= 0 && count <= MaxRows, "Embedding Merge: vector row limit exceeded")
      Vector.tabulate(count) { r =>
        if r < rows.size then rows(r).clone()
        else new Array[Float](width)
      }

    def rollRows(rows: Vector[Array[Float]], shift: Int): Vector[Array[Float]] =
      if rows.isEmpty then rows
      else
        val n = rows.size
        val k = ((shift % n) + n) % n
        if k == 0 then rows.map(_.clone())
        else (rows.drop(n - k) ++ rows.take(n - k)).map(_.clone())

    var left: Option[StreamVectors] = None
    var right: Option[StreamVectors] = None
    var sign = 1f

    def combine(): Unit =
      if left.isDefined then
        val l = left.get
        val r = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing right operand"))
        val count = math.max(l.head.size, r.head.size)
        val merged = Vector.tabulate(streamNames.size) { s =>
          val w = streamWidths(s)
          val la = resizeRows(l(s), count, w)
          val ra = resizeRows(r(s), count, w)
          Vector.tabulate(count) { rowIdx =>
            val rowL = la(rowIdx); val rowR = ra(rowIdx)
            Array.tabulate(w)(c => rowL(c) + sign * rowR(c))
          }
        }
        right = Some(checked(merged))
        left = None

    for action <- actions do
      if action.whole || action.kind == "add" || action.kind == "sub" then combine()
      action.kind match
        case "text" =>
          right = Some(checked(resolveText(action.textVal)))
        case "add" | "sub" =>
          left = right
          right = None
          sign = if action.kind == "add" then 1f else -1f
        case "append" | "token_id" | "token_special" =>
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand before append"))
          val next = checked(
            if action.kind == "append" then resolveText(action.textVal)
            else if action.kind == "token_id" then resolveToken(Left(action.intVal))
            else resolveToken(Right(action.textVal))
          )
          require(cur.head.size + next.head.size <= MaxRows, "Embedding Merge: vector row limit exceeded")
          right = Some(checked(Vector.tabulate(streamNames.size)(s => cur(s) ++ next(s))))
        case "resize" =>
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand for resize"))
          right = Some(checked(Vector.tabulate(streamNames.size)(s => resizeRows(cur(s), action.intVal, streamWidths(s)))))
        case "roll" =>
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand for roll"))
          right = Some(checked(Vector.tabulate(streamNames.size)(s => rollRows(cur(s), action.intVal))))
        case "mul" | "div" =>
          if action.stream.isDefined && streamNames.size != 2 then
            throw IllegalArgumentException("Embedding Merge: L/G suffix requires SDXL streams")
          if action.kind == "div" && action.numVal == 0.0 then
            throw IllegalArgumentException("Embedding Merge: division by zero")
          require(action.numVal.isFinite, "Embedding Merge: nonfinite scalar")
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand for scalar op"))
          val factor = (if action.kind == "mul" then action.numVal else 1.0 / action.numVal).toFloat
          right = Some(checked(Vector.tabulate(streamNames.size) { s =>
            if action.stream.forall(_ == streamNames(s)) then
              cur(s).map(row => row.map(v => v * factor))
            else cur(s).map(_.clone())
          }))
        case other =>
          throw IllegalArgumentException(s"Embedding Merge: unknown action '${other}'")
    combine()
    checked(right.getOrElse(throw IllegalArgumentException("Embedding Merge: empty expression")))
