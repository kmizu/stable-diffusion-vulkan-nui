package sdfx.vulkan

import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer

private[vulkan] final case class PromptSpan(text: String,weight: Double)


/** NUI attention syntax and fixed-length CLIP chunks; no tokenizer or Python dependency. */
private[vulkan] object VulkanPromptAttention:
  private val attention=Pattern.compile("""\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*(x?[+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:""",Pattern.UNICODE_CHARACTER_CLASS)
  private val breaks=Pattern.compile("""\s*\bBREAK\b\s*""",Pattern.UNICODE_CHARACTER_CLASS)
  private val maxText=1048576
  private val maxSpans=65536

  def parse(text: String): Vector[PromptSpan] =
    require(text.length<=maxText,"invalid or oversized prompt")
    val spans=ArrayBuffer.empty[PromptSpan]
    val rounds=ArrayBuffer.empty[Int]
    val squares=ArrayBuffer.empty[Int]
    var characters=0L
    def append(s: String,w: Double): Unit =
      characters+=s.length
      require(spans.size<maxSpans && characters<=maxText,"prompt expansion exceeds limit")
      spans+=PromptSpan(s,w)
    def multiply(start: Int,factor: Double): Unit =
      require(factor.isFinite,"nonfinite prompt weight")
      for i <- start until spans.size do
        val updated=spans(i).weight*factor
        require(updated.isFinite,"prompt weight overflow")
        spans(i)=spans(i).copy(weight=updated)
    def pop(stack: ArrayBuffer[Int]): Int=stack.remove(stack.size-1)
    val matcher=attention.matcher(text)
    while matcher.find() do
      val token=matcher.group()
      val explicit=matcher.group(1)
      if token.startsWith("\\") then append(token.substring(1),1)
      else if token=="(" || token=="[" then
        require(rounds.size+squares.size<4096,"prompt nesting exceeds limit")
        (if token=="(" then rounds else squares)+=spans.size
      else if explicit!=null && rounds.nonEmpty then
        if explicit.startsWith("x") then
          explicit.substring(1).toDoubleOption match
            case None => append(token,1)
            case Some(value) =>
              require(value.isFinite && math.abs(value)<=maxSpans,"prompt repeat exceeds limit")
              val start=pop(rounds)
              val segment=spans.drop(start).toVector
              val count=math.max(1,math.abs(value).toInt)
              val tail=math.abs(value)%1+(if math.abs(value)>=1 then 1 else 0)
              require(start.toLong+segment.size.toLong*count<=maxSpans,"prompt repeat exceeds limit")
              val length=segment.map(_.text.length.toLong).sum
              require(characters+length*(count-1)<=maxText,"prompt expansion exceeds limit")
              spans.dropRightInPlace(spans.size-start)
              characters-=length
              for repetition <- 0 until count; span <- segment do
                val factor=(if repetition==count-1 && tail>1e-4 then tail else 1)*(if value<0 then -1 else 1)
                append(span.text,span.weight)
                multiply(spans.size-1,factor)
        else
          val start=pop(rounds)
          explicit.toDoubleOption match
            case Some(weight) => multiply(start,weight)
            case None => append(token,1)
      else if token==")" && rounds.nonEmpty then multiply(pop(rounds),1.1)
      else if token=="]" && squares.nonEmpty then multiply(pop(squares),1/1.1)
      else
        breaks.split(token,-1).zipWithIndex.foreach { (part,index) =>
          if index>0 then append("BREAK",-1)
          append(part,1)
        }
    rounds.foreach(multiply(_,1.1))
    squares.foreach(multiply(_,1/1.1))
    if spans.isEmpty then append("",1)
    // Assemble equal-weight runs once, avoiding quadratic concatenation for long prompts.
    val merged=Vector.newBuilder[PromptSpan]
    var start=0
    while start<spans.size do
      val builder=new java.lang.StringBuilder(spans(start).text)
      var end=start+1
      while end<spans.size && spans(end).weight==spans(start).weight do
        builder.append(spans(end).text)
        end+=1
      merged+=PromptSpan(builder.toString,spans(start).weight)
      start=end
    merged.result()
