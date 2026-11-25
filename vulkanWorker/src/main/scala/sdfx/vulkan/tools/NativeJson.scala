// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.{IOException,OutputStream,StringWriter}
import java.nio.charset.StandardCharsets.UTF_8
import scala.util.Using

enum NativeJson:
  case Obj(fields: Map[String,NativeJson])
  case Arr(values: Vector[NativeJson])
  case Str(value: String)
  case Number(value: BigDecimal)
  case Bool(value: Boolean)
  case Null

object NativeJson:
  def validUnicode(s: String): Boolean =
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if Character.isHighSurrogate(c) then
        if i + 1 == s.length || !Character.isLowSurrogate(s.charAt(i + 1)) then return false
        i += 2
      else if Character.isLowSurrogate(c) then return false
      else i += 1
    true

  val maxChars=16*1024*1024
  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(32).maxStringLength(maxChars).build()).build()
  def parse(text: String): NativeJson =
    require(text.length<=maxChars && validUnicode(text),"invalid or oversized worker JSON")
    try Using.resource(factory.createParser(text)) { p =>
      def value(): NativeJson=p.currentToken() match
        case JsonToken.START_OBJECT =>
          val fields=Map.newBuilder[String,NativeJson]
          while p.nextToken()!=JsonToken.END_OBJECT do
            require(p.currentToken()==JsonToken.FIELD_NAME,"invalid JSON field")
            val name=p.currentName()
            require(validUnicode(name),"invalid Unicode field")
            p.nextToken();fields+=name->value()
          Obj(fields.result())
        case JsonToken.START_ARRAY =>
          val values=Vector.newBuilder[NativeJson]
          while p.nextToken()!=JsonToken.END_ARRAY do values+=value()
          Arr(values.result())
        case JsonToken.VALUE_STRING =>
          val s=p.getText()
          require(validUnicode(s),"invalid Unicode string")
          Str(s)
        case JsonToken.VALUE_NUMBER_INT | JsonToken.VALUE_NUMBER_FLOAT => Number(BigDecimal(p.getDecimalValue()))
        case JsonToken.VALUE_TRUE => Bool(true)
        case JsonToken.VALUE_FALSE => Bool(false)
        case JsonToken.VALUE_NULL => Null
        case _ => throw IllegalArgumentException("invalid worker JSON value")
      p.nextToken()
      val result=value()
      require(p.nextToken()==null,"trailing worker JSON")
      result
    }
    catch case e: IOException => throw IllegalArgumentException("invalid worker JSON",e)
  def render(value: NativeJson): String =
    val out=new StringWriter()
    Using.resource(factory.createGenerator(out)) { g =>
      def write(v: NativeJson): Unit=v match
        case Obj(fields) =>
          g.writeStartObject()
          fields.toVector.sortBy(_._1).foreach { (key,entry) => g.writeFieldName(key);write(entry) }
          g.writeEndObject()
        case Arr(values) => g.writeStartArray();values.foreach(write);g.writeEndArray()
        case Str(s) => g.writeString(s)
        case Number(n) => g.writeNumber(n.bigDecimal)
        case Bool(b) => g.writeBoolean(b)
        case Null => g.writeNull()
      write(value)
    }
    out.toString
