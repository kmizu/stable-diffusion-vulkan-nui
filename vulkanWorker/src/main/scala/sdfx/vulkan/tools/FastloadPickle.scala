package sdfx.vulkan.tools

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream, DataOutputStream}
import java.nio.charset.{StandardCharsets as CS}
import java.util.Base64
import scala.collection.mutable
import ujson.{Arr, Obj, Value}

/** Data-only interpreter. Globals stay symbolic and are matched only by the constructors below. */
private[tools] object FastloadPickle:
  val Limit = 128 * 1024 * 1024
  val base = Base64.getEncoder
  def tag(t: String): Obj = Obj("type" -> t)
  val nil = tag("null")
  def str(s: String): Value = Obj("type" -> "string","value" -> s)
  def integer(n: BigInt): Value = Obj("type" -> "int","value" -> n.toString)
  def number(n: Double): Value = Obj("type" -> "float","value" -> n)
  def bool(b: Boolean): Value = Obj("type" -> "bool","value" -> b)
  def seq(t: String, xs: Seq[Value]): Value = Obj("type" -> t,"items" -> Arr.from(xs))
  def dict(fields: Iterable[(String,Value)]): Value = seq("dict",fields.iterator.map((k,v) => Arr(k,v)).toVector)
  def fields(x: Value): Map[String,Value] =
    require(kind(x) == "dict","Expected Fastload dictionary")
    val pairs = items(x).map { pair => require(pair.arr.length == 2); pair(0).str -> pair(1) }
    require(pairs.map(_._1).distinct.size == pairs.size,"Duplicate Fastload key")
    pairs.toMap
  def items(x: Value): Vector[Value] = x("items").arr.toVector
  def kind(x: Value): String = x("type").str
  def text(x: Value): String = {require(kind(x) == "string"); x("value").str}
  def int(x: Value): BigInt = {require(kind(x) == "int"); BigInt(x("value").str)}
  def binary(s: String): Array[Byte] =
    require(s.length.toLong <= (Limit+2L)/3*4,"Fastload bytes limit")
    Base64.getDecoder.decode(s)
  private def dtype(s: String): String =
    require(s.matches("[<>=|]?(?:[iu][1248]|f[248]|b1)"),"Unsupported NumPy dtype")
    val code = if "<>=|".contains(s.head) then s else (if s.last=='1' then "|" else "=")+s
    require(code.head != '|' || code.last == '1',"Invalid NumPy endianness")
    code
  def shape(x: Value): Vector[Int] =
    val xs = x("shape").arr.map { v => val n=v.num; require(n.isWhole && n>=1 && n<=16384); n.toInt }.toVector
    require(xs.nonEmpty && xs.size<=3 && Set("C","F")(x("order").str),"Invalid NumPy array")
    xs
  def expected(x: Value): Int =
    val n = shape(x).foldLeft(dtype(x("dtype").str).last.asDigit.toLong)(_ * _)
    require(n<=Limit,"NumPy array exceeds limit"); n.toInt

  def write(root: Value): Array[Byte] =
    require(kind(root) == "list" && items(root).forall(v => kind(v) == "dict"),"Expected Fastload units")
    val data = new ByteArrayOutputStream
    val out = new DataOutputStream(data)
    var count = 0
    def op(c: Char): Unit = out.write(c.toInt)
    def le(i: Int): Unit = out.writeInt(Integer.reverseBytes(i))
    def string(s: String): Unit =
      val bs=s.getBytes(CS.UTF_8); require(bs.length.toLong+data.size+5<=Limit); op('X'); le(bs.length);out.write(bs)
    def bytes(bs: Array[Byte]): Unit =
      require(bs.length.toLong+data.size+5<=Limit); op('B'); le(bs.length); out.write(bs)
    def whole(n: BigInt): Unit =
      require(n.abs.bitLength<=64,"Fastload integer exceeds limit")
      if n.isValidInt then {op('J');le(n.toInt)}
      else {val b=n.toByteArray.reverse;out.write(0x8a);out.write(b.length);out.write(b)}
    def global(m: String,n: String): Unit = {op('c');out.writeBytes(m+"\n"+n+"\n")}
    def visit(x: Value, depth: Int): Unit =
      count+=1
      require(count<=100000 && depth<=64 && data.size<Limit,"Fastload structure exceeds limit")
      val typ=kind(x)
      val keys=typ match
        case "null" => Set("type")
        case "dict" | "tuple" | "list" => Set("type","items")
        case "array" => Set("type","dtype","shape","order","data")
        case "bytes" => Set("type","base64")
        case _ => Set("type","value")
      require(x.obj.keySet == keys,"Invalid Fastload wire fields")
      typ match
        case "null" => op('N')
        case "string" => string(text(x))
        case "bool" => out.write(if x("value").bool then 0x88 else 0x89)
        case "int" => val n=int(x); require(n.toString==x("value").str); whole(n)
        case "float" => val n=x("value").num; require(n.isFinite); op('G');out.writeDouble(n)
        case "bytes" => bytes(binary(x("base64").str))
        case "list" => op(']'); op('('); items(x).foreach(visit(_,depth+1));op('e')
        case "tuple" => op('('); items(x).foreach(visit(_,depth+1));op('t')
        case "dict" =>
          fields(x)
          op('}');op('(')
          items(x).foreach(p => {string(p(0).str); visit(p(1),depth+1)})
          op('u')
        case "array" =>
          val bs=binary(x("data").str)
          require(bs.length==expected(x),"NumPy buffer length mismatch")
          global("numpy.core.multiarray","_reconstruct");global("numpy","ndarray");whole(0);out.write(0x85)
          bytes(Array('b'.toByte));out.write(0x87);op('R');op('(');whole(1)
          op('(');shape(x).foreach(n => whole(n));op('t')
          global("numpy","dtype");string(dtype(x("dtype").str))
          out.write(0x89);out.write(0x88);out.write(0x87);op('R')
          out.write(if x("order").str=="F" then 0x88 else 0x89);bytes(bs);op('t');op('b')
        case _ => throw IllegalArgumentException("Unsupported Fastload value")
    out.write(0x80);out.write(3);visit(root,0);op('.');require(data.size<=Limit);data.toByteArray

  def read(data: Array[Byte]): Value =
    require(data.length<=Limit)
    val input = new ByteArrayInputStream(data); val in = new DataInputStream(input)
    def take(n: Long): Array[Byte] =
      require(n>=0 && n<=input.available(),"Invalid pickle length"); in.readNBytes(n.toInt)
    def le(): Int = Integer.reverseBytes(in.readInt())
    def line(): String =
      val b=new ByteArrayOutputStream
      var c=in.readUnsignedByte()
      while c!=10 do
        require(b.size<4096,"Pickle line exceeds limit");b.write(c);c=in.readUnsignedByte()
      b.toString(CS.UTF_8)
    def unicodeLine(): String =
      val result=new java.lang.StringBuilder
      var slashCount=0
      var c=in.readUnsignedByte()
      while c!=10 do
        if c==92 then slashCount+=1 else
          val escaped=(slashCount&1)==1 && (c==117 || c==85)
          result.append("\\".repeat(slashCount-(if escaped then 1 else 0)))
          slashCount=0
          if escaped then
            val digits=if c==117 then 4 else 8
            val hex=new String(take(digits),CS.US_ASCII)
            require(hex.forall(ch=>Character.digit(ch,16)>=0),"Invalid pickle Unicode escape")
            val code=java.lang.Long.parseLong(hex,16)
            require(code<=Character.MAX_CODE_POINT,"Invalid pickle codepoint")
            result.appendCodePoint(code.toInt)
          else result.append(c.toChar)
        c=in.readUnsignedByte()
      result.append("\\".repeat(slashCount)).toString
    val mark=tag("~mark")
    val stack=mutable.ArrayBuffer.empty[Value]; val memo=mutable.Map.empty[Int,Value]
    def pop(): Value = {require(stack.nonEmpty,"Incomplete pickle stack");stack.remove(stack.size-1)}
    def marked(): Vector[Value] =
      val n=stack.lastIndexWhere(_ eq mark);require(n>=0,"Missing pickle mark")
      val vs=stack.slice(n+1,stack.size).toVector;stack.dropRightInPlace(stack.size-n);vs
    def put(i:Int): Unit = {require(i>=0 && i<100000);memo(i)=stack.last}
    def global(m:String,n:String): Value =
      val allowed=m match
        case "numpy" => Set("ndarray","dtype")
        case "numpy.core.multiarray" | "numpy._core.multiarray" => Set("_reconstruct")
        case "numpy.core.numeric" | "numpy._core.numeric" => Set("_frombuffer")
        case "_codecs" => Set("encode")
        case "builtins" | "__builtin__" => Set("set")
        case "internal_controlnet.external_code" => Set("ControlNetUnit","ResizeMode","ControlMode")
        case "internal_controlnet.args" => Set("ControlNetUnit")
        case "scripts.enums" => Set("InputMode","ResizeMode","ControlMode","HiResFixOption","PuLIDMode","ControlNetUnionControlType")
        case _ => Set.empty[String]
      require(allowed(n),"Unsupported pickle global");Obj("type" -> "~global","module" -> m,"name" -> n)
    def byteVal(a:Array[Byte]): Value = Obj("type" -> "bytes","base64" -> base.encodeToString(a))
    def array(d:Value,s:Value,b:Value,order:String):Value =
      require(kind(d)=="~dtype" && kind(s)=="tuple" && kind(b)=="bytes")
      val v=Obj("type"->"array","dtype"->d("value"),"shape"->Arr.from(items(s).map(i=>ujson.Num(int(i).toDouble))),"data"->b("base64"),"order"->order)
      require(binary(v("data").str).length==expected(v),"Invalid array buffer");v
    def update(d:Value,values:Vector[Value]):Unit =
      require(kind(d)=="dict" && values.size%2==0)
      val kv=d("items").arr
      values.grouped(2).foreach(p=>kv+=Arr(text(p.head),p.last))
      require(kv.size<=100000)
    def collectionAppend(t:String,values:Vector[Value]):Unit =
      val top=stack.last;require(kind(top)==t);top("items").arr++=values;require(top("items").arr.size<=100000)
    var ops=0
    while input.available()>0 do
      ops+=1
      require(ops<=1000000 && stack.size<=100000 && memo.size<=100000,"Pickle structure exceeds limit")
      in.readUnsignedByte() match
        case 0x80 => require(in.readUnsignedByte()<=5)
        case 0x95 => val n=java.lang.Long.reverseBytes(in.readLong());require(n>=0 && n<=input.available())
        case 40 => stack+=mark
        case 78 => stack+=nil
        case 0x88 => stack+=bool(true)
        case 0x89 => stack+=bool(false)
        case 74 => stack+=integer(le())
        case 75 => stack+=integer(in.readUnsignedByte())
        case 77 => stack+=integer(java.lang.Short.toUnsignedInt(java.lang.Short.reverseBytes(in.readShort())))
        case 0x8a | 0x8b => val op=data(data.length-input.available()-1)&255;val n=if op==0x8a then in.readUnsignedByte() else le();require(n>=0 && n<=9);stack+=integer(if n==0 then 0 else BigInt(take(n).reverse))
        case 73 | 76 => val s=line();stack+=(s match {case "00"=>bool(false);case "01"=>bool(true);case _=>integer(BigInt(s.stripSuffix("L")))})
        case 70 => stack+=number(line().toDouble)
        case 71 => stack+=number(in.readDouble())
        case 86 => stack+=str(unicodeLine())
        case op @ (88 | 0x8c | 0x8d | 66 | 67 | 0x8e | 0x96) =>
          val n=op match
            case 88 | 66 => le().toLong
            case 0x8c | 67 => in.readUnsignedByte().toLong
            case _ => java.lang.Long.reverseBytes(in.readLong())
          val bs=take(n)
          stack+=(if Set(88,0x8c,0x8d)(op) then str(CS.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(bs)).toString) else byteVal(bs))
        case 93 => stack+=seq("list",Nil)
        case 125 => stack+=dict(Nil)
        case 41 => stack+=seq("tuple",Nil)
        case 0x8f => stack+=seq("~set",Nil)
        case 0x90 => val v=marked();require(v.forall(kind(_)=="string"));collectionAppend("~set",v)
        case 108 => stack+=seq("list",marked())
        case 116 => stack+=seq("tuple",marked())
        case op @ (0x85 | 0x86 | 0x87) =>
          val n=op-0x84;require(stack.size>=n);val a=stack.takeRight(n).toVector;stack.dropRightInPlace(n);stack+=seq("tuple",a)
        case 100 => val v=marked();val d=dict(Nil);update(d,v);stack+=d
        case 97 => val v=pop();collectionAppend("list",Vector(v))
        case 101 => val v=marked();collectionAppend("list",v)
        case 115 => val v=pop();val k=pop();update(stack.last,Vector(k,v))
        case 117 => val v=marked();update(stack.last,v)
        case 113 => put(in.readUnsignedByte())
        case 114 => put(le())
        case 112 => put(line().toInt)
        case 0x94 => put(memo.size)
        case 104 => stack+=memo(in.readUnsignedByte())
        case 106 => stack+=memo(le())
        case 103 => stack+=memo(line().toInt)
        case 99 => stack+=global(line(),line())
        case 0x93 => val name=text(pop());stack+=global(text(pop()),name)
        case 0x81 =>
          val args=pop();val g=pop();require(kind(g)=="~global" && g("name").str=="ControlNetUnit" && kind(args)=="tuple" && items(args).isEmpty)
          stack+=Obj("type"->"~unit","modern"->g("module").str.endsWith(".args"))
        case 82 =>
          val args=pop();val g=pop();require(kind(g)=="~global" && kind(args)=="tuple");val a=items(args)
          val v=g("name").str match
            case "dtype" => require(a.size==3 && a(1)==bool(false) && a(2)==bool(true));Obj("type"->"~dtype","value"->dtype(text(a.head)))
            case "_reconstruct" => require(a.size==3 && kind(a.head)=="~global" && a.head("name").str=="ndarray" && a(1)==seq("tuple",Seq(integer(0))) && a(2)==byteVal(Array(98.toByte)));tag("~array")
            case "_frombuffer" => require(a.size==4);array(a(1),a(2),a(0),text(a(3)))
            case "encode" => require(a.size==2 && text(a(1))=="latin1");val s=text(a.head);require(s.forall(_<=255));byteVal(s.getBytes(CS.ISO_8859_1))
            case "set" => require(a.size==1 && kind(a.head)=="list" && items(a.head).forall(kind(_)=="string"));seq("~set",items(a.head))
            case enumName =>
              val allowed=enumName match
                case "InputMode" => Set("simple","batch","merge")
                case "ResizeMode" => Set("Just Resize","Crop and Resize","Resize and Fill")
                case "ControlMode" => Set("Balanced","My prompt is more important","ControlNet is more important")
                case "HiResFixOption" => Set("Both","Low res only","High res only")
                case "PuLIDMode" => Set("Fidelity","Extremely style")
                case "ControlNetUnionControlType" => Set("OpenPose","Depth","Soft Edge","Hard Edge","Normal Map","Segmentation","Tile","Inpaint","Unknown")
                case _ => Set.empty[String]
              require(a.size==1 && allowed(text(a.head)),"Unsupported pickle reduction")
              a.head
          stack+=v
        case 98 =>
          val state=pop();val target=stack.last
          kind(target) match
            case "~dtype" =>
              val a=items(state);require(kind(state)=="tuple" && a.size==8 && a.drop(2)==Vector(nil,nil,nil,integer(-1),integer(-1),integer(0)))
              target("value")=dtype(text(a(1))+target("value").str.drop(1))
            case "~array" =>
              val a=items(state);require(kind(state)=="tuple" && a.size==5 && a.head==integer(1) && kind(a(3))=="bool")
              val built=array(a(2),a(1),a(4),if a(3)("value").bool then "F" else "C")
              target.obj++=built.obj
            case "~unit" =>
              val f=fields(state)
              val built=if !target("modern").bool then state else
                require(f.keySet==Set("__dict__","__fields_set__","__private_attribute_values__") && kind(f("__fields_set__"))=="~set" && fields(f("__private_attribute_values__")).isEmpty);f("__dict__")
              require(kind(built)=="dict")
              target.obj.remove("modern")
              target.obj++=built.obj
            case _ => throw IllegalArgumentException("Unsupported pickle BUILD")
        case 46 =>
          require(input.available()==0 && stack.size==1,"Invalid pickle termination")
          val root=pop()
          // Rewalking with the same bounds as the writer rejects cycles, alias expansion and unresolved globals.
          write(root);return root
        case _ => throw IllegalArgumentException("Unsupported pickle opcode")
    throw IllegalArgumentException("Incomplete Fastload pickle")
