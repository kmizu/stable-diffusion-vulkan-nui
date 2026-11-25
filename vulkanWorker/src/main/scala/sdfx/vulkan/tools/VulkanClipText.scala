package sdfx.vulkan.tools

import java.io.{BufferedOutputStream,DataInputStream,DataOutputStream}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path,StandardCopyOption}
import java.util.concurrent.{CompletableFuture,TimeUnit}
import scala.collection.mutable
import scala.util.Using

/** Native OpenCLIP ViT-H text tower; the input tensors come straight from the pinned Transformers checkpoint. */
private[tools] object VulkanClipText:
  private val revision="1c2b8495b28150b8a4922ee1c8edee224c284c0c"
  private val digest="036e6e2bd49697511f4f8b8cb5ee465f93025f7a69a145eadeb9a881ace9b18d"
  private def bin=Path.of(sys.env.getOrElse("SDFX_VULKAN_NCNN_DIR",sys.env.getOrElse("SDFX_VULKAN_LIBRARY_DIR","."))).toAbsolutePath
  def available:Boolean=Files.isRegularFile(bin.resolve("sdv-ncnn.exe")) && Files.isRegularFile(bin.resolve("ncnn.dll"))
  private lazy val tokenizer=ClipFastTokenizer.loadBundled()
  private def convert(model:Path,params:Path,weights:Path,check:()=>Unit):Unit =
    val graph=mutable.ArrayBuffer.empty[String]
    var outputs=0
    def layer(op:String,inputs:Seq[String],names:Seq[String],settings:String=""):Unit =
      val io=(inputs++names).mkString(" ")
      graph+=s"$op layer${graph.size} ${inputs.size} ${names.size} $io $settings"
      outputs+=names.size
    Using.resource(CheckpointIO.open(model)) { file =>
      Using.resource(new BufferedOutputStream(Files.newOutputStream(weights))) { out =>
        def tensor(key:String,tag:Boolean=false):Unit =
          check()
          val t=file.tensors(key)
          require(t.dtype=="F32",s"Expected float32 $key")
          if tag then out.write(Array.fill[Byte](4)(0))
          out.write(file.raw(t))
        def norm(prefix:String,input:String,output:String):Unit =
          layer("LayerNorm",Seq(input),Seq(output),"0=1024 1=1e-5 2=1")
          tensor(prefix+".weight");tensor(prefix+".bias")
        def linear(prefix:String,input:String,output:String,width:Int,count:Int,bias:Boolean=true):Unit =
          layer("InnerProduct",Seq(input),Seq(output),s"0=$count 1=${if bias then 1 else 0} 2=${width*count}")
          tensor(prefix+".weight",true);if bias then tensor(prefix+".bias")
        for in<-Seq("tokens","positions","mask") do layer("Input",Nil,Seq(in))
        layer("Split",Seq("mask"),(0 until 24).map(i=>s"mask$i"))
        layer("Embed",Seq("tokens"),Seq("words"),"0=1024 1=49408 2=0 3=50593792")
        tensor("text_model.embeddings.token_embedding.weight",true)
        layer("Embed",Seq("positions"),Seq("position"),"0=1024 1=77 2=0 3=78848")
        tensor("text_model.embeddings.position_embedding.weight",true)
        layer("BinaryOp",Seq("words","position"),Seq("h0"),"0=0")
        for i<-0 until 24 do
          val enc=s"text_model.encoder.layers.$i"
          layer("Split",Seq(s"h$i"),Seq(s"norm$i",s"skip$i"))
          norm(enc+".layer_norm1",s"norm$i",s"qkv$i")
          layer("MultiHeadAttention",Seq(s"qkv$i",s"mask$i"),Seq(s"att$i"),"0=1024 1=16 2=1048576 5=1")
          for which<-Seq("q_proj","k_proj","v_proj","out_proj") do
            val base=s"$enc.self_attn.$which"
            tensor(base+".weight",true);tensor(base+".bias")
          layer("BinaryOp",Seq(s"att$i",s"skip$i"),Seq(s"res$i"),"0=0")
          layer("Split",Seq(s"res$i"),Seq(s"normb$i",s"skipb$i"))
          norm(enc+".layer_norm2",s"normb$i",s"mlp$i")
          linear(enc+".mlp.fc1",s"mlp$i",s"inter$i",1024,4096)
          layer("GELU",Seq(s"inter$i"),Seq(s"gelu$i"),"0=0")
          linear(enc+".mlp.fc2",s"gelu$i",s"interb$i",4096,1024)
          layer("BinaryOp",Seq(s"interb$i",s"skipb$i"),Seq(s"h${i+1}"),"0=0")
        norm("text_model.final_layer_norm","h24","final")
        linear("text_projection","final","out",1024,1024,false)
      }
    }
    Files.writeString(params,s"7767517\n${graph.size} $outputs\n"+graph.mkString("\n")+"\n")
  private def ensure(cache:Path,check:()=>Unit,progress:(String,Int,Int)=>Unit):Path =
    val root=sys.env.get("SDFX_VULKAN_CLIP_TEXT_DIR").map(Path.of(_)).getOrElse(cache.resolve("text-h14")).toAbsolutePath
    Files.createDirectories(root)
    val result=root.resolve("ncnn-v1")
    if !Files.isRegularFile(result.resolve("text.param")) || !Files.isRegularFile(result.resolve("text.bin")) then
      val src=VulkanClip.asset(root,"model.safetensors",s"https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/$revision/model.safetensors",digest,check,progress)
      val stage=Files.createTempDirectory(root,".text-")
      try
        convert(src,stage.resolve("text.param"),stage.resolve("text.bin"),check)
        check()
        try Files.move(stage,result,StandardCopyOption.ATOMIC_MOVE)
        catch case _:java.nio.file.FileAlreadyExistsException=>()
      finally if Files.exists(stage) then
        Files.deleteIfExists(stage.resolve("text.param"));Files.deleteIfExists(stage.resolve("text.bin"));Files.deleteIfExists(stage)
    require(sdfx.core.imagebrowser.BrowserFiles.sha256(result.resolve("text.bin"))=="69354cb04a92aa7fa02a56302a34be45e88154a15ca73efe149ba390ba4f4e8d","CLIP text weight digest does not match")
    result
  final class Encoder(private val process:Process,check:()=>Unit) extends AutoCloseable:
    private val input=new DataOutputStream(process.getOutputStream)
    private val output=new DataInputStream(process.getInputStream)
    def score(text:String,image:Array[Float]):Double =
      check()
      val encoded=tokenizer.encode(text)
      val ids=if encoded.length<=77 then encoded else encoded.take(76)++Array(49407L)
      require(ids.length>=2 && ids.head==49406 && ids.last==49407)
      input.writeInt(Integer.reverseBytes(ids.length));ids.foreach(i=>input.writeInt(Integer.reverseBytes(i.toInt)));input.flush()
      val bytes=output.readNBytes(4096)
      require(bytes.length==4096,s"CLIP text encoder exited while scoring '$text'")
      val floats=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer
      var dot=0.0;var norm=0.0
      for j<-image.indices do
        val x=floats.get().toDouble
        dot+=x*image(j);norm+=x*x
      require(norm>0 && norm.isFinite,"Invalid text embedding")
      dot/math.sqrt(norm)
    def close():Unit =
      try input.close() finally
        if !process.waitFor(100,TimeUnit.MILLISECONDS) then process.destroyForcibly()
        output.close()
  def withEncoder[T](cache:Path,device:String,check:()=>Unit,progress:(String,Int,Int)=>Unit)(run:Encoder=>T):T =
    val folder=ensure(cache,check,progress)
    val process=new ProcessBuilder(bin.resolve("sdv-ncnn.exe").toString,"--clip-text",device,folder.resolve("text.param").toString,folder.resolve("text.bin").toString)
      .redirectError(ProcessBuilder.Redirect.INHERIT).start()
    val encoder=new Encoder(process,check)
    val work=CompletableFuture.supplyAsync(()=>run(encoder))
    try
      while !work.isDone do {check();Thread.sleep(75)}
      try work.get() catch case ex:java.util.concurrent.ExecutionException=>throw ex.getCause
    finally
      if !work.isDone then process.destroyForcibly()
      encoder.close()
