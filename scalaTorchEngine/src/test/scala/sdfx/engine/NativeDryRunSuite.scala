package sdfx.engine

import sdfx.scalatorch.NativeDryRunWorker
import java.io.{BufferedReader, ByteArrayInputStream, ByteArrayOutputStream, InputStreamReader, PrintStream}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files

class NativeDryRunSuite extends munit.FunSuite:
  test("NativeDryRunWorker serves hello, list_samplers, load_model, txt2img and shutdown in pure JVM") {
    val tmp = Files.createTempDirectory("native-dryrun-test")
    try
      val requests = Seq(
        """{"id":"1","type":"list_samplers","params":{}}""",
        """{"id":"2","type":"load_model","params":{"path":"fake-sd15.safetensors"}}""",
        s"""{"id":"3","type":"txt2img","params":{"output_dir":"${tmp.toString.replace("\\", "\\\\")}","params":{"width":64,"height":64,"steps":2,"seed":7}}}""",
        """{"id":"4","type":"shutdown","params":{}}"""
      ).mkString("\n") + "\n"
      val out = new ByteArrayOutputStream()
      NativeDryRunWorker.serve(
        new BufferedReader(new InputStreamReader(new ByteArrayInputStream(requests.getBytes(UTF_8)), UTF_8)),
        out
      )
      val lines = out.toString(UTF_8).linesIterator.filter(_.nonEmpty).map(ujson.read(_).obj).toVector
      assertEquals(lines.head("type").str, "hello")
      assertEquals(lines.head("dry_run").bool, true)
      assert(lines.head("features").arr.map(_.str).contains("tgate"))
      assert(lines.exists(o => o.get("id").contains(ujson.Str("1")) && o("type").str == "result" && o("result")("samplers").arr.nonEmpty))
      assert(lines.exists(o => o.get("id").contains(ujson.Str("2")) && o("type").str == "result"))
      val gen = lines.find(o => o.get("id").contains(ujson.Str("3")) && o("type").str == "result").get
      val png = java.nio.file.Path.of(gen("result")("images").arr.head("path").str)
      assert(Files.isRegularFile(png))
      assertEquals(lines.last("id").str, "4")
      assertEquals(lines.last("type").str, "result")
    finally
      Files.walk(tmp).sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists)
  }
