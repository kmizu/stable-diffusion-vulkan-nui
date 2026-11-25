package sdfx.vulkan.tools

import java.nio.file.{Files,Path}
import java.util.Base64
import munit.FunSuite

final class FastloadSuite extends FunSuite:
  private val fixtures =
    val relative=Path.of("python","tests","fixtures","controlnet_fastload")
    Iterator.iterate(Path.of("").toAbsolutePath)(_.getParent).takeWhile(_!=null)
      .map(_.resolve(relative)).find(Files.isDirectory(_)).getOrElse(fail("Missing upstream Fastload test fixtures"))

  test("read genuine legacy and Pydantic ControlNet units, and emit interoperable numeric NumPy buffers") {
    import FastloadPickle.*
    for dialect<-Seq("unit","modern-unit");protocol<-2 to 5 do
      val path=fixtures.resolve(s"$dialect-protocol-$protocol.pickle")
      val original=read(Files.readAllBytes(path))
      assertEquals(read(write(original)),original)
      val unit=fields(items(original).head)
      assertEquals(unit("enabled"),bool(true))
      val pixels=fields(unit("image"))("image")
      assertEquals(pixels("dtype").str,"|u1")
      assertEquals(shape(pixels),Vector(3,4,3))
      assertEquals(binary(pixels("data").str).toVector,(0 until 36).map(_.toByte).toVector)
  }

  test("a pending global is rejected before it can escape to Fastload wire data") {
    intercept[IllegalArgumentException] {
      FastloadPickle.read("cnumpy\ndtype\n.".getBytes(java.nio.charset.StandardCharsets.US_ASCII))
    }
  }
