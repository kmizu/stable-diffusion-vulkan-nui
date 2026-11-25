package sdfx.vulkan

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.EngineError
import sdfx.vulkan.tools.VulkanTools
import ujson.{Arr, Obj, Value}

class VulkanToolsSuite extends munit.FunSuite:
  private def workspace[A](run: Path => A): A =
    val dir = Files.createTempDirectory("vulkan-tools-")
    try run(dir)
    finally Using.resource(Files.walk(dir))(_.iterator.asScala.toList.reverse.foreach(Files.delete))

  private def call(kind: String, value: Value, publish: () => Unit = () => ()): Value =
    VulkanTools.execute(kind, value, () => false, publish, (_,_) => (), "test device", None,
      (_,_,_,_) => throw AssertionError("unexpected model upscaler"))

  private def source(dir: Path): Path =
    val input = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB)
    input.setRGB(0,0,0x123456)
    val path = dir.resolve("input.png")
    PngMetadata.write(input,path,Seq("parameters" -> "original"))
    path

  test("extension registry deduplicates nested roots and treats marked directories as one model") {
    workspace { dir =>
      val set = Files.createDirectories(dir.resolve("set"))
      val model = Files.createDirectories(set.resolve("edge"))
      Files.createFile(model.resolve("config.json"))
      Files.createFile(model.resolve("weights.safetensors"))
      Files.createFile(set.resolve("standalone.SAFETENSORS"))
      val entries = call("list_files", Obj("kind" -> "controlnet", "dirs" -> Arr(dir.toString,set.toString)))("files").arr
      assertEquals(entries.map(_("name").str).toList,List("set/edge","set/standalone"))
    }
  }

  test("standalone preprocessing preserves source infotext and changes actual RGB samples") {
    workspace { dir =>
      val req = Obj("preprocessor" -> "invert", "standalone" -> true, "image_path" -> source(dir).toString,
        "output_dir" -> dir.toString)
      val value = call("preprocess",req)
      assertEquals(value("info")("module").str,"invert")
      val output = Path.of(value("path").str)
      assertEquals(ImageIO.read(output.toFile).getRGB(0,0) & 0xffffff,0xedcba9)
      assertEquals(PngMetadata.readTexts(output)("Source parameters"),"original")
      assert(PngMetadata.readParameters(output).exists(_.contains("ControlNet Preprocessor")))
    }
  }

  test("nearest and Lanczos resize and report hardware metadata, cancellation cleans unpublished output") {
    workspace { dir =>
      val req = Obj("upscaler" -> "Nearest", "scale" -> 2, "image_path" -> source(dir).toString,
        "output_dir" -> dir.toString, "hardware_info" -> Obj("enabled" -> true))
      val nearest = call("upscale",req)
      val image = ImageIO.read(Path.of(nearest("path").str).toFile)
      assertEquals((image.getWidth,image.getHeight),(8,6))
      assertEquals(image.getRGB(1,1) & 0xffffff,0x123456)
      assert(nearest("hardware_info").render().contains("test device"))
      req("upscaler") = "Lanczos"
      val lanczos = ImageIO.read(Path.of(call("upscale",req)("path").str).toFile)
      assert(image.getRGB(1,1) != lanczos.getRGB(1,1))
      def files = Using.resource(Files.list(dir))(_.iterator.asScala.toSet)
      val previous = files
      intercept[CancellationException](call("upscale",req,() => throw CancellationException()))
      assertEquals(files,previous)
    }
  }

  test("difference generation and preview agree on a known changed pixel") {
    workspace { dir =>
      val input = source(dir)
      val changed = ImageIO.read(input.toFile)
      changed.setRGB(2,1,0xffffff)
      val after = dir.resolve("altered.png")
      ImageIO.write(changed,"png",after.toFile)
      val settings = Obj("base_path" -> input.toString, "threshold" -> 1)
      val mask = VulkanTools.differenceMask(settings,changed,() => false)
      val preview = call("inpaint_difference_preview",Obj("base_path" -> input.toString,
        "altered_path" -> after.toString, "output_dir" -> dir.toString))
      val exported = ImageIO.read(Path.of(preview("mask_path").str).toFile)
      for y <- 0 until 3; x <- 0 until 4 do
        assertEquals(mask.getRGB(x,y) & 255,if x == 2 && y == 1 then 255 else 0)
        assertEquals(exported.getRGB(x,y), mask.getRGB(x,y))
    }
  }

  test("unsupported neural image tools report unsupported") {
    val err = intercept[EngineError](call("image_tool",Obj("tool" -> "not-a-tool")))
    assertEquals(err.code,"unsupported")
  }
  test("model keyword save, plan and generation share mapping rules, advancing iteration on commit") {
    workspace { dir =>
      val file = Files.createFile(dir.resolve("m.safetensors"))
      val catalog = Files.createDirectory(dir.resolve("mappings"))
      val saved = call("model_keyword",Obj("action" -> "save", "kind" -> "checkpoint",
        "model_path" -> file.toString, "mapping_dir" -> catalog.toString, "keywords" -> "red|blue"))
      assertEquals(saved("keywords").arr.map(_.str).toList,List("red","blue"))
      val options = Obj("_mapping_dir" -> catalog.toString,"model_choice" -> "iterate")
      val request = sdfx.core.protocol.GenerationRequest(sdfx.core.model.GenerationParams(
        prompt = "bird", hooks = Map("model_keyword" -> options)),dir.toString)
      val model = sdfx.core.protocol.LoadedModel("m",file.toString,"sd15","unused")
      val state = new VulkanTools.Session
      def plan = sdfx.vulkan.tools.VulkanKeyword.plan(request,model,state,42L,() => false).get
      val first = plan
      assertEquals(first.images.head.prompt,"red bird")
      assertEquals(plan.images.head.prompt,"red bird")
      first.commit()
      assertEquals(plan.images.head.prompt,"blue bird")
    }
  }
