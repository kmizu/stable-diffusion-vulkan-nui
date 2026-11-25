package sdfx.ui.view

import javafx.scene.image.{Image, ImageView}
import javafx.stage.Stage

import scala.util.control.NonFatal

/** Loads the multi-resolution SDNUI brand icons and applies them to JavaFX stages and toolbars. */
object AppIcon:
  val Sizes: Seq[Int] = Seq(16, 24, 32, 48, 64, 128, 256)

  def resourcePath(size: Int): String = s"/sdfx/ui/icons/icon-$size.png"

  lazy val images: Seq[Image] =
    Sizes.flatMap { size =>
      try
        Option(getClass.getResourceAsStream(resourcePath(size))).flatMap { stream =>
          try
            val img = new Image(stream)
            if img.isError then None else Some(img)
          finally stream.close()
        }
      catch case NonFatal(_) => None
    }

  def apply(stage: Stage): Unit =
    if images.nonEmpty then stage.getIcons.setAll(images*)

  def emblem(size: Double = 22): ImageView =
    val targetPx = if size * 2 > 64 then 128 else 64
    val view = new ImageView()
    try
      Option(getClass.getResourceAsStream(resourcePath(targetPx))).foreach { stream =>
        try
          val img = new Image(stream)
          if !img.isError then view.setImage(img)
        finally stream.close()
      }
    catch case NonFatal(_) => ()
    view.setFitWidth(size)
    view.setFitHeight(size)
    view.setPreserveRatio(true)
    view.setSmooth(true)
    view.getStyleClass.add("app-emblem")
    view
