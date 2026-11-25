package sdfx.ui.view

import javafx.geometry.{Insets, Pos}
import javafx.scene.control.*
import javafx.scene.image.Image
import javafx.scene.input.{Clipboard, ClipboardContent}
import javafx.scene.layout.{BorderPane, Priority, StackPane, VBox}
import sdfx.core.png.PngMetadata
import sdfx.ui.Fx
import sdfx.ui.vm.Mode

import java.io.ByteArrayInputStream
import java.nio.file.{Files, Path}
import java.util.Base64
import scala.util.control.NonFatal

/** Actions on a shown / selected image, shared by the preview buttons and the gallery context menu. */
trait ImageActions:
  def sendTo(mode: Mode, path: Path, infotext: Option[String]): Unit
  def restore(infotext: String): Unit
  def compare(sideB: Boolean, path: Path): Unit = ()
  def compareCurrent(): Unit = ()
  def delete(path: Path): Unit

object ImageActions:
  /** Copies decoded pixels; a scrambled source file is not offered as a plaintext image file. */
  def copyImage(ctx: UiContext, path: Path): Unit =
    ctx.imageCopyReads.request(path) {
      case scala.util.Success(display) =>
        val content = new ClipboardContent()
        if !display.scrambled then content.putFiles(java.util.List.of(path.toFile))
        content.putImage(display.image)
        if Clipboard.getSystemClipboard.setContent(content) then
          if display.scrambled && display.passwordVerified.isEmpty then ctx.info("Image copied; password could not be verified")
          else ctx.success("Image copied")
        else ctx.error("Cannot copy image to clipboard")
      case scala.util.Failure(error) => ctx.error("Cannot copy image: " + Fx.message(error).stripPrefix("requirement failed: "))
    }

  def openFolder(path: Path): Unit =
    try
      if System.getProperty("os.name").toLowerCase.contains("win") then
        new ProcessBuilder("explorer.exe", "/select,", path.toAbsolutePath.toString).start()
      else java.awt.Desktop.getDesktop.open(path.getParent.toFile)
    catch case NonFatal(e) => System.err.println(s"[sdfx] cannot open folder: ${e.getMessage}")

/** Big preview (live engine previews, then the final image) with its infotext and actions. */
final class PreviewPane(ctx: UiContext, actions: ImageActions):
  val zoom = new ZoomImageView()
  private val empty = new Label("Generated images appear here\nCtrl+Enter to generate")
  empty.setGraphic(Icons.sparkles(28))
  empty.setContentDisplay(ContentDisplay.TOP)
  empty.setGraphicTextGap(10)
  empty.getStyleClass.add("empty-preview")
  empty.setMouseTransparent(true)
  def imageStatus: String = empty.getText
  private val imageReads = new sdfx.ui.vm.ImageReadGate[ImageFiles.DisplayImage](ctx.services.imageScrambleSession,
    run => Fx.run(run()), (path, secret) => ImageFiles.readForDisplay(path, secret))(using Fx.bg)
  private val banner = new Label("")
  banner.getStyleClass.add("banner")
  banner.setWrapText(true)
  banner.setVisible(false)
  banner.setManaged(false)
  private val liveBadge = W.label("LIVE PREVIEW", "live-badge")
  liveBadge.setVisible(false)
  private val expandBtn = W.vectorButton(Icons.maximize(14), "Enlarge image (click image)")(openLightbox())
  expandBtn.getStyleClass.add("preview-expand-btn")
  expandBtn.setId("preview-enlarge-btn")
  expandBtn.setVisible(false)
  expandBtn.setManaged(false)
  private val stack = new StackPane(empty, zoom, liveBadge, expandBtn, banner)
  stack.getStyleClass.add("preview-stack")
  StackPane.setAlignment(liveBadge, Pos.TOP_LEFT)
  StackPane.setMargin(liveBadge, new Insets(10))
  StackPane.setAlignment(expandBtn, Pos.TOP_RIGHT)
  StackPane.setMargin(expandBtn, new Insets(10))
  StackPane.setAlignment(banner, Pos.TOP_CENTER)
  StackPane.setMargin(banner, new Insets(14))
  zoom.onImageClick = Some(() => openLightbox())
  W.tip(zoom, "Click: enlarge · Ctrl+wheel: zoom · drag: pan")

  private val info = new TextArea()
  info.setEditable(false)
  info.setWrapText(true)
  info.getStyleClass.add("infotext")
  info.setPromptText("Generation parameters of the shown image")
  info.setPrefRowCount(4)
  info.setMinHeight(36)
  private val fileLabel = W.label("", "file-label")

  @volatile private var current: Option[(Path, Option[String])] = None
  def currentPath: Option[Path] = current.map(_._1)
  ctx.watchImageAccess(zoom)(() =>
    if zoom.getUserData != null then current.foreach((path, info) => show(path, info)))

  private def withCurrent(f: (Path, Option[String]) => Unit): Unit =
    current match
      case Some((p, t)) => f(p, t)
      case None         => ctx.info("No image selected")

  private val compareButton = W.button("Compare", "Compare current mode input/output or previous txt2img result")(actions.compareCurrent())
  compareButton.setGraphic(Icons.columns(13))
  compareButton.setGraphicTextGap(5)
  compareButton.setId("preview-send-comparison")
  def comparisonButton(option: String): Unit =
    compareButton.setText(if option == "Icon" then "" else "Compare")
    compareButton.setContentDisplay(if option == "Icon" then ContentDisplay.GRAPHIC_ONLY else ContentDisplay.LEFT)
    compareButton.setManaged(option != "Off")
    compareButton.setVisible(option != "Off")
  private def actionBtn(label: String, icon: javafx.scene.Node, tip: String)(f: => Unit): Button =
    val b = W.button(label, tip)(f)
    b.setGraphic(icon)
    b.setGraphicTextGap(5)
    b
  private val buttons = W.hbox(
    6, compareButton,
    actionBtn("img2img", Icons.send(13), "Send this image and its parameters to img2img")(withCurrent((p, t) => actions.sendTo(Mode.Img2Img, p, t))),
    actionBtn("Inpaint", Icons.brush(13), "Send this image and its parameters to Inpaint")(withCurrent((p, t) => actions.sendTo(Mode.Inpaint, p, t))),
    actionBtn("Pano", Icons.globe(13), "View this image as a 360° panorama in the Panorama tab")(withCurrent((p, _) => ctx.sendToPanorama(p))),
    actionBtn("Restore", Icons.restore(13), "Restore the generation parameters of this image into the current tab")(withCurrent((_, t) =>
      t match
        case Some(text) => actions.restore(text)
        case None       => ctx.info("This image has no generation parameters")
    )),
    W.spacer(),
    W.vectorButton(Icons.copy(14), "Copy the image to the clipboard")(withCurrent((p, _) => { ImageActions.copyImage(ctx, p) })),
    W.vectorButton(Icons.folder(14), "Show in folder")(withCurrent((p, _) => ImageActions.openFolder(p))),
    W.vectorButton(Icons.trash(14), "Delete this image (asks first)")(withCurrent((p, _) => actions.delete(p)))
  )
  buttons.getStyleClass.add("preview-actions")

  private val compareActions = new javafx.scene.layout.FlowPane(6, 4,
    W.button("Compare → A", "Use this image as comparison A", Seq("compact-button"))(withCurrent((p, _) => actions.compare(false, p))),
    W.button("Compare → B", "Use this image as comparison B", Seq("compact-button"))(withCurrent((p, _) => actions.compare(true, p))))
  private val infoBox = new VBox(6, W.hbox(8, fileLabel), info, buttons, compareActions)
  infoBox.getStyleClass.add("info-box")
  infoBox.setPadding(new Insets(8, 10, 8, 10))
  VBox.setVgrow(info, Priority.ALWAYS)

  val imageNode: StackPane = stack
  val infoNode: VBox = infoBox

  def setBanner(text: Option[String]): Unit =
    banner.setText(text.getOrElse(""))
    banner.setVisible(text.isDefined)
    banner.setManaged(text.isDefined)

  // live previews are decoded in the background; frames that arrive while one is decoding replace each other
  private var shownGeneration = 0L
  private var liveBusy = false
  private var liveNext: Option[String] = None

  private def openLightbox(): Unit =
    if zoom.getUserData != null then current.foreach((p, _) => ctx.openLightbox(p, true))
    else Option(zoom.image).filter(!_.isError).foreach(img => ctx.openLightboxImage(img, "Live preview"))

  private def setExpandVisible(visible: Boolean): Unit =
    expandBtn.setVisible(visible)
    expandBtn.setManaged(visible)

  /** Shows a saved image; parameters are read from the PNG unless given. */
  def show(path: Path, infotext: Option[String] = None): Unit =
    zoom.setUserData(path)
    shownGeneration += 1
    liveNext = None
    liveBadge.setVisible(false)
    setExpandVisible(false)
    zoom.setImage(null)
    empty.setText("Loading image…")
    empty.setVisible(true)
    fileLabel.setText(path.getFileName.toString)
    current = Some(path -> infotext)
    imageReads.request(path) {
      case scala.util.Success(display) if zoom.getUserData == path =>
        zoom.setImage(display.image)
        empty.setVisible(false)
        setExpandVisible(true)
        if display.scrambled && display.passwordVerified.isEmpty then
          fileLabel.setText(path.getFileName.toString + " · password unverified")
      case scala.util.Failure(error) if zoom.getUserData == path =>
        zoom.setImage(null)
        setExpandVisible(false)
        empty.setText("Cannot display image: " + Fx.message(error).stripPrefix("requirement failed: "))
        empty.setVisible(true)
      case _ => ()
    }
    info.setText(infotext.getOrElse(""))
    if infotext.isEmpty then
      Fx.async(PngMetadata.readParameters(path)) { t =>
        if current.exists(_._1 == path) then
          current = Some(path -> t)
          info.setText(t.getOrElse(""))
      }(_ => ())

  /** Shows a live engine preview (base64 JPEG), decoded off the FX thread; late frames are dropped. */
  def showLive(b64: String): Unit =
    imageReads.invalidate()
    if liveBusy then liveNext = Some(b64)
    else
      liveBusy = true
      val gen = shownGeneration
      Fx.async(new Image(new ByteArrayInputStream(Base64.getDecoder.decode(b64)))) { img =>
        liveBusy = false
        if gen == shownGeneration && !img.isError then
          zoom.setUserData(null) // visible live pixels no longer belong to the last saved file
          zoom.setImage(img)
          empty.setVisible(false)
          liveBadge.setVisible(true)
          setExpandVisible(true)
        liveNext.foreach { n =>
          liveNext = None
          showLive(n)
        }
      } { _ =>
        liveBusy = false
        liveNext = None
      }

  /** Ends live previews (job finished / failed / cancelled): frames still decoding are dropped. */
  def stopLive(): Unit =
    shownGeneration += 1
    liveNext = None
    liveBadge.setVisible(false)

  def clear(): Unit =
    imageReads.invalidate()
    empty.setText("Generated images appear here\nCtrl+Enter to generate")
    shownGeneration += 1
    current = None
    zoom.setUserData(null)
    zoom.setImage(null)
    empty.setVisible(true)
    liveBadge.setVisible(false)
    setExpandVisible(false)
    fileLabel.setText("")
    info.clear()

