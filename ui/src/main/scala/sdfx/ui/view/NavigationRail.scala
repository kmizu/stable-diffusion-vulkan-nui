package sdfx.ui.view

import javafx.collections.ListChangeListener
import javafx.geometry.{Insets, Pos}
import javafx.scene.control.{Button, ContentDisplay, Label, ScrollPane, Tab, TabPane, Tooltip}
import javafx.scene.layout.{HBox, Priority, StackPane, VBox}

import java.util.IdentityHashMap
import scala.jdk.CollectionConverters.*

/** Native desktop sidebar navigation synchronized with the hidden `#main-tabs` TabPane.
  * Keeping all tabs in `#main-tabs` preserves 100% compatibility with existing tab lookup and i18n.
  */
final class NavigationRail(tabs: TabPane):
  private val createIds = Set("txt2img", "img2img", "inpaint")
  private val workspaceTabIds = Set(
    "png-info-tab", "extras-tab", "comparison-tab", "image-browser-tab",
    "panorama-tab-page", "styles-editor-tab", "prompt-tools-tab"
  )
  private val workspaceNames = Set(
    "PNG Info", "Extras", "Comparison", "Image Browser", "Panorama", "Styles", "Prompt Tools"
  )

  private var collapsed = false
  private var moreExpanded = false

  private val buttonsByTab = new IdentityHashMap[Tab, Button]()
  private val labelsByButton = new IdentityHashMap[Button, Label]()
  private val sectionLabels = scala.collection.mutable.ArrayBuffer.empty[Label]

  private val collapseBtn = new Button()
  collapseBtn.setId("nav-collapse-btn")
  collapseBtn.getStyleClass.addAll("icon-button", "nav-collapse-btn")
  collapseBtn.setGraphic(Icons.sidebarToggle(15))
  W.tip(collapseBtn, "Collapse or expand navigation sidebar")

  private val brandTitle = W.label("Workspace", "nav-header-title")
  private val topRow = new HBox(8, brandTitle, W.spacer(), collapseBtn)
  topRow.setAlignment(Pos.CENTER_LEFT)
  topRow.getStyleClass.add("nav-top-row")

  private val primaryBox = new VBox(2)
  private val workspaceBox = new VBox(2)
  private val moreListBox = new VBox(2)
  moreListBox.setVisible(false)
  moreListBox.setManaged(false)

  private val moreChevron = new StackPane(Icons.chevronRight(12))
  moreChevron.setMouseTransparent(true)
  private val moreText = W.label("More tools", "nav-more-label")
  private val moreCount = sdfx.ui.i18n.Localizer.skip(W.label("", "nav-more-count"))
  private val moreHeaderContent = new HBox(8, moreChevron, moreText, W.spacer(), moreCount)
  moreHeaderContent.setAlignment(Pos.CENTER_LEFT)
  private val moreToggle = new Button()
  moreToggle.setId("nav-more-tools-toggle")
  moreToggle.setGraphic(moreHeaderContent)
  moreToggle.setMaxWidth(Double.MaxValue)
  moreToggle.getStyleClass.add("nav-more-toggle")
  moreToggle.setOnAction(_ => setMoreExpanded(!moreExpanded))

  private def setMoreExpanded(expanded: Boolean): Unit =
    moreExpanded = expanded
    moreListBox.setVisible(expanded)
    moreListBox.setManaged(expanded)
    moreChevron.getChildren.setAll(if expanded then Icons.chevronDown(12) else Icons.chevronRight(12))

  private def iconForTab(tab: Tab): javafx.scene.Node =
    val id = Option(tab.getId).getOrElse("")
    val orig = sdfx.ui.i18n.Localizer.original(tab)
    (id, orig) match
      case (_, "txt2img")                                   => Icons.sparkles(15)
      case (_, "img2img")                                   => Icons.image(15)
      case (_, "Inpaint")                                   => Icons.brush(15)
      case (_, "PNG Info")                                  => Icons.info(15)
      case (_, "Extras")                                    => Icons.wand(15)
      case ("comparison-tab", _) | (_, "Comparison")        => Icons.columns(15)
      case ("image-browser-tab", _) | (_, "Image Browser")  => Icons.gallery(15)
      case ("panorama-tab-page", _) | (_, "Panorama")       => Icons.globe(15)
      case ("styles-editor-tab", _) | (_, "Styles")         => Icons.palette(15)
      case ("prompt-tools-tab", _) | (_, "Prompt Tools")    => Icons.sliders(15)
      case ("system-info-tab", _) | (_, "System Info")      => Icons.cpu(15)
      case (_, "VRAM Estimator")                             => Icons.cpu(15)
      case ("image-scrambling-tab", _)                      => Icons.lock(15)
      case ("controlnet-fastload-tab", _) | ("controlnet-fastload-gallery-tab", _) => Icons.layers(15)
      case _                                                => Icons.sliders(14)

  private def tabBucket(tab: Tab): Int =
    val id = Option(tab.getId).getOrElse("")
    val orig = sdfx.ui.i18n.Localizer.original(tab)
    if createIds.contains(orig.toLowerCase(java.util.Locale.ROOT)) then 0
    else if workspaceTabIds.contains(id) || workspaceNames.contains(orig) then 1
    else 2

  private def makeNavButton(tab: Tab): Button =
    val icon = iconForTab(tab)
    val label = sdfx.ui.i18n.Localizer.skip(W.label(tab.getText, "nav-item-label"))
    label.setMaxWidth(Double.MaxValue)
    HBox.setHgrow(label, Priority.ALWAYS)
    val row = new HBox(10, icon, label)
    row.setAlignment(Pos.CENTER_LEFT)
    val btn = new Button()
    btn.setGraphic(row)
    btn.setContentDisplay(ContentDisplay.GRAPHIC_ONLY)
    btn.setMaxWidth(Double.MaxValue)
    btn.getStyleClass.add("nav-item")
    btn.setFocusTraversable(true)
    val tip = new Tooltip(tab.getText)
    tip.setShowDelay(javafx.util.Duration.millis(400))
    btn.setTooltip(tip)
    tab.textProperty.addListener((_, _, text) => {
      label.setText(text)
      tip.setText(text)
    })
    btn.setOnAction(_ => tabs.getSelectionModel.select(tab))
    buttonsByTab.put(tab, btn)
    labelsByButton.put(btn, label)
    btn

  private def sectionHeader(title: String): Label =
    val l = W.label(title, "nav-section-label")
    sectionLabels += l
    l

  private val createHeader = sectionHeader("CREATE")
  private val workspaceHeader = sectionHeader("WORKSPACE")

  private val railBody = new VBox(4, createHeader, primaryBox, workspaceHeader, workspaceBox, moreToggle, moreListBox)
  railBody.setPadding(new Insets(4, 8, 12, 8))

  private val scroll = new ScrollPane(railBody)
  scroll.setFitToWidth(true)
  scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  scroll.getStyleClass.add("nav-scroll")
  VBox.setVgrow(scroll, Priority.ALWAYS)

  val node: VBox = new VBox(topRow, scroll)
  node.getStyleClass.add("nav-rail")
  node.setPrefWidth(196)
  node.setMinWidth(196)
  node.setMaxWidth(196)

  private def rebuild(): Unit =
    buttonsByTab.clear()
    labelsByButton.clear()
    primaryBox.getChildren.clear()
    workspaceBox.getChildren.clear()
    moreListBox.getChildren.clear()
    tabs.getTabs.asScala.foreach { tab =>
      val btn = makeNavButton(tab)
      tabBucket(tab) match
        case 0 => primaryBox.getChildren.add(btn)
        case 1 => workspaceBox.getChildren.add(btn)
        case _ => moreListBox.getChildren.add(btn)
    }
    val moreN = moreListBox.getChildren.size
    moreCount.setText(if moreN > 0 then moreN.toString else "")
    moreToggle.setVisible(moreN > 0)
    moreToggle.setManaged(moreN > 0)
    syncSelection(tabs.getSelectionModel.getSelectedItem)
    applyCollapsedState()

  private def syncSelection(selected: Tab): Unit =
    buttonsByTab.values().asScala.foreach(_.getStyleClass.remove("nav-item-active"))
    if selected != null then
      Option(buttonsByTab.get(selected)).foreach { btn =>
        btn.getStyleClass.add("nav-item-active")
        if tabBucket(selected) == 2 && !moreExpanded then setMoreExpanded(true)
      }

  private def applyCollapsedState(): Unit =
    val w = if collapsed then 52.0 else 196.0
    node.setPrefWidth(w)
    node.setMinWidth(w)
    node.setMaxWidth(w)
    brandTitle.setVisible(!collapsed)
    brandTitle.setManaged(!collapsed)
    sectionLabels.foreach { l => l.setVisible(!collapsed); l.setManaged(!collapsed) }
    moreText.setVisible(!collapsed)
    moreText.setManaged(!collapsed)
    moreCount.setVisible(!collapsed)
    moreCount.setManaged(!collapsed)
    labelsByButton.values().asScala.foreach { l => l.setVisible(!collapsed); l.setManaged(!collapsed) }
    if collapsed then node.getStyleClass.add("nav-rail-collapsed")
    else node.getStyleClass.remove("nav-rail-collapsed")

  collapseBtn.setOnAction(_ => {
    collapsed = !collapsed
    applyCollapsedState()
  })

  tabs.getTabs.addListener((_: ListChangeListener.Change[? <: Tab]) => rebuild())
  tabs.getSelectionModel.selectedItemProperty.addListener((_, _, next) => syncSelection(next))
  rebuild()
