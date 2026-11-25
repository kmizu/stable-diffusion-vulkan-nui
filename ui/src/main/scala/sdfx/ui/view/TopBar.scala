package sdfx.ui.view

import javafx.geometry.{Insets, Pos}
import javafx.scene.control.*
import javafx.scene.layout.{HBox, Region}
import sdfx.core.protocol.{MemoryInfo, ModelEntry}
import sdfx.engine.{EngineSelection, EngineState}
import sdfx.ui.vm.ModelVM

/** Checkpoint / VAE / clip skip selection, engine status and settings. */
final class TopBar(
    ctx: UiContext,
    model: ModelVM,
    onCheckpoint: String => Unit,
    onVae: String => Unit,
    onRefresh: () => Unit,
    onRestart: () => Unit,
    onSettings: () => Unit,
    onToggleTheme: () => Unit = () => ()
):
  // ---- checkpoint picker -------------------------------------------------------------------------------
  private val ckptName = W.label("", "picker-name")
  private val ckptBadge = W.label("", "badge")
  private val ckptChevron = Icons.chevronDown(12)
  private val ckptBtn = new Button()
  ckptBtn.setGraphic(W.hbox(8, ckptName, ckptBadge, W.spacer(), ckptChevron))
  ckptBtn.getStyleClass.add("picker")
  ckptBtn.setMinWidth(220)
  ckptBtn.setPrefWidth(310)
  ckptBtn.setMaxWidth(440)
  W.tip(ckptBtn, "Stable Diffusion checkpoint (click to search)")

  private val popup = new SearchPopup[ModelEntry](
    "Checkpoint",
    () => ctx.catalog.snapshot.get.checkpoints,
    m => ctx.checkpointLabel(m.name),
    m => Option(m.kind).filter(_.nonEmpty),
    m => onCheckpoint(m.name),
    "No checkpoints found.\nPut .safetensors in models/checkpoints or add folders in Settings."
  )
  ckptBtn.setOnAction(_ => popup.show(ckptBtn))

  private def kindOf(name: String): Option[String] =
    val entry = ctx.catalog.snapshot.get.checkpoints.find(_.name == name)
    TopBar.checkpointKind(entry, ctx.services.supervisor.loadedModel)

  def refreshCheckpoint(): Unit =
    val n = model.checkpoint.get
    ckptName.setText(if n.isEmpty then "Select checkpoint…" else ctx.checkpointLabel(n))
    val k = kindOf(n)
    ckptBadge.setText(k.getOrElse(""))
    ckptBadge.setVisible(k.isDefined)
    ckptBadge.getStyleClass.removeIf(_.startsWith("badge-"))
    k.foreach(x => ckptBadge.getStyleClass.add(s"badge-${x.toLowerCase.replaceAll("[^a-z0-9]", "")}"))
  model.checkpoint.addListener((_, _, _) => refreshCheckpoint())
  ctx.catalog.snapshot.addListener((_, _, _) => refreshCheckpoint())
  ctx.hiresPreferences.addListener((_, _, _) => refreshCheckpoint())
  refreshCheckpoint()

  private val refreshBtn = W.vectorButton(Icons.refresh(15), "Rescan model folders")(onRefresh())

  // ---- VAE / clip skip ---------------------------------------------------------------------------------
  private val vae = new ComboBox[String](ctx.catalog.vaes)
  vae.setValue(model.vae.get)
  vae.setPrefWidth(175)
  W.tip(vae, "VAE: Automatic = the checkpoint's own VAE")
  private var syncing = false
  vae.valueProperty.addListener((_, _, v) =>
    if !syncing && v != null && v != model.vae.get then
      model.vae.set(v)
      onVae(v)
  )
  model.vae.addListener((_, _, v) =>
    if v != vae.getValue then
      syncing = true
      try vae.setValue(v)
      finally syncing = false
  )
  ctx.catalog.vaes.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) =>
    syncing = true
    try vae.setValue(model.vae.get)
    finally syncing = false
  )

  private val clip = W.intSpinner(model.clipSkip, 1, 12, 1, 68)
  W.tip(clip, "Clip skip: 1 = normal, 2 is common for anime models")

  // ---- engine status -----------------------------------------------------------------------------------
  private val chip = W.label("● Starting", "chip", "chip-starting")
  private val restartBtn = W.button("Restart", "Restart the inference engine", Seq("danger-button"))(onRestart())
  restartBtn.setGraphic(Icons.refresh(13))
  restartBtn.setGraphicTextGap(6)
  restartBtn.setVisible(false)
  restartBtn.setManaged(false)
  private val vram = W.label("", "vram")
  W.tip(vram, "GPU memory used by the engine (allocated / total)")
  private val themeBtn = W.vectorButton(Icons.sun(15), "Switch between dark and light theme")(onToggleTheme())
  themeBtn.setId("theme-toggle-btn")
  def setTheme(theme: String): Unit =
    themeBtn.setGraphic(if theme == "light" then Icons.moon(15) else Icons.sun(15))
  setTheme(ctx.services.settings.theme)
  private val settingsBtn = W.vectorButton(Icons.settings(15), "Settings")(onSettings())
  private val dryBadge = W.label("DRY RUN", "badge", "badge-dry")
  dryBadge.setVisible(ctx.services.dryRun)
  dryBadge.setManaged(ctx.services.dryRun)
  W.tip(dryBadge, "Dry-run engine: placeholder images, no GPU needed")

  def setState(s: EngineState, loading: Option[String]): Unit =
    refreshCheckpoint()
    chip.getStyleClass.removeIf(_.startsWith("chip-"))
    val native = EngineSelection.usesNativeBackend(ctx.services.settings)
    val (text, cls, tipText) = (s, loading) match
      case (EngineState.Crashed(msg), _) => ("● Engine crashed", "chip-crashed", msg)
      case (EngineState.Stopped, _)      => ("● Stopped", "chip-crashed", "The engine is not running")
      case (EngineState.Starting, _)     =>
        ("● Starting…", "chip-starting", if EngineSelection.availableBackends.contains("vulkan") then "Starting Vulkan engine…" else if native then "Starting scala-torch engine…" else "Starting the inference engine…")
      case (_, Some(what))               => (s"● Loading $what", "chip-busy", s"Loading $what")
      case (EngineState.Busy, _)         => ("● Busy", "chip-busy", "The engine is working")
      case (EngineState.Ready(h), _) =>
        val dev = if h.dryRun then "dry run" else h.device
        val shortDev = dev.stripPrefix("NVIDIA GeForce ").stripPrefix("NVIDIA ")
        val tip =
          if h.python.trim.isEmpty || h.engineVersion.toLowerCase(java.util.Locale.ROOT).contains("scala-torch") then
            s"Engine ${h.engineVersion} · Native JVM (no Python) · $dev"
          else
            s"Engine ${h.engineVersion} · Python ${h.python} · torch ${h.torch.getOrElse("-")} · diffusers ${h.diffusers.getOrElse("-")}"
        (s"● Ready · $shortDev", "chip-ready", tip)
    chip.setText(text)
    chip.getStyleClass.add(cls)
    W.tip(chip, tipText)
    val crashed = s match
      case EngineState.Crashed(_) | EngineState.Stopped => true
      case _                                            => false
    restartBtn.setVisible(crashed)
    restartBtn.setManaged(crashed)

  def setMemory(m: Option[MemoryInfo]): Unit =
    vram.setText(m.filter(_.totalMb > 0).map { x =>
      x.deviceUsedMb match
        case Some(used) => f"GPU total ${used / 1024}%.1f / ${x.totalMb / 1024}%.1f GB"
        case None => f"VRAM ${x.allocatedMb / 1024}%.1f / ${x.totalMb / 1024}%.1f GB"
    }.getOrElse(""))

  private val mcpChip = W.label("", "chip", "chip-mcp")
  mcpChip.setVisible(false)
  mcpChip.setManaged(false)

  /** MCP HTTP server status: Some(url) while it listens. */
  def setMcp(url: Option[String]): Unit =
    mcpChip.setVisible(url.isDefined)
    mcpChip.setManaged(url.isDefined)
    url.foreach { u =>
      val port = u.split(':').lastOption.map(_.takeWhile(_.isDigit)).getOrElse("")
      mcpChip.setText(s"⇄ MCP :$port")
      W.tip(mcpChip, s"MCP HTTP server listening on $u (Streamable HTTP). Configure it in Settings › MCP.")
    }

  private def sep(): Region =
    val r = new Region()
    r.getStyleClass.add("toolbar-sep")
    r.setMinWidth(1)
    r.setPrefWidth(1)
    r.setMaxWidth(1)
    r.setMinHeight(20)
    r.setPrefHeight(20)
    r.setMaxHeight(20)
    r

  private val title = W.label("SDNUI", "app-title")
  title.setMinWidth(Region.USE_PREF_SIZE)
  private val brand = W.hbox(8, AppIcon.emblem(22), title)
  brand.setMinWidth(Region.USE_PREF_SIZE)

  val node: HBox =
    val h = new HBox(
      8,
      brand,
      sep(),
      W.label("Checkpoint", "bar-label"),
      ckptBtn,
      W.button("Models…", "Browse model previews")(ctx.openModelBrowser("checkpoint", None)),
      refreshBtn,
      sep(),
      W.label("VAE", "bar-label"),
      vae,
      W.label("Clip skip", "bar-label"),
      clip,
      W.spacer(),
      dryBadge,
      vram,
      mcpChip,
      chip,
      restartBtn,
      sep(),
      themeBtn,
      settingsBtn
    )
    h.setAlignment(Pos.CENTER_LEFT)
    h.getStyleClass.add("top-bar")
    h.setPadding(new Insets(8, 14, 8, 14))
    h

object TopBar:
  private[ui] def checkpointKind(entry: Option[ModelEntry], loaded: Option[sdfx.core.protocol.LoadedModel]): Option[String] =
    def known(kind: String): Boolean = kind.nonEmpty && !kind.equalsIgnoreCase("unknown")
    entry.flatMap { model =>
      Option(model.kind).filter(known).orElse {
        loaded.filter(_.path == model.path).map(_.kind).filter(known)
      }
    }
