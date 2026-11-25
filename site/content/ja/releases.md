---
title: リリースノート
nav: リリースノート
description: Stable Diffusion NUI（旧 SDFX）の各リリースで追加・変更された機能と、配布ファイルの SHA-256 の一覧です。
---

各リリースの配布ファイルは GitHub のリリースページからダウンロードできます。どの版も Java ランタイムを同梱していますが、モデルファイルは別に用意する必要があります（v0.11 以降は Python なしの `scala-torch` が標準です）。

## v0.33

2026-11-25 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.33)

- **SDNUI モノグラムファビコン・アプリアイコン刷新**: 幾何学「N」ディフュージョンプリズム＋スパークデザインへ刷新し、Web マニュアル（ルート `/favicon.ico`・`assets/favicon.svg`・PWA アイコン）、JavaFX 全ウィンドウ（フルスクリーン `PanoramaTab` を含む）、`SDNUI.exe` 埋め込みアイコンを統一しました。
- **`scala-torch 0.1.5` 高速化カーネル（Maven Central 公開版・既定有効）**:
  - **FlashAttention-2 32×32 ディスパッチ修正**: SDXL 1024×1024 / SD1.5 512×512 self-attention のマルチワープ化。
  - **SageAttention + SpargeAttn（既定 ON）**: K チャネル平均平滑化 + INT8 `QK^T` + オンライン warp スキップを高解像度アテンション（`qt, kt >= 2048`）へ自動適用。
  - **動的 W8A8 INT8 FFN Linear + Fused GEGLU（既定 ON）**: `CUBLAS_COMPUTE_32I` を大規模 FFN 層（`M >= 1024`, `M*N*K >= 8e9`）に限定適用し、画質を損なわず演算量を削減。
- **Limited Interval CFG & 正式版 AYS 拡張**: `CFG Schedule`（`active_start`）と `Fast mode`（既定 `guidance_from = 0.10`、既存インフォテキスト後方互換）へ Limited Interval CFG を搭載し、`Align Your Steps` を `DPM++ 2M SDE` に加えて `DPM++ 2M`（zero terminal sigma）および `Euler a` にも対応させました。
- **画質優先ポリシーによる除外**: 実機目視で滲みや局所ノイズが出た `DaTo (2×2)` と `Bottleneck Sampling` は収録を見送り、鮮明さを保つ手法のみ搭載しています（[v0.33 の検証記録](https://github.com/kmizu/stable-diffusion-nui/blob/main/docs/v0.33-speedup-verification.md)）。

**SDNUI-v0.33-windows-x64.zip** を展開し、**SDNUI/SDNUI.exe** を起動してください。SHA-256: `c26485b6c18ee5412f4fd00b35675a0b102305fb81ca801b6f120599f74dc9e6`

## v0.32

2026-11-23 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.32)

- **推論高速化（`scala-torch 0.1.4`）**: HiCache 収縮係数、FasterCache CFG-Cache + 高周波強調、PostDiff 段階的解像度拡散を追加し、`Fast mode` デフォルトを刷新しました。
- **UI・脱 Python 改善**: 画像クリック拡大ライトボックス、モザイク自動検閲（NudeNet）の純 JVM ネイティブ移植、生成履歴・プリセット保存上限の引き上げを行いました。

**SDNUI-v0.32-windows-x64.zip** を展開し、**SDNUI/SDNUI.exe** を起動してください。SHA-256: `67f9ef9ec993e333f9e7303ffc7e97fc11f643e02aacfb48cd62ff66405c2a1d`

## v0.31

2026-11-23 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.31)

- **Python レス標準起動と純 JVM ドライラン**: `scala-torch` (`0.1.3-gh53`) を標準エンジンとし、`--dry-run` も Python 不要の純 JVM ワーカーで動作します。
- **研究論文ベースの高速化**: FlashAttention-2、DeepCache + TaylorSeer 予測、T-GATE クロスアテンションキャッシュ、Align Your Steps (AYS) スケジュールを追加・統合し、Fast mode の PNG 復元と MCP メタデータ検査にも対応しました。
- **ネイティブ風デスクトップ UI 刷新**: 左側の折りたたみ式ナビゲーションサイドバー、Windows ダーク/ライトタイトルバー連携、ベクターアイコン、進捗トラック付きスライダー、Fast mode クイックトグル、拡張機能検索・有効フィルタを追加しました。

**SDNUI-v0.31-windows-x64.zip** を展開し、**SDNUI/SDNUI.exe** を起動してください。SHA-256: `7251fa25326898b8f9b4967e7f90bb4042db4ffcd85cd650e7437ff4929b6ca5`

## v0.1

2026-11-02 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.1)

日本語パックを内蔵し、[設定方法](localization.html)と[最初の画像の作り方](first-image.html)を案内しています。日本語化は部分的で、重複を除く実画面の文言の 74.1% が対象です。v0.06 以降の Panorama、compile、Prompt Tools、State Manager、タグ補完、Corridor Crawler は部分実装です。

**SDNUI-v0.1-windows-x64.zip** を展開し、**SDNUI/SDNUI.exe** を起動してください。SHA-256: `3ea05b8e5b0fec575cc4282459d03f3766766617f681a2927e081e1c79ef91b3`

## v0.06

2026-11-02 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.06)

### 追加

- **Aesthetic Score**、**AnimateDiff**、**Booru Prompt**、**Character Searcher**、**CLIP IO**。
- **部分実装**(未対応の範囲を文書化済みで、issue は開いたまま): Aesthetic Gradients、Batch Face Swap、CLIP Interrogator、CN Animation(動画から動画のみ)、Extras の ControlNet Preprocessor、画面の Localization(組み込みの日本語パック)、ネイティブの Image Browser タブ。詳細は[拡張機能のページ](extensions.html)と GitHub のリリースノートにあります。

### 入手方法

**SDNUI-v0.06-windows-x64.zip** をダウンロードして展開し、**SDNUI/SDNUI.exe** を起動します。SHA-256: `e12d6a7f1e0240a8cffe90015d8bdf2cccec27550137f12ecdf00e7d64fcaa4c`

### 検証

- Scala と Torch CPU のテストを全件実行し、各機能を実際の SD1.5 と SDXL で MCP 経由で確認して、画像を目視しました。
- 配布版の `SDNUI.exe` を両モデルで実行し、ダウンロードした ZIP の SHA-256 をビルドと照合しました。

## v0.05

2026-11-02 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.05)

v0.04 から拡張機能を 10 件追加しました。

### 追加・変更

- **ガイダンスとアテンション**: Semantic Guidance (#115)、Agent Attention (#4)、CD Tuner (#9)、Characteristic Guidance (#11)、Incantations (#63)。
- **サンプリングとアップスケール**: Sonar (#118)、LLuL (#75)、Loopback Scaler (#76)、Old SD firstpasser (#94)。
- **安全機能**: NSFW checker (#91)。NudeNet による検閲、ライブプレビュー、Extras、SD safety checker と NSFWJS に対応します。

### 入手方法

**SDNUI-v0.05-windows-x64.zip** をダウンロードして展開し、**SDNUI/SDNUI.exe** を起動します。SHA-256: `fbdb3ba845bf66b37a3dc006d8b80b7e3525b0f15e1b24b2010910ae10920523`

### 検証

- Scala と Torch CPU のテストを全件実行しました。
- 各機能を実際の SD1.5 と SDXL モデルで MCP 経由で確認し、画像を目視しました。配布版の `SDNUI.exe` も両モデルで実行しました。

## v0.04

2026-11-01 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.04)

v0.03 以降に 10 件の issue を解決し、AI エージェントから MCP で使うためのワークフローを追加しました。

### 追加・変更

- **ControlNet Fastload（#17）**: 生成の前に、Fastload 画像または `.cni` ファイルから ControlNet のユニットを読み込めるようになりました。ユニットは各画像と一緒に保存できます（画像に埋め込む、別ファイルにする、またはその両方）。形式は A1111 の拡張機能と互換です。また、組み込みの ControlNet に、コントロールモード、canny のプリプロセッサ解像度、Pixel Perfect を追加しました。
- **埋め込み**: Embedding Inspector（#46）、Embedding Merge（#47）、Embedding Editor（#45）を追加しました。
- **生成**: img2img の Hires Fix（#61）、Conditioning Highres Fix（#16）、Model Presets（#87）、Model Keyword（#85）を追加しました。
- **ファイル**: チェックポイントや LoRA のメタデータを扱う Metadata Utils（#82）と、アップストリームの拡張機能と互換の画像スクランブル（#48）を追加しました。画像スクランブルは安全な暗号化ではありません。
- **MCP**: エージェントが、モデル、ControlNet、AlwaysOn の項目を調べ、ジョブごとに拡張機能を設定し、画像の生成・編集、進捗の確認、PNG メタデータの確認と再現、ジョブの取り消しを行えるようになりました。stdio と Streamable HTTP の両方で使えます。詳しくは [MCP](mcp.html) を参照してください。

### 使い方

**SDNUI-v0.04-windows-x64.zip** をダウンロードして展開し、**SDNUI/SDNUI.exe** を起動します。MCP サーバーとして使うときは `SDNUI.exe --mcp` で起動します。既存の `sdfx` の設定キーはそのまま使えます。

### 検証

- Scala と Torch CPU のテスト一式を実行しました。
- 各機能を実際の SD1.5 と SDXL のモデルで確認し、生成画像を目で確認しました。
- 公式の MCP Python SDK から、stdio と HTTP の両方で SD1.5 / SDXL を使った操作を確認しました。
- 展開したパッケージの `SDNUI.exe` で確認し、同梱のソースファイルと jar がリリースのコミットと一致することを確かめました。

各機能の制限は、リポジトリの `docs/*-verification.md` に記録しています。

ZIP の SHA-256: `b58c4a8dab854598f3e5072408ba24c37cf15996f8766d40ccc798ef62d38cb7`

## v0.03

2026-10-27 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.03)

Stable Diffusion FX の名前を **Stable Diffusion NUI** に変更しました。

### 追加・変更

- **ローカルのタグ補完**: 自分の CSV を読み込み、キーボードで候補を選んで、入力中のタグだけを置き換えます。ポジティブ / ネガティブの両方のプロンプトに対応し、IME で変換中の入力は補完の対象から外します。
- **名前付き生成設定**: モードごとの設定全体を、選択式のプリセットとは別に保存・確認・復元・削除できます。必要なリソースが見つからないときは、理由を表示して復元を止めます。
- **WD バッチの PNG メタデータ**: 実際に使われたプロンプト、設定、シード、WD の識別情報が、アプリを起動し直しても PNG Info と復元で使えるようになりました（#141）。
- **バッチの流れ**: 同時に生成したすべての画像がライブプレビューに表示されます。生成ごとに別のギャラリーがあり、ZIP で書き出せます。

### 使い方

**SDNUI-v0.03-windows-x64.zip** をダウンロードして展開し、**SDNUI/SDNUI.exe** を起動します。既存の `sdfx` の設定キーはそのまま使えます。

### 検証と制限

ネイティブ UI の確認、SD1.5 / SDXL での基本的な生成と PNG の再読み込み、代表的な VRAM 設定での生成と解放、展開した Windows の EXE で確認しました。キーボードと IME の確認は JavaFX の合成イベントで行っており、実際の日本語 IME の組み合わせをすべて試したわけではありません。

このリリースは合意した範囲のネイティブ実装で、アップストリームの拡張機能のすべての機能には対応していません。#23 と #24 は、より広い機能のために開いたままです。ほかのインテロゲーターや、LoRA / ControlNet / スクリプトとの連携は #140 で扱います。範囲、根拠、制限はリポジトリの `docs/V003_RELEASE_SCOPE.md` を参照してください。

ZIP の SHA-256: `75fca7e13167e0e5846d78904653bdc41710ebd04863a6c84fb82e577c5ea396`

## v0.02

2026-10-27 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.02)

### 追加・変更

- img2img に WD v3 バッチを追加しました。フォルダ内の画像を 1 枚ずつタグ付けしてから生成し、結果をギャラリーに表示します。
- バッチの取り消しをジョブ単位に限定し、Windows で生成画像の登録に失敗する問題を修正しました。
- Width / Height の入力欄をそろえました。モデルの追加は歯車アイコン → **Model folders** から行えます。

### 使い方

ZIP（`SDFX-v0.02-windows-x64.zip`）を展開して `SDFX/SDFX.exe` を起動します。

### 検証と制限

SD1.5 と SDXL のそれぞれで、実際の画面から 2 枚の画像を投入する基本動作を確認しました。

WD バッチは、指定したローカルの WD v3 モデルとタグ CSV に対応します。入手先、追加の依存関係、設定と再起動の手順は ZIP 内の `WINDOWS_RELEASE.md` に記載しています。このリリースの WD 対応は限定版で、ほかのインテロゲーター、LoRA / ControlNet / スクリプトとの併用、生成 PNG への WD メタデータの保存などは今後の課題でした。

ZIP の SHA-256: `057d0128942fbeb8979d4379688094d8963dff218d41c0323d9507432ef59451`

## v0.01

2026-10-27 公開。[GitHub のリリースページ](https://github.com/kmizu/stable-diffusion-nui/releases/tag/v0.01)

SDFX の Windows 向けプレビュー版です。

### 内容

- Java ランタイムを同梱した Windows x64 のネイティブアプリケーションです。
- モデルカードのプレビューと、その表示を切り替える **Show** / **Blur** / **Hide** を備えています。

### 使い方

ZIP（`SDFX-v0.01-windows-x64.zip`）を展開して `SDFX.exe` を起動します。Python 3.10〜3.12 と画像生成の依存関係は、初回起動時に別途セットアップするか、互換性のある既存の Python 環境を選びます。モデルファイルは含まれていないため、手元のモデルのフォルダを設定します。

### 検証と制限

パッケージ版のアプリをローカルの SD1.5 チェックポイントで確認し、モデルカードの機能は SDXL でも確認しました。初期のプレビュー版で、拡張機能の一覧に対する残りの作業は issue で管理しています。

ZIP の SHA-256: `6e4a182ff136f44f4be289d39ef61b4078a23e0d651ddf263a6a2d8a69942f8b`
