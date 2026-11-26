# Windows Vulkan inference verification — 2026-11-26




## Windows v0.42

The ZIP SHA-256 is `41847f4897f95d97fa9dd6fd9eb3bc2cfae74ac7724c01919d92714134b8cecc`. The packaged executable ran MultiDiffusion through the MCP `img2img` tool with 320 × 320 tiles, 64 px overlap, and Gaussian blending. SD1.5 completed at 512 × 512 on Radeon 780M, and SDXL completed at 768 × 768 on RTX 3090 from a fresh ZIP extraction. Both runs used a system-only PATH and reported `cuda=false`.

## Windows v0.41

The ZIP SHA-256 is `e1facd1ce694e230076b37a1fcea8ed64bedc1197b92e507a23db757f670dc4c`. The Windows executable ran the Weight Gradient hook through the MCP `img2img` tool using the bundled Vulkan DLL and JDK 25 with a system-only PATH. The staged package completed an SD1.5 run on Radeon 780M at 512 × 512. The executable from a fresh ZIP extraction completed an SDXL run on RTX 3090 at 768 × 768. Both workers reported `cuda=false`.

## Windows v0.40

The ZIP SHA-256 is `e08c17493fe1be4a16b4b4f347889fcac30818c2abcc9f0c627064653c1027ef`. The executable ran Prompt Fusion through the MCP `img2img` tool using the bundled JDK 25 and Vulkan DLL with a system-only PATH. The staged package generated at 512 × 512 on Radeon 780M with SD1.5 and at 768 × 768 on RTX 3090 with SDXL. A fresh ZIP extraction repeated the SDXL test on RTX 3090. The worker reported a Vulkan device and `cuda=false` in all three runs.

## Windows v0.39

The ZIP SHA-256 is `6829b85761a96f35318f66999ebe6380bb0baa60cb0285e82541534b4f122a31`. Using the bundled JDK and Vulkan DLL over MCP, the executable exported an unpadded conditioning PT and read it back for img2img. That image matched a `?prompt(..., no_padding=True)` and identity `?eval` on both Radeon 780M with SD1.5 at 512 x 512 and RTX 3090 with SDXL at 768 x 768. A fresh extraction of the ZIP also passed the MCP replay with SD1.5 on RTX 3090.

The two unpadded SD1.5 CLIP output rows each reached cosine similarity above 0.99999 against an independent CPU Transformers run with the same checkpoint and two input tokens. Export-only options retained the original generation pixels in paired tests.

## Windows v0.38

The v0.38 package includes the patched Vulkan DLL, JVM script runtime and the optional WD, character search and BLIP base models. Its ZIP SHA-256 is `e412150bfe62ce744bf0874b87cd6c27a67366f9fa7b11687b89b7514586c217`.

The staged executable ran img2img on Radeon 780M with SD1.5 at 512 x 512 and SDXL at 768 x 768 using the bundled Vulkan DLL and JDK 25. The MCP worker advertised Dynamic Thresholding, Deep Shrink, Token Downsampling, SVD, Chara Searcher and NSFW Checker. A script executed on the bundled JVM and returned the expected result with a system-only PATH.

## GPU comparison using the same packaged EXE

SDV-NUI v0.37 ran a single img2img job on each adapter with the same input PNG, three Euler steps, Karras scheduling, and seed 12345. The host had other applications running. The measurements are single runs.

| Checkpoint and size | Radeon 780M generation | RTX 3090 generation | Radeon 780M end to end | RTX 3090 end to end |
| --- | ---: | ---: | ---: | ---: |
| animekawa_v10, 512 x 512 | 42.77 s | 3.72 s | 60.80 s | 10.16 s |
| animaPencilXL_v500, 768 x 768 | 75.40 s | 10.07 s | 100.26 s | 22.35 s |

End-to-end times include process startup, model selection, generation and shutdown of the MCP connection.

## v0.34 Vulkan port

The extracted `SDV-NUI.exe` completed real img2img jobs over MCP on the Radeon 780M with SD1.5 at 512 × 512 and SDXL at 768 × 768. Each run used the same PNG source, three Euler steps, Karras scheduling and seed 12345. End-to-end times including worker startup, checkpoint loading and output were 28 and 77 seconds. Both runs used a system-only PATH and the bundled Java and inference runtimes. The ZIP tested here had SHA-256 `ddd5b4df7d4244f623a2fc8176a77b36e111a890c1e40017ace7eb763fa1c52d`.

The 780M also produced a three-frame SD1.5 AnimateDiff sequence and a 512 × 512 SDXL PhotoMaker v1 image using one reference. The motion frames were visually distinct. A four-step PhotoMaker run produced a recognizable fox with color artifacts. These short runs exercise the model paths. They do not measure identity retention or image quality. GIF, WebP, MP4 and WebM exports completed. The SD1.5 quick system benchmark completed ten 256 × 256 steps in 12.2 seconds, and the VRAM calibration recorded all four requested cases.

See the [feature inventory](vulkan-parity.md) for tested combinations and outstanding work. Intel hardware has not been tested.

## Earlier txt2img package

The tested Windows x64 ZIP was 370,275,849 bytes. Its SHA-256 was `a5f3d81967b846ec6cbc960bb4fe1597a1a5c799dad5600660d160ac6d538eb1`. The extracted desktop JAR was `4bf4ad5909bf0d6d9b14103ccc73e155a10faec9c1c6e7ecabdbe22465347d79` and the worker JAR was `e32ae68ad604283b46c72f530554e585b7c8a56869c8b893209b61cdd9a8fa76`. Build instructions and supported controls are in [WINDOWS_VULKAN.md](WINDOWS_VULKAN.md).

| Route | Selected device | Checkpoint | Size | Observed result |
|---|---|---|---|---|
| Extracted desktop, real Generate action via `--autogen` | Radeon 780M | animekawa_v10, SD1.5 | 512 × 512 | PNG, gallery preview and correct device label. Eight Euler steps, Karras, seed 12345. |
| Java 25 FFM smoke | Radeon 780M | animaPencilXL_v500, SDXL | 768 × 768 | Eight steps. Sampling took 117.07 s, VAE decode 16.54 s. |
| Extracted executable, external MCP client over stdio | GeForce RTX 3090 | animaPencilXL_v500, SDXL | 768 × 768 | Model selection, status and txt2img completed. PNG metadata reported the selected adapter and 14.48 s generation time. |
| Fresh Windows Sandbox, extracted ZIP, desktop Generate | GeForce RTX 3090 through the guest Vulkan driver | animekawa_v10, SD1.5 | 512 × 512 | PNG contained the RTX adapter name. The guest used the bundled runtimes with networking disabled and no CUDA Toolkit. |

Each image was inspected visually. The prompt asks for a fox in a meadow. Images showed the subject and landscape on both adapters. At the same seed the desktop AMD SD1.5 image matched the direct FFM invocation pixel for pixel. Timing here describes individual runs and includes different phases as labeled. Intel hardware was unavailable.

The clean Windows guest initially had the Vulkan loader but lacked the VC++ and OpenMP runtime DLLs. The final ZIP includes the required redistributable files. The clean guest verified the ZIP hash, extracted it to its own local disk, and generated an image with its own settings directory. The host runs also used a process PATH restricted to Windows system directories. The Java worker loaded ggml-vulkan and vulkan-1. The 23 shipped inference/runtime DLLs had no CUDA, cuDNN, cuBLAS or Torch imports reported by `dumpbin /dependents`. The assembled desktop JAR had exactly one engine provider, `sdfx.engine.vulkan.VulkanBackendProvider`.

## Protocol and coverage

A native cancellation test interrupted a 12-step job at its first step and then generated another image using the same model context. The packaged MCP dry-run also generated an image on the JVM. A direct worker session loaded SD1.5, retained its model token after a missing checkpoint was requested, switched to SDXL with a new token, and unloaded it. The worker protocol tests covered malformed-request recovery, unknown-device diagnostics, explicit hook rejection and normal shutdown. A fresh Vulkan profile defaults the Model Keyword hook off. Saving the GPU setting preserves hidden settings shared with conventional packages.

The migrated generation path covers checkpoint selection, txt2img, sampler and scheduler selection, clip skip, progress, interruption, batch images, PNG output and MCP/desktop integration. The worker's `hello.features` is the source of engine capability discovery. The legacy worker's feature list and the extension ledger include img2img, conditioning, postprocessing, upscaling, model editing and numerous extension hooks. Backend support for these features remains separate migration work. Generic host-side tools can still appear in the UI and MCP tool catalog.

## JVM test results

The initial parallel aggregate exposed the dry-run launcher, Model Keyword default, and protocol-field regressions. Three model-card scanner assertions also failed during concurrent execution and the full catalog suite passed in isolation. After corrections, the Vulkan-focused suites passed (142 tests with one scala-torch-specific skip); that skipped case passed on the conventional build alongside eight legacy launch and dry-run tests. The FFM worker protocol suite passed four tests. A subsequent sequential sbt run of all seven Vulkan-profile modules exited successfully, with 265 cases executed successfully, 13 skipped and prior successful tests reused by sbt 2 testQuick.
