# Windows Vulkan inference verification — 2026-11-26

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
