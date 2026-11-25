# Windows x64 Vulkan edition

Extract the entire Vulkan ZIP to a writable directory, then open `SDV-NUI.exe`. In Settings, add a folder containing an SD1.5 or SDXL checkpoint to the checkpoint directories. Select the checkpoint in the main window and generate a txt2img image. The application includes its Java runtimes, inference DLL and Vulkan compute backend.

Install the graphics manufacturer's Vulkan-capable Windows driver. The package includes the Microsoft Visual C++ x64 runtime DLLs used by its native libraries. To exercise the desktop or MCP with a simulated image, add `--dry-run`. That mode runs on the JVM without loading a checkpoint or Vulkan DLL.

## Supported generation

The worker supports SD1.5 and SDXL txt2img, img2img, masked inpainting, hires passes, external VAEs, LoRAs, textual inversion and single-network ControlNet. Sampler and scheduler choices come from the worker. See the [feature inventory](vulkan-parity.md) for individual commands, extension hooks and limits.

Temporal Lab reads and processes image sequences in the JVM. For **Extract frames from video**, install FFmpeg and put both `ffmpeg.exe` and `ffprobe.exe` on `PATH` before starting the app. The worker uses `ffprobe` to read the frame rate and dimensions, then `ffmpeg` to decode the selected video stream.

## Choose a GPU

The Vulkan worker enumerates adapters when it starts. Select **Settings → Engine → Vulkan GPU** and restart the engine, or set `SDFX_VULKAN_DEVICE` before starting the desktop or MCP server. The environment override is useful for running two copies on different adapters. Use the identifier reported for your machine. For example, this selects the second Vulkan adapter:

```powershell
$env:SDFX_VULKAN_DEVICE = 'Vulkan1'
.\SDV-NUI.exe
```

Vulkan device numbering depends on the installed hardware and driver. On an integrated GPU the model and image buffers share system memory. Start with SD1.5 at 512 × 512 to check the setup. For MCP use `SDV-NUI.exe --mcp --home <settings-directory>`.

## Build

Use a Windows x64 JDK 25, sbt, PowerShell, and the Visual Studio 2022 C++ x64 redistributable files. The packager finds them through VS Installer or `VCToolsRedistDir`. You can also pass `-VulkanVcRedistDir` with the path to `VC/Redist/MSVC/<version>`. From the repository root:

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-25'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
$env:SDFX_VULKAN_ONLY = 'true'
sbt -batch ui/assembly vulkanWorker/assembly
.\scripts\package_windows.ps1 -VulkanOnly -JdkHome $env:JAVA_HOME
```

The packager fetches the Windows Vulkan archive from [stable-diffusion.cpp f296cb0](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1129-f296cb0) and verifies SHA-256 `00f1093279b64a1a4b3a14d8cb7b6fe1920cb8eda31f2fde30e17546fbf8e78d`. For an offline build, pass the downloaded ZIP with `-VulkanArchive`. The packaged DLLs and their upstream license texts are under `vulkan/bin`.
