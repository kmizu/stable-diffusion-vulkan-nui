# Windows x64 Vulkan edition

Extract the entire Vulkan ZIP to a writable directory, then open `SDV-NUI.exe`. In Settings, add a folder containing an SD1.5 or SDXL checkpoint to the checkpoint directories. Select the checkpoint in the main window and generate a txt2img image. The application includes its Java runtimes, inference DLL and Vulkan compute backend.

Install the graphics manufacturer's Vulkan-capable Windows driver. The package includes the Microsoft Visual C++ x64 runtime DLLs used by its native libraries. To exercise the desktop or MCP with a simulated image, add `--dry-run`. That mode runs on the JVM without loading a checkpoint or Vulkan DLL.

## Supported generation

The worker supports SD1.5 and SDXL txt2img, img2img, masked inpainting, hires passes, external VAEs, LoRAs, textual inversion and single-network ControlNet. Sampler and scheduler choices come from the worker. See the [feature inventory](vulkan-parity.md) for individual commands, extension hooks and limits.

When an img2img hook runs on a Vulkan graph, the worker resizes the internal canvas to the dimensions the model requires and restores the requested output size after decoding. Masked img2img composites at the output size using the original image and mask.

For Fooocus Inpaint, put `inpaint_v26.fooocus.patch` and `fooocus_inpaint_head.pth` together in a configured ControlNet model directory. Enable **Fooocus Inpaint** in img2img, select the patch, and supply an image and mask with an SDXL checkpoint. An explicit head path overrides the sibling file. The start and end sliders control the head's denoising window. The patch's weight deltas apply throughout the masked pass.

The worker writes a temporary patched U-Net, about 4.8 GiB for a typical FP16 SDXL checkpoint, once per job. It uses the original checkpoint for text and VAE and removes the temporary U-Net when the job ends. Supply the Fooocus model files separately.

Temporal Lab reads and processes image sequences in the JVM. For **Extract frames from video**, install FFmpeg and put both `ffmpeg.exe` and `ffprobe.exe` on `PATH` before starting the app. The worker uses `ffprobe` to read the frame rate and dimensions, then `ffmpeg` to decode the selected video stream.


## Image tagging, character search and CLIP captions

The Windows package includes the ncnn Vulkan executable. Each optional model runs on the adapter selected in the app.

| Feature | Windows package option | Existing model directory |
| --- | --- | --- |
| WD v3 captions and batch interrogation | `-IncludeVulkanWd` | Set `SDFX_VULKAN_WD_DIR` to the folder containing `wd.ncnn.param`, `wd.ncnn.bin` and `selected_tags.csv`. |
| Character Searcher import | `-IncludeVulkanCharacterModels` | Set `SDFX_VULKAN_CHARA_DIR` to the converted person detector and ISNet files. This option also includes WD v3. |
| BLIP-base captions, standalone or in CLIP Interrogator | `-IncludeVulkanBlip` | Set `SDFX_VULKAN_BLIP_DIR` to the converted BLIP model directory. |

The ncnn build helper checks the source and converted model SHA-256 digests. Character Searcher uses the original OpenAI ViT-B/32 model for similarity searches. It shares the original app’s `SDFX_CHARA_CACHE` directory and creates a lossless vision-only safetensors cache on first use. To import video, provide FFmpeg and ffprobe in PATH or set `SDFX_FFMPEG` and `SDFX_FFPROBE` to the executables.

For BLIP-base, convert the pinned Salesforce checkpoint once on a build machine with CPU PyTorch 2.7.1, Transformers 4.51.3 and ONNX 1.21.0. Use Python 3.10 or newer and the Windows pnnx 20261020 executable. The app reads the resulting ncnn weights and vocabulary.

```powershell
hf download Salesforce/blip-image-captioning-base config.json pytorch_model.bin vocab.txt --revision 82a37760796d32b1411fe092ab5d4e227313294b --local-dir blip-source
python scripts/convert_vulkan_blip.py blip-source path/to/pnnx.exe converted-blip
$env:SDFX_VULKAN_BLIP_DIR = (Resolve-Path converted-blip).Path
```

CLIP Interrogator offers caption, fast and classic prompt generation with BLIP-base. Label retrieval uses the pinned LAION ViT-H/14 image encoder and cached text tables. Best and negative prompts also use LAION’s text encoder to score newly assembled phrases. Set `SDFX_VULKAN_CLIP_TEXT_DIR` to choose its cache directory. The original `model.safetensors` download takes 3.94 GB and the converted text model takes another 1.42 GB. The worker reuses this cache across sessions.


## Choose a GPU

The Vulkan worker enumerates adapters when it starts. Select **Settings → Engine → Vulkan GPU** and restart the engine, or set `SDFX_VULKAN_DEVICE` before starting the desktop or MCP server. The environment override is useful for running two copies on different adapters. Use the identifier reported for your machine. For example, this selects the second Vulkan adapter:

```powershell
$env:SDFX_VULKAN_DEVICE = 'Vulkan1'
.\SDV-NUI.exe
```

Vulkan device numbering depends on the installed hardware and driver. On an integrated GPU the model and image buffers share system memory. Start with SD1.5 at 512 × 512 to check the setup. For MCP use `SDV-NUI.exe --mcp --home <settings-directory>`.

## Build

Use a Windows x64 JDK 25, sbt, PowerShell, and the Visual Studio 2022 C++ x64 Build Tools, CMake, Git and the x64 redistributable files. The packager finds them through VS Installer or `VCToolsRedistDir`. You can also pass `-VulkanVcRedistDir` with the path to `VC/Redist/MSVC/<version>`. From the repository root:

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-25'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
$env:SDFX_VULKAN_ONLY = 'true'
sbt -batch ui/assembly vulkanWorker/assembly
.\scripts\package_windows.ps1 -VulkanOnly -JdkHome $env:JAVA_HOME
```

The packager fetches the Windows Vulkan archive from [stable-diffusion.cpp f296cb0](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1129-f296cb0) and verifies SHA-256 `00f1093279b64a1a4b3a14d8cb7b6fe1920cb8eda31f2fde30e17546fbf8e78d`. To reuse a local archive, pass the downloaded ZIP with `-VulkanArchive`. The packaged DLLs and their upstream license texts are under `vulkan/bin`.

## Face restoration and WD tags

The packager builds a patched CLIP vision loader and the ncnn Vulkan bridge. First builds download the pinned source and SDK. Add `-IncludeVulkanWd` to bundle the WD ViT tagger weights and tag table.

For GFPGAN 1.4, download the [FaceFusion ONNX model](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx) and verify SHA-256 `accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385`. Name it `gfpgan.onnx` and convert with pnnx 20261020:

```powershell
pnnx.exe gfpgan.onnx 'inputshape=[1,3,512,512]' 'fp16=0'
```

Place `gfpgan.ncnn.bin` and `gfpgan.ncnn.param` together in a configured face restoration model directory. Select the BIN in Restore faces or Restore faces (extras). The JVM detects frontal faces and blends restored crops at the selected visibility.

## Output safety filters

The Windows package includes the NudeNet detector under `vulkan/bin/nsfw`. For the optional Stable Diffusion Safety Checker and NSFWJS, prepare the model files with [the converter](../scripts/build_vulkan_safety.py).

```powershell
python scripts/build_vulkan_safety.py --safety C:/models/safety_checker.bin --nsfwjs C:/models/nsfwjs.onnx --pnnx C:/tools/pnnx.exe --output C:/SDV-NUI/vulkan/bin/nsfw
```

The converter verifies the source hashes before preparing the files. The installed app uses the converted files without Python. To store the models elsewhere, set `SDFX_VULKAN_NSFW_DIR` to a directory containing the NudeNet and optional model files.

## SD-CN-Animation optical flow

The video-to-video plugin uses torchvision RAFT Small C_T_V2 or Large C_T_SKHT_V2. Install the encoder and update weights for the chosen model alongside the packaged ncnn bridge. A separate build machine with PyTorch, torchvision and pnnx can prepare both variants. The following command was tested with PyTorch 2.1.2, torchvision 0.16.2 and pnnx 20261020.

```powershell
python scripts/build_vulkan_raft.py --pnnx C:/tools/pnnx.exe --output C:/SDV-NUI/vulkan/bin
```

The converter writes four ncnn files under each of `raft-small` and `raft-large`. To keep the converted files elsewhere, point `SDFX_VULKAN_RAFT_DIR` at their parent directory. For a custom torchvision state dictionary, add `--kind small` or `--kind large` and `--weights path/to/model.pth`. Only the conversion step uses Python. Frame decoding and encoding use FFmpeg and ffprobe on PATH. The selected Vulkan adapter runs the learned encoders and recurrent update, and the JVM computes correlations, flow upsampling, warping and occlusion blending.

For CodeFormer, download the [models-3.0.0 CodeFormer ONNX export](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx). The conversion helper requires Python with NumPy and ONNX on the conversion machine. Copy its output files into a face restoration model directory and select `codeformer.ncnn.bin` in the app.

```powershell
python scripts/convert_vulkan_codeformer.py codeformer.onnx models/face_restore --pnnx C:/tools/pnnx.exe
```

The CodeFormer weight slider controls the model's fidelity input. On Radeon 780M, the converted graph at weight 0.5 matched the reference ONNX portrait within 0.04 levels per 8-bit channel before rounding.

## Local custom scripts

The optional `-IncludeVulkanCustomCode` package profile includes a private JVM-based GraalPy interpreter for the Custom code plugin. Enable **Allow custom code** in Settings and restart the worker to run locally authored scripts. The standard diffusion worker uses Java and Vulkan independently of this option.

Use `print` for the output log, `result` for text, and `images` for paths to pass back to the desktop. For example:

```python
images = pipe("watercolor fox", width=512, height=512, num_inference_steps=12).images
result = str(pipe)
```

`engine.device` and `engine.model` describe the active worker. The callable `pipe` forwards generation parameters to Vulkan and returns image paths. Scripts that directly access PyTorch tensors or diffusers pipeline internals require adaptation.
