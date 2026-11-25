# Vulkan protocol and extension coverage

The Vulkan worker accepts JSON Lines protocol v1 requests. Its `hello.features` describes the running worker. The inventories below compare its handlers with `python/sdfx/server.py::QUEUED_TYPES` and `python/sdfx/hooks.py::KNOWN_HOOKS`. Some plugins run entirely in the desktop application.

## Commands

| Command | Implemented scope |
| --- | --- |
| `list_files` | Enumerates VAE, LoRA, embedding, ControlNet, upscaler, face-restoration, SVD, Zero123 and PhotoMaker files. Recognized model directories count as single entries. Discovery does not test a file's compatibility with inference. |
| `preprocess` | Processes none, inpaint, invert, Canny, luminance/intensity recolor, tile resample, tile colorfix and Gaussian blur. Standalone mode writes the processing parameters and preserves source infotext. |
| `image_tool` | CAS upscale, loopback scaler, color correction, color enhancement and pixel art. |
| `inpaint_difference_preview` | Computes the difference mask and source/effective mask overlay with morphology, blur and inversion controls. |
| `upscale` | Nearest and Lanczos image resizing. Selected RGB upscaler weights use the native upscaler API on the selected adapter. RealESRGAN x4plus anime 6B produced a 512 x 512 output from a 256 x 256 input on the RTX 3090. |
| `tokenize` | Token inspection for the loaded SD1.5 or SDXL model. |
| `merge_models`, `convert_model` | JVM safetensors checkpoint merge and conversion. |
| `model_keyword`, `model_keyword_plan` | Check, save or delete mappings, explicitly install the upstream LoRA catalog, and preview batched prompt and adapter plans. |
| `metadata_utils` | Reads and edits safetensors metadata in a new file. Validates source identity and payload hash before publication. |
| `list_models`, `list_samplers`, `load_model`, `set_vae`, `unload`, `txt2img`, `img2img`, `system_info` | Model management, diffusion and diagnostics. See the [Windows Vulkan guide](WINDOWS_VULKAN.md) for supported settings and model families. |

The model loader supports external VAEs, checkpoint dtype, RAM or disk offload, FP8 diffusion-weight storage and VAE tiling. Generation can apply per-request LoRAs, named textual-inversion embeddings and one full-range balanced ControlNet. SD1.5 and SDXL ran each conditioning path on RTX 3090. On Radeon 780M, SD1.5 Canny in Diffusers and CompVis formats loaded all 340 tensors and produced identical pixels.

Direct control commands include `ping`, `shutdown`, `interrupt`, `interruption_protection`, `restart_authorization`, `system_status` and `list_devices`.

Img2img supports crop and fill resize modes, mask inversion, blur, masked-area fill, and full-resolution inpainting with a final composite on the source canvas. SD1.5 and SDXL tests on Radeon 780M preserved pixels outside the mask. Hires supports pixel or latent upscaling with independent refinement prompts, CFG, sampler, scheduler and LoRAs. The latent refinement graph currently requires generation without ControlNet, textual inversion or other sampling hooks.

| Command | Implemented scope |
| --- | --- |
| `memory`, `system_benchmark`, `vram_estimator` | Adapter memory, timed real generation and optional VRAM calibration. Generation and calibration ran on Radeon 780M. |
| `system_submission_preview`, `system_submit` | Build a system report preview and handle explicit submissions. |
| `embedding_inspector`, `embedding_editor`, `embedding_merge` | Look up real checkpoint token vectors, inspect cosine neighbors, preview edits, save with backup and stale-file checks, or export merge expressions as safetensors. Radeon 780M tests on SD1.5 and SDXL reproduced the exported embedding image exactly from the inline merge expression. A draft changed the image and clearing it restored every pixel. |
| `vae_tools` | Full VAE and TAESD encode/decode run on the selected Vulkan adapter. Approx cheap and the eight-layer A1111 Approx NN decoder run on the JVM. Supported latent containers include lossless L PNG, RGBA PNG, CMYK TIFF, CMYK JPEG100 and raw safetensors. SD1.5 and SDXL passed real encode/decode checks, including on Radeon 780M. |
| `animatediff_encode` | Assemble captured frames as GIF, MP4, WebP or WebM previews according to the worker's supported formats. |
| `clip_interrogator` | Encode images on the selected Vulkan adapter using OpenCLIP ViT-H/14, then rank published label vectors on the JVM. Analyze returns five labels per category. Prompt returns the closest single label. The first run fetches 2.76 GB of SHA-256-pinned resources. Set `SDFX_VULKAN_CLIP_DIR` to use an existing cache. The 780M matched independent cosine rankings for all 25 labels. BLIP captions and batch interrogation remain unavailable. |
| `controlnet_fastload_data` | Read and write standalone CNI and image trailers, transfer ControlNet settings and NumPy images/masks, preview native inputs, choose per-job load/save priority, and browse or filter an image gallery. The bounded JVM parser interprets tagged data without invoking serialized globals. Eight upstream protocol 2-5 fixtures (legacy and Pydantic units), the upstream CNI and embedded PNG matched the Python reader. The Python reader also decoded worker exports. A per-job Fastload import on Radeon 780M reproduced every pixel of the directly configured ControlNet run for SD1.5 at 448 x 448 and SDXL at 640 x 640 (eight Euler steps each). Protected sessions can prepare inputs in memory. Readable trailer export requires an unprotected session. |
| `temporal_lab` | Preview a motion mask, propagate stable regions between reference/generated frames (DFI), deflicker a sequence, fuse style frames, or extract frames from a video using FFmpeg and ffprobe on PATH. Against the Python/OpenCV implementation, a four-frame fixture matched exactly for preview, deflicker, fusion and extraction. DFI differed by at most one intensity level per channel. |

Remaining Python commands include `aesthetic_embedding`, `batch_interrogate_execute`, `batch_interrogate_prepare`, `caption`, `chara_searcher`, `cn_animation`, `img2vid`, `novel_view`, `restore_faces` and `run_code`.

## Generation hooks

| Hook | Implemented scope |
| --- | --- |
| `soft_inpainting` | Euler and Euler a sampling blend each step with the re-noised original through a continuous mask, then composite by mask and pixel difference. Paired mask settings changed the output on SD1.5 and SDXL. The 768-pixel SDXL fixture preserved the fox and redrew the flowers at the high end of the mask. |
| `hires_fix_tweaks` | Repeat the refinement pass with independent seeds, subseeds, variation strength and centered seed-resize noise. Effective settings travel with each result. Real pixel and latent Hires jobs reproduced the requested seed and repeat order on SD1.5 and SDXL. Each repeat recomputes the deterministic first pass. |
| `latent_upscale` | Encode the source at its own resolution, resize its four-channel latent, then run the requested number of denoising steps. Supports seven interpolation choices including adaptive area and antialiased bilinear/bicubic. SD1.5 and SDXL ran masked area interpolation on Radeon 780M. |
| `inpaint_difference`, `img2img_hires_fix` | Compute a source difference mask for inpainting, or upscale an img2img result and optionally run a second diffusion pass. |
| `pnginfo_injection`, `model_keyword` | Record checkpoint, external VAE, LoRA, ControlNet and embedding identities. Match checkpoint and adapter keywords and advance batch counters after success. |
| `model_sampling`, `neveroom`, `tiling` | Override epsilon/v-prediction, move diffusion weights to CPU, enable VAE tiling, or select circular x/y padding for an entire pass. Zero-terminal SNR and scheduled tiling are unavailable. |
| `apg`, `cfg_schedule` | Adaptive projected guidance with fixed momentum and per-step CFG scale. Adaptive momentum schedules and sigma limiters are unavailable. |
| `save_intermediate` | Capture denoised, noisy or preview frames by pass and step, optional final frame, PNG metadata, manifest and optional video or encoding recipe. SD1.5 and SDXL generated frames at the requested interval on RTX 3090. |
| `steps_animation` | Capture denoised steps, set start, duration, FPS, interpolation, codec and scoped frame/video directories. Supports capture-only, cleanup after successful export and bounded partial export on interruption. RTX 3090 tests captured four steps from each family and produced a six-frame, one-second H.264 video. |
| `latent_mirroring`, `resharpen`, `anti_burn`, `color_grading` | Euler or Euler a graphs transform the sampled latent directly. Mirror, rotate, extrapolate a step delta, average the final steps, or recenter/normalize color channels. Tested together on both SD1.5 and SDXL. |
| `cfg_cache`, `rescale_cfg`, `vectorscope_cc` | Euler can cache positive-minus-negative model output for fixed intervals and rescale CFG predictions from their measured variance. Adaptive cosine thresholds and high-frequency boosting are unavailable. Vectorscope adjusts brightness, RGB channels, contrast and saturation in latent space with its own strength curve. Both families completed the combined Euler graph on RTX 3090. |
| `soft_inpainting` | Euler and Euler a sampling blend each step with the re-noised original through a continuous mask, then composite by mask and pixel difference. Paired mask settings changed the output on SD1.5 and SDXL. The 768-pixel SDXL fixture preserved the fox and redrew the flowers at the high end of the mask. |
| `hires_fix_tweaks` | Repeat the refinement pass with independent seeds, subseeds, variation strength and centered seed-resize noise. Effective settings travel with each result. Real pixel and latent Hires jobs reproduced the requested seed and repeat order on SD1.5 and SDXL. Each repeat recomputes the deterministic first pass. |
| `latent_upscale` | Encode img2img inputs at source resolution and resize the latents before diffusion. |
| `gpu_temperature_guard` | Read the configured NVIDIA, ROCm or OpenHardwareMonitor sensor between sampling steps and pause until cooldown or timeout. Real RTX 3090 readings triggered pauses and maximum-wait recovery during SD1.5 and SDXL tests. Custom DGL graphs currently omit step polling. |
| `mahiro`, `detail_daemon`, `cads`, `postdiff` | Mahiro adjusts the denoised CFG result using positive and negative branch predictions. Detail Daemon changes the training timestep and model-input scale per branch. CADS anneals noise on actual text embeddings. PostDiff enlarges the early clean estimate, re-noises at full resolution and continues Euler sampling. Both model families completed separate real-model checks for each hook. PostDiff currently supports Euler txt2img. |
| `embedding_merge` | Compose weighted embedding vectors for generation. |
| `animatediff`, `photomaker` | Single-context SD1.5 motion generation and SDXL PhotoMaker identity conditioning. Radeon 780M produced three distinct motion frames and completed PhotoMaker sampling. |
| `ip_adapter` | Encode the reference through CLIP ViT-H, project the image tokens and inject them into cross-attention. Both classic SD1.5 and SDXL ViT-H adapters generated images on RTX 3090. At 768 x 768 and 12 Euler steps the SDXL reference changed the fox composition and palette without the banding seen at 384 x 384 and five steps. Plus follows the patch-token graph and awaits a model test. The pinned native vision loader requires ViT-H, so bigG SDXL adapters and standard Revision are unavailable. |

Further Python generation hooks without a corresponding verified implementation in this inventory include `adetailer`, `aesthetic_gradients`, `aesthetic_score`, `agent_attention`, `batch_face_swap`, `cd_tuner`, `characteristic_guidance`, `clip_io`, `compile`, `composable_lora`, `conditioning_highres_fix`, `controllllite`, `deepcache`, `dynamic_thresholding`, `face_restore`, `fooocus_inpaint`, `freeu`, `hypertile`, `img2imgalt`, `incantations`, `kohya_hrfix`, `latent_modifier`, `llul`, `multidiffusion`, `negpip`, `neutral_prompt`, `nsfw_checker`, `pag`, `prompt_fusion`, `reference`, `revision`, `sag`, `semantic_guidance`, `sonar`, `tgate`, `token_downsampling`, `weight_gradient`.

### Boundaries of the port

The pinned Vulkan library exposes complete diffusion graphs, vision, VAE, upscaler and motion components. Its ViT-H vision runner rejects the larger bigG encoder used by standard Revision. Model-internal attention and activation hooks such as FreeU, PAG and SAG need their own graph changes. Captioning, face restoration, character search and novel-view synthesis require separate inference models. The entries above identify which operations currently have a worker implementation.

The per-step latent graphs currently support Euler and Euler a. The two prediction-level CFG hooks require Euler. Custom graphs reject unsupported combinations of ControlNet, embeddings, high-resolution refinement and other sampling hooks. Low-resolution five-step tests establish that the model, sampler and VAE paths run. Some strongly graded or averaged results were blurry. The 768 x 768 IP-Adapter comparison used twelve steps and was inspected alongside a same-seed baseline. Fifteen-step 768 x 768 SDXL comparisons also produced recognizable fox images with default Detail Daemon and PostDiff, including the expected composition change from the lower-resolution first stage.

## JVM implementation provenance

Pixel processing, file discovery and safetensors metadata utilities under `vulkanWorker/src/main/scala/sdfx/vulkan/tools` adapt the model-independent implementations at [kmizu/scala-torch 3c3b415](https://github.com/kmizu/scala-torch/tree/3c3b415cad58bc63e1ae7c37c03ab21183068b9f). Their BSD-3-Clause notice is included in the worker JAR at `META-INF/LICENSE-scala-torch.txt`. These commands use Java ImageIO and local files. The neural upscaler uses stable-diffusion.cpp through the Vulkan worker's FFM binding.
