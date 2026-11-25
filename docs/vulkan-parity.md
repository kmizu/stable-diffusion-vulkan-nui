# Vulkan protocol and extension coverage

The Vulkan worker accepts JSON Lines protocol v1 requests. Its `hello.features` describes the running worker. The inventories below compare its handlers with `python/sdfx/server.py::QUEUED_TYPES` and `python/sdfx/hooks.py::KNOWN_HOOKS`. Some plugins run entirely in the desktop application.

## Commands

| Command | Implemented scope |
| --- | --- |
| `list_files` | Enumerates VAE, LoRA, embedding, ControlNet, upscaler, face-restoration, SVD, Zero123 and PhotoMaker files. Recognized model directories count as single entries. Discovery does not test a file's compatibility with inference. |
| `preprocess` | Processes none, inpaint, invert, Canny, luminance/intensity recolor, tile resample, tile colorfix and Gaussian blur. Standalone mode writes the processing parameters and preserves source infotext. |
| `image_tool` | CAS upscale, loopback scaler, color correction, color enhancement and pixel art. |
| `inpaint_difference_preview` | Computes the difference mask and source/effective mask overlay with morphology, blur and inversion controls. |
| `upscale` | Nearest and Lanczos image resizing. Selected compatible RGB upscaler weights use the native upscaler API on the selected Vulkan adapter, then resize to the requested dimensions. |
| `tokenize` | Token inspection for the loaded SD1.5 or SDXL model. |
| `merge_models`, `convert_model` | JVM safetensors checkpoint merge and conversion. |
| `model_keyword`, `model_keyword_plan` | Check, save or delete mappings, explicitly install the upstream LoRA catalog, and preview batched prompt and adapter plans. |
| `metadata_utils` | Reads and edits safetensors metadata in a new file. Validates source identity and payload hash before publication. |
| `list_models`, `list_samplers`, `load_model`, `set_vae`, `unload`, `txt2img`, `img2img`, `system_info` | Model management, diffusion and diagnostics. See the [Windows Vulkan guide](WINDOWS_VULKAN.md) for supported settings and model families. |

Direct control commands include `ping`, `shutdown`, `interrupt`, `interruption_protection`, `restart_authorization`, `system_status` and `list_devices`.

| Command | Implemented scope |
| --- | --- |
| `memory`, `system_benchmark`, `vram_estimator` | Adapter memory, timed real generation and optional VRAM calibration. Generation and calibration ran on Radeon 780M. |
| `system_submission_preview`, `system_submit` | Build a system report preview and handle explicit submissions. |
| `embedding_inspector`, `embedding_editor`, `embedding_merge` | Inspect embeddings and edit or merge vectors in supported formats. |
| `vae_tools` | Encode and decode with the selected VAE. Both SD1.5 and SDXL passed the worker/UI VAE contract checks. |
| `animatediff_encode` | Assemble captured frames as GIF, MP4, WebP or WebM previews according to the worker's supported formats. |
| `temporal_lab` | Video extraction, frame deflicker, style fusion and frame interpolation. |

Remaining Python commands include `aesthetic_embedding`, `batch_interrogate_execute`, `batch_interrogate_prepare`, `caption`, `chara_searcher`, `clip_interrogator`, `cn_animation`, `controlnet_fastload_data`, `img2vid`, `novel_view`, `restore_faces` and `run_code`.

## Generation hooks

| Hook | Implemented scope |
| --- | --- |
| `inpaint_difference`, `img2img_hires_fix` | Compute a source difference mask for inpainting, or upscale an img2img result and optionally run a second diffusion pass. |
| `pnginfo_injection`, `model_keyword` | Record checkpoint, external VAE, LoRA, ControlNet and embedding identities. Match checkpoint and adapter keywords and advance batch counters after success. |
| `model_sampling`, `neveroom`, `tiling` | Override epsilon/v-prediction, move diffusion weights to CPU, enable VAE tiling, or select circular x/y padding for an entire pass. Zero-terminal SNR and scheduled tiling are unavailable. |
| `apg`, `cfg_schedule` | Adaptive projected guidance with fixed momentum and per-step CFG scale. Adaptive momentum schedules and sigma limiters are unavailable. |
| `save_intermediate` | Capture denoised, noisy or preview frames by pass and step, optional final frame, PNG metadata, manifest and optional video or encoding recipe. SD1.5 and SDXL generated frames at the requested interval on RTX 3090. |
| `embedding_merge` | Compose weighted embedding vectors for generation. |
| `animatediff`, `photomaker` | Single-context SD1.5 motion generation and SDXL PhotoMaker identity conditioning. Radeon 780M produced three distinct motion frames and completed PhotoMaker sampling. |
| `ip_adapter` | Encode the reference through CLIP ViT-H, project the image tokens and inject them into cross-attention. Both classic SD1.5 and SDXL ViT-H adapters generated images on RTX 3090. Plus follows the patch-token graph and awaits a model test. The pinned native vision loader requires ViT-H, so bigG SDXL adapters and standard Revision are unavailable. |

Further Python generation hooks without a corresponding verified implementation in this inventory include `adetailer`, `aesthetic_gradients`, `aesthetic_score`, `agent_attention`, `anti_burn`, `batch_face_swap`, `cads`, `cd_tuner`, `cfg_cache`, `characteristic_guidance`, `clip_io`, `color_grading`, `compile`, `composable_lora`, `conditioning_highres_fix`, `controllllite`, `deepcache`, `detail_daemon`, `dynamic_thresholding`, `face_restore`, `fooocus_inpaint`, `freeu`, `gpu_temperature_guard`, `hires_fix_tweaks`, `hypertile`, `img2imgalt`, `incantations`, `kohya_hrfix`, `latent_mirroring`, `latent_modifier`, `latent_upscale`, `llul`, `mahiro`, `multidiffusion`, `negpip`, `neutral_prompt`, `nsfw_checker`, `pag`, `postdiff`, `prompt_fusion`, `reference`, `rescale_cfg`, `resharpen`, `revision`, `sag`, `semantic_guidance`, `soft_inpainting`, `sonar`, `steps_animation`, `tgate`, `token_downsampling`, `vectorscope_cc`, `weight_gradient`.

## JVM implementation provenance

Pixel processing, file discovery and safetensors metadata utilities under `vulkanWorker/src/main/scala/sdfx/vulkan/tools` adapt the model-independent implementations at [kmizu/scala-torch 3c3b415](https://github.com/kmizu/scala-torch/tree/3c3b415cad58bc63e1ae7c37c03ab21183068b9f). Their BSD-3-Clause notice is included in the worker JAR at `META-INF/LICENSE-scala-torch.txt`. These commands use Java ImageIO and local files. The neural upscaler uses stable-diffusion.cpp through the Vulkan worker's FFM binding.
