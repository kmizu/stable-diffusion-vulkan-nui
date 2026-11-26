# Windows Vulkan inference verification — 2026-11-26




## Windows v0.64 — Aesthetic Gradients with LoRA

An independent CPU PyTorch check formed a rank-two Q-projection delta and applied the fixture's alpha and adapter weight before three Adam updates. On Radeon 780M, maximum absolute loss differences were 0.000357 for SD1.5 and 0.000456 for SDXL. On RTX 3090 the SDXL difference was 0.000084 at the first step and 0.003500 at the third. The same merged weights feed the Aesthetic style projection before training.

Masked img2img compared no LoRA with a real fixed-weight adapter at 0.65 using the same seed. Radeon 780M used SD1.5 and `20_yui`, and RTX 3090 used SDXL and `alraune_v1_illustrious_b-000007`. Both changed the masked area and preserved every unmasked RGB channel. The worker logged 216 and 198 applied CLIP-L adapter tensors respectively. The input mask had zero blur and left visible seams in the comparison images.

The staged `SDV-NUI.exe` repeated masked LoRA generation through MCP on both adapters with `cuda=false`. The run logs showed LoRA deltas applied before training and a trained CLIP-L overlay loaded for inference. Both PNGs retained every unmasked pixel and recorded the aesthetic embedding fingerprint. The images were inspected.

All 69 Vulkan worker tests and 14 aesthetic plugin tests passed. The ZIP SHA-256 is `2652557ee2c19350e9c81958e64849c08bc4a6ab9cd3a09d4b5977eadf3b72d9`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.63 — Aesthetic Gradients with ControlNet and Hires

Matched-seed masked img2img ran with a balanced Canny unit on Radeon 780M using SD1.5 at 512 × 512 and RTX 3090 using SDXL at 768 × 768. Removing the unit or changing the Aesthetic blend weight from 0.85 to 0.01 changed the generated region on both adapters. All three comparisons on each adapter preserved every unmasked RGB channel.

RTX 3090 also ran SD1.5 from 512 × 512 to 576 × 576 through built-in Latent Hires and through the separate Img2Img Hires Fix. Both used Canny and different prompts for the two passes. The Latent Hires result preserved the Lanczos-resized source wherever the resized mask was zero. Radeon 780M ran built-in Lanczos Hires with the same prompt in both passes and reused one trained overlay. The separate refinement operates on the resized result of the base pass. The short test images were inspected. Zero mask blur left visible edges on the masked images.

The staged `SDV-NUI.exe` ran the separate Img2Img Hires Fix through MCP on Radeon 780M with SD1.5 at 576 × 576 and on RTX 3090 with SDXL at 832 × 832. Both used Canny and a different refinement prompt. The runtime reported `cuda=false`, loaded two distinct CLIP-L overlays and wrote the embedding fingerprint and refinement settings to the PNG. The short packaged outputs retained a recognizable subject with visible texture artifacts.

All 69 Vulkan worker tests and 15 aesthetic and Img2Img Hires plugin tests passed. The ZIP SHA-256 is `c6fa30988ce15b6c3c5bbd36169562b202c4459a2393095e3015dd5779b3fd0f`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.62 — Aesthetic text interpolation

Both Vulkan adapters projected the first style chunk and trained CLIP-L for three Adam steps with an interpolated target. Independent CPU PyTorch used the same checkpoint tensors, tokens and fractions. At fraction 0.35 with SD1.5, the maximum absolute loss differences were 0.000128 on Radeon 780M and 0.000249 on RTX 3090. At fraction 1.0 with SDXL, they were 0.000162 and 0.002650 respectively. The largest difference came at the first SDXL step on RTX 3090. The v0.61 inference DLL returned an explicit unsupported error for a nonzero style fraction when paired with the new worker.

The staged `SDV-NUI.exe` generated masked SD1.5 img2img on Radeon 780M and masked SDXL img2img on RTX 3090 through MCP with `cuda=false`. The PNGs recorded the style text, fraction and embedding fingerprint. Both retained every RGB channel outside the mask. The images were inspected. The 780M image showed a visible edge at the zero-blur test mask.

All 68 Vulkan worker tests and 14 aesthetic plugin tests passed. The ZIP SHA-256 is `4b91060da873aeb195471e4ea6255ae517fa34785d80187d0a67491b548fd1ba`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.61 — Aesthetic Gradients in img2img

An isolated full CLIP-L training run updated 196 parameter tensors on Vulkan with each adapter. Three Adam iterations moved the SD1.5 loss from −0.245517 to −0.560078 on RTX 3090 and the SDXL loss from −0.231040 to −0.381057 on Radeon 780M. The maximum absolute difference from independent CPU PyTorch was 0.000731 for SD1.5 and 0.000571 for SDXL.

Direct masked jobs produced different pixels inside the mask at blend weights 0.25 and 1.0. All pixels outside the mask matched the source. The staged `SDV-NUI.exe` used the trained CLIP-L overlay and wrote the embedding fingerprint to PNG infotext in masked SDXL on RTX 3090 and masked SD1.5 on Radeon 780M. Both selected Vulkan with `cuda=false`. An incorrect fingerprint was rejected, and interrupting a 100-step training job after its first progress event cancelled the native call.

A separate unmasked SDXL img2img job on RTX 3090 used positive personalization with an empty negative prompt. The masked and unmasked outputs were inspected. The short unmasked run rendered a recognizable fox with strong saturation and visible texture artifacts.

All 67 Vulkan worker tests and 14 aesthetic plugin tests passed. The ZIP SHA-256 is `d01c3369d02b9ae9119a4b16bdeb85d94a0a2bd6e8c7a4e1ce285ae8ebef3115`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.60 — Composable LoRA layer counts

The staged `SDV-NUI.exe` ran masked img2img through MCP with `cuda=false` on RTX 3090 with SDXL and Radeon 780M with SD1.5. Active LoRA evaluations counted 788 layers and 192 layers respectively, matching independent counts of the adapters' U-Net tensor groups. Branches with zero weight reported zero layer evaluations. PNGs and plots were inspected, and the generated images matched the v0.59 runs pixel for pixel.

A 512 × 512 SDXL Heun run on RTX 3090 counted 1,576 layer evaluations for a branch when the solver evaluated it twice in one step. A synthetic SD1.5 adapter with both weight and bias deltas for a single GroupNorm layer counted one application per branch and preserved the same image pixels before and after the counting fix. All 65 Vulkan worker tests and the Composable LoRA plugin test passed.

The ZIP SHA-256 is `8c881f5323f781248a74ee7a0690f006585cb0c5326df2ca134e511d31b0e774`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.59 — Numeric Composable LoRA commands

The staged `SDV-NUI.exe` ran masked img2img over MCP at 768 × 768 with SDXL on RTX 3090 and at 512 × 512 with SD1.5 on Radeon 780M. Both reported `cuda=false`. A numeric `#cmd` selected weight 0.45 for the first two steps and 0.225 for the last three. Both traces reported zero for the other branches. The native log showed one runtime U-Net adapter initialization per run. Every pixel outside each mask matched the source.

The shared and separate U-Net paths produced identical pixels in a fixed-seed 512 × 512 SDXL comparison on RTX 3090. A separate 768 × 768 run used nine distinct nonzero weights with one runtime U-Net adapter initialization. At 576 × 576, a Latent Hires test checked fresh command clocks and weights for both the positive and the negative branches. The images and plots were inspected.

All 65 Vulkan worker tests and the Composable LoRA plugin test passed. The Clip I/O real SD1.5 regression produced the same pixels as its independent Torch conditioning fixture.

The ZIP SHA-256 is `ed199229417e43278ed156dc35f80cf061a1607be917aee3d7d8c2d5361772a6`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.58 — Composable LoRA weight plots

The staged `SDV-NUI.exe` returned the generated image and a 1200 × 680 plot PNG through MCP for masked SDXL img2img at 768 × 768 on RTX 3090 and SD1.5 img2img at 512 × 512 on Radeon 780M. Both reported `cuda=false` and preserved every source pixel outside the mask. Each trace recorded a U-Net adapter weight of 0.45 in the selected branch at steps 1 and 2, zero thereafter, and zero in the other evaluated branches. The short runs showed distortions inside the masked area.

Direct SDXL on RTX 3090 verified that an adapter window beyond the last step reported zero weight for each evaluation. Two 576 × 576 Latent Hires runs verified independent base and refinement clocks at CFG 1 and CFG 6. CFG 1 skipped the negative branch. The CFG 6 run used 12 base steps and 8 Hires steps and produced a recognizable watercolor fox. The trace reports the count of model evaluations per row.

All 60 Vulkan worker tests, 46 engine tests and the Composable LoRA plugin test passed. One engine test was skipped. The ZIP SHA-256 is `196566862116ce432c7370ac9f80f7bc5fa3ab59463778d7384a3d03a5f371bc`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.57 — Composable LoRA step controls

Direct masked SDXL img2img at 768 × 768 on RTX 3090 and SD1.5 at 512 × 512 on Radeon 780M produced different edited pixels for early and late LoRA windows. A window beyond the pass matched the branch without an adapter pixel for pixel on both devices. RTX 3090 reproduced the early-window seed exactly. These cases retained all source pixels outside the mask, and the images were inspected.

On RTX 3090, a separate 512 × 512 SDXL run exercised an increasing ramp and alternating an adapter with an empty slot. The runs differed inside the mask and preserved its exterior. A 576 × 576 Latent Hires pass also used a separate timed branch assignment. The short SDXL cases showed visible distortions inside the edit. An earlier 768 × 768 ramp run reached the host-memory guard while packaging was active.

All 59 Vulkan worker tests and the Composable LoRA plugin test passed. The tests cover inclusive fractional endpoints, repeated adapter weights for text encoding, alternating choices, the Hires offset and the legacy img2img step denominator.

The staged `SDV-NUI.exe` ran both masked jobs through MCP with the default Model Keyword configuration, a system-only PATH and `cuda=false`. RTX 3090 used DPM++ 2M and Radeon 780M used Euler. The PNGs retained the scheduled LoRA bracket and every unmasked pixel. The ZIP SHA-256 is `1e1b4bef8476c3d4f06a726efde7da9de440928fb462655f0859888c4fe1d465`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staging files.

## Windows v0.56 — Static Composable LoRA

RTX 3090 tested masked SDXL at 768 × 768 with a U-Net-only SafeTensors adapter. Radeon 780M tested masked SD1.5 at 512 × 512 with an adapter covering both text and U-Net. Moving the same LoRA between two AND branches changed the edited region on both GPUs. Returning to the original branch reproduced the same seed pixel for pixel. All unmasked source pixels matched.

RTX 3090 also tested the negative adapter switches and a separate branch assignment for the 832 × 832 Latent Hires pass. A two-adapter job applied standard LoRA to one branch and LoKr to another, logged both sets of matched tensors and kept the unmasked image pixel-exact. The images were inspected, and all 57 Vulkan worker tests and the Composable LoRA plugin test passed.

The staged `SDV-NUI.exe` ran both masked jobs through MCP with the default Model Keyword configuration, a system-only PATH and `cuda=false`. RTX 3090 used DPM++ 2M and Radeon 780M used Euler. Both PNGs preserved the original branch tag and settings, and every unmasked source pixel matched. The native log confirmed application of the selected adapter at runtime.

The ZIP SHA-256 is `be011e3f95811190806710e4250e10dac25936ec68153e3db790ef4ea6cd8848`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the tested staged files byte for byte.

## Windows v0.55 — Characteristic Guidance

Direct masked SDXL img2img on RTX 3090 at 768 × 768 and SD1.5 on Radeon 780M at 512 × 512 exercised the native characteristic solver. The corrected output differed inside the edit, and both adapters retained every unmasked pixel. The one-iteration fallback matched ordinary img2img pixel for pixel. RTX 3090 also reproduced the corrected seed exactly, matched the disabled job at CFG scale 1, matched the fallback with an empty negative prompt and exercised three quantile bands per latent channel.

RTX 3090 exercised 832 × 832 Latent Hires, 576 × 576 Lanczos Hires and a two-evaluation window with reuse. The Lanczos path keeps built-in resizing in FP32. The complete Lanczos fallback image still differed at the pixel level from the ordinary end-to-end Hires route. The four-step, CFG 10 stress cases showed color artifacts with ordinary and corrected sampling. All 52 Vulkan worker tests and four Characteristic Guidance plugin tests passed. The images were inspected. [Characteristic Guidance paper](https://proceedings.mlr.press/v235/zheng24f.html).

A separate eight-step SDXL run at CFG 6 converged in five iterations and retained the recognizable fox and meadow in both outputs. The two masked images differed inside the edit and matched outside it.

The staged `SDV-NUI.exe` ran both masked models over MCP with a system-only PATH and `cuda=false`. The first evaluation converged in six iterations on RTX 3090 and five on Radeon 780M. Both PNGs recorded the CHG parameters and preserved unmasked RGB. The ZIP SHA-256 is `1cfe531dcddfa63af25fbbd67ed61da12832febef949c22f8d38425bf1eb4a16`. The archived launcher, application JAR, worker JAR and inference DLL matched the staged files.

## Windows v0.54 — Graph image dimensions

Vulkan graph img2img now prepares the input image, generation mask and conditioning images at the model's internal canvas size. The worker restores the requested dimensions before compositing with the original mask. Latent Hires uses an internally aligned target as well.

Direct LLuL tests on RTX 3090 with SDXL returned 800 × 776 for plain and masked img2img and 864 × 808 for Latent Hires. Radeon 780M with SD1.5 returned 544 × 520 and 608 × 552 respectively. Both masked runs retained every source RGB channel outside the mask. The images were inspected, and all 49 Vulkan worker tests passed.

The staged `SDV-NUI.exe` repeated both masked jobs through MCP on a system-only PATH with `cuda=false`. The ZIP SHA-256 is `af52a082c688e19e9bceea5d83b8fe47e7000651c4da531a110eab6f0077ca3b`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the verified staged files byte for byte.

## Windows v0.53 — LLuL

The staged `SDV-NUI.exe` ran the LLuL output-layer hook through MCP `img2img` on Radeon 780M with SD1.5 at 512 × 512 and on RTX 3090 with SDXL at 768 × 768. Both runs used weight 0.15 and a system-only PATH, reported `cuda=false`, recorded LLuL settings in the PNG and preserved every RGB channel outside the img2img mask.

The ZIP SHA-256 is `34d1ae2bda957c7d8aa131b9161ae18840e993fab005325bba8d872600421ebf`. The archived launcher, desktop JAR, worker JAR and inference DLL matched the staged files byte for byte.

Direct tests on RTX 3090 exercised the region coordinates, spherical blending, bicubic antialiasing and the four advanced layer types. The `OUT03` filter ran ten region operations per active evaluation. Zero weight, an unreachable start and an unmatched filter preserved the baseline. A black LLuL blend mask matched the graph with no active layer, and the same-seed replay matched exactly. A seven-step sliced Heun run applied LLuL on evaluation eight with automatic Max Steps. A 576 × 512 SD1.5 test distinguished area from fixed average pooling in the U-Net's odd-width middle blocks.

An 832 × 832 SDXL Latent Hires run on RTX 3090 applied LLuL in both sampling passes. An earlier concurrent run reached the runtime's host-memory guard at the second pass. All 48 Vulkan worker tests and seven LLuL plugin tests passed. The images were inspected. Strong output-layer blending at weight 1 produced color artifacts in the selected region on both adapters. Two direct SDXL requests at 776 × 768 and 800 × 768 hit an encoded-latent shape mismatch during image preparation.

## Windows v0.52 — ControlNet-LLLite

The ZIP SHA-256 is `5656197d6b5f02fe6df292bc5dd1d7ea40a03f9c6ec017d76038c578d039500b`. Using a system-only PATH, the staged `SDV-NUI.exe` found the named Canny adapter through the model catalog and accepted its hint through MCP. The worker reported Vulkan with `cuda=false` on both adapters. Both PNGs recorded the adapter and kept every unmasked pixel. The ZIP contains the tested launcher, worker JAR and native inference DLL byte for byte.

RTX 3090 tested SDXL masked img2img at 768 × 768 and Radeon 780M at 512 × 512 with the published Canny adapter. Both logged 136 applied projections per evaluation and kept all unmasked RGB values. On RTX 3090, changing the hint or narrowing the timestep window changed the edited region. The original hint reproduced its image exactly after intervening changes. The published Depth adapter also ran using an RGB test input, which checked the model path and its effect on the output. The four-step 780M comparison showed pronounced saturation inside the edit with and without the adapter. The images were inspected.

RTX 3090 also tested Reference-only together with LLLite, a separate zero-strength baseline with an empty negative prompt, a reversed interval, fallback to the init image and both hint sources in ordinary Hires. Zero strength reproduced normal Hires pixel for pixel. The nine-channel SDXL Inpaint checkpoint ran a separate 832 × 832 Latent Hires test. Another 832 × 832 test ran LLLite and ordinary Canny through the separate Img2Img Hires Fix. All 45 worker tests passed. The test weights were supplied externally. [Canny adapter](https://huggingface.co/kohya-ss/controlnet-lllite/blob/d14d1353be5a2326229d83e14f9b3c11f3a671bd/controllllite_v01032064e_sdxl_canny.safetensors) · [Depth adapter](https://huggingface.co/kohya-ss/controlnet-lllite/blob/2de0a69f3f27645acc62ba8167203f9d09635f8d/controllllite_v01008016e_sdxl_depth_anime.safetensors).

## Windows v0.51 — Agent Attention

The ZIP SHA-256 is `6d871d8f9ea4578da96791f98efb62607021ebf37931246e481f3696f65e2a64`. The staged `SDV-NUI.exe` ran masked img2img over MCP with SD1.5 on Radeon 780M and SDXL on RTX 3090 using a system-only PATH. It reported Vulkan with `cuda=false`, logged agent-token use, recorded Agent Attention settings in the PNG and preserved every unmasked pixel on both adapters. The archive contains the tested launcher, worker JAR and inference DLL byte for byte.

The worker tested masked SDXL at 768 × 768 on RTX 3090 and SD1.5 at 512 × 512 on Radeon 780M. Native traces recorded ten and five layers respectively using agent tokens. SDXL also exercised ordinary attention on merged tokens, switched stage in FP32, stopped the hook after two evaluations, repeated a same-seed run exactly, and used the hook only during an 832 × 832 Latent Hires pass. Both GPUs preserved the source outside the mask. Test images were inspected. Masked SD1.5 also ran Agent Attention with Canny on RTX 3090 and Reference only with Canny on Radeon 780M. Additional RTX 3090 runs completed Agent Attention with Canny in the ordinary 576 × 576 Hires pass and with the nine-channel SDXL Inpaint checkpoint at 832 × 832. Disabled stages, missing Hires and a depth limit that excluded all SDXL attention layers each matched an unmodified unmasked img2img run pixel for pixel. Hires-only also matched an unmodified run through the separate Img2Img Hires Fix. All 43 worker tests and five Agent Attention plugin tests passed.

## Windows v0.50 — Reference only

The ZIP SHA-256 is `aa8c4b162ae40ace9e345ae4dff09c64086094cdb52521f5e4e92232e06fc338`. The staged `SDV-NUI.exe` accepted a base64 reference over MCP, generated masked SDXL img2img on RTX 3090 with a system-only PATH, recorded the settings in the PNG and preserved every unmasked pixel. The trace showed the reference self-attention path in 70 layers of each CFG branch. The archive contains the tested launcher, worker JAR and inference DLL byte for byte.

Direct tests used SDXL at 768 × 768 on RTX 3090 and SD1.5 at 512 × 512 on Radeon 780M. Both ran masked img2img and kept the original RGB outside the edit. Zero weight made reference selection a no-op. On RTX 3090, changing the reference or style fidelity changed the edited region. Replaying the original reference matched all pixels. The SDXL Inpaint checkpoint with nine U-Net input channels also passed, and standard SDXL completed a latent Hires pass at 832 × 832. Radeon 780M logged 16 reference self-attention layers per branch. All 41 worker tests passed.

## Windows v0.49 — DeepCache and Fast Mode

The ZIP SHA-256 is `387dced983db138a660837ae738f624ea696d0eb6fb93179bbcedb51b3ee71d9`. The staged `SDV-NUI.exe` ran masked SDXL img2img over MCP on RTX 3090 and masked SD1.5 img2img on Radeon 780M with a system-only PATH and reported `cuda=false`. The sampler logged reuse from all three caches and recorded Fast Mode settings in the PNG. Every unmasked pixel matched the input. The ZIP contains the tested launcher, worker JAR and native inference DLL byte for byte.

Direct worker tests used masked SD1.5 at 512 × 512 on Radeon 780M and SDXL at 768 × 768 on RTX 3090. The logs recorded skipped deep U-Net blocks and first- and second-order forecasts. Radeon 780M ran the shorter interval with contraction 0.65. RTX 3090 also ran branch 1 and an 832 × 832 latent Hires pass. Both GPUs retained every unmasked pixel and reproduced the same-seed output with all four Fast Mode hooks. Inspected SDXL output showed pronounced color artifacts in the short extrapolation run. All 39 worker tests and five Fast Mode plugin tests passed.

## Windows v0.48 — T-GATE

The ZIP SHA-256 is `08ba3928b5eac5310df4184030f312089b6df2006d719e370f2165801eb99064`. The staged `SDV-NUI.exe` ran masked SDXL img2img through MCP on RTX 3090 with a system-only PATH. It saved and reused 70 cross-attention outputs, recorded the gate step in the PNG, and preserved every unmasked pixel. The ZIP contains the tested worker JAR and native DLL byte for byte.

Radeon 780M tested SD1.5 at 512 × 512 and reused 16 outputs. RTX 3090 tested SDXL at 768 × 768 with Euler, Heun and CFG 1, plus 832 × 832 Latent Hires with scheduled CFG. The nine-channel SDXL Inpaint checkpoint also reused 70 outputs and preserved all unmasked pixels. Radeon 780M verified separate img2img Hires Fix and masked Canny ControlNet. A same-seed job with an intervening change to the prompt and seed reproduced the original gated image. All 37 worker tests passed.

In a warmed RTX 3090 comparison with 14 effective Euler evaluations at 768 × 768, three alternating runs per setting had a median worker time of 9.41 s with the gate beyond the schedule and 10.09 s with gate step 4. Median native sampling times were 4.94 s and 5.11 s. Same-device cache saves queue their copies before synchronizing. The batched and synchronous copies produced identical RGB in the Radeon 780M ControlNet check.

## Windows v0.47 — CD Tuner

The ZIP SHA-256 is `d02e740ed57c5a6ac197b1edaf44df24306433c64584c96b7511484bd81dac23`. The staged `SDV-NUI.exe` processed `<cdt:d1=3;col1=1>`, additional AlwaysOn values and a 768 × 768 mask through MCP on RTX 3090 with a system-only PATH. The PNG stored the effective `CDT` fields, and every unmasked source pixel matched. The ten-step output retained the fox and meadow. A three-step check with stronger settings produced color artifacts in the edited region. The ZIP contains the tested Java worker and native DLL byte for byte.

Direct worker tests compared a same-seed stopped-detail control with Detail and Contrast 1, the denoised color shift, a two-region Color Map, and both VAE saturation controls on SD1.5 at 512 × 512 on Radeon 780M and SDXL at 768 × 768 on RTX 3090. Each setting altered only the masked region, and the stopped control replayed exactly after active tuning. RTX 3090 also ran the detail hook with a nine-channel SDXL Inpaint checkpoint and the built-in SDXL Latent Hires pass at 832 × 832. Radeon 780M ran the separate img2img Hires Fix and one detected ADetailer face with Apply once on and off. Representative outputs were visually inspected. All 36 worker tests, nine CD Tuner plugin tests and 35 MCP metadata tests passed.

## Windows v0.46 — Fooocus Inpaint

The Windows ZIP SHA-256 is `f05760d224ed0bd8a5b63e3bbddd87517770584a98d01a790264bbaf29e7c962`. The Java 25 worker used the original Fooocus v2.6 patch and head with `animaPencilXL_v500` at 768 × 768 on Radeon 780M and RTX 3090. Both PNGs retained every pixel outside the mask and were visually inspected. The worker reported `cuda=false` on both GPUs. An independent archive read matched all 14,400 exported head values and eleven sampled weights after dequantization and FP16 merging.

On RTX 3090, the same seed produced distinct results with Fooocus disabled, with the weight deltas alone and with the head active throughout sampling. The staged `SDV-NUI.exe` also found the patch by model name via MCP, ran a masked SDXL edit with a system-only PATH and preserved all unmasked pixels. The ZIP contains the tested worker JAR and inference DLL byte for byte. The user supplies the Fooocus weights separately.

The nine-channel `sd_xl_base_1.0_inpainting_0.1` checkpoint completed on Radeon 780M and preserved every pixel outside the mask. The patched mask region showed strong color artifacts in this seven-step run. The stock patch targets a four-channel input convolution, so the worker leaves the nine-channel input weight unchanged when its shape differs. Image quality for this combination remains unverified.

Each job wrote a 5,135,174,472-byte temporary U-Net. Both completed runs released that file. Early runs reached sampling but failed at cleanup on Windows while the native loader held a memory map. The current worker disables memory mapping for the scoped graph, unloads the temporary components and restores the usual setting when the job ends.

## Windows v0.45

The ZIP SHA-256 is `8afb4687c47c1b01ab53fcec7a8a2546d9cd286505e0faa41b84a17a3208c0fd`. The bundled MCP `img2img` tool processed a negative-weight term through NegPiP on Radeon 780M with SD1.5 at 512 × 512. The executable from a fresh ZIP extraction repeated the test on RTX 3090 with SDXL at 768 × 768. Both workers reported `cuda=false` with a system-only PATH.

## Windows v0.44

The ZIP SHA-256 is `cbe8d5908949acacca63ca626f9a40c39092938a40cc3ac8df7d3a616bfec17e`. The packaged MCP `img2img` tool ran FreeU inpainting with a PNG mask and zero mask blur. The staged executable ran SD1.5 at 512 × 512 on Radeon 780M. The executable from a fresh ZIP extraction ran SDXL at 768 × 768 on RTX 3090. Both kept every source pixel outside the mask, used a system-only PATH and reported `cuda=false`.

## Windows v0.43

The ZIP SHA-256 is `4e694e00d5ef40d1012839be265b6737e2907078eeb6926be02572c081b41b9d`. The packaged MCP `img2img` tool applied Latent Modifier sharpness, tone mapping, spectral clamp and seeded noise. The staged executable generated SD1.5 at 512 × 512 on Radeon 780M. The executable from a fresh ZIP extraction generated SDXL at 768 × 768 on RTX 3090. Both used the bundled Java 25 and Vulkan DLL with a system-only PATH and reported `cuda=false`.

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
