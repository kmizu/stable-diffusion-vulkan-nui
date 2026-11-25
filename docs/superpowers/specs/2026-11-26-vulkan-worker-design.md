# Vulkan inference worker for Windows x64

The Vulkan distribution runs GPU inference on compatible AMD, Intel and NVIDIA devices through a single native build. Users supply their graphics driver and checkpoint. The package supplies the Java 25 worker runtime and the Vulkan build of stable-diffusion.cpp with its runtime libraries and licences.

## Approach

Use Java 25 FFM to call the public C ABI of a pinned stable-diffusion.cpp revision. This puts model execution and shader maintenance in the upstream library. Writing tensor kernels would also support Vulkan but would require implementing and maintaining the model operators here. A CLI subprocess would simplify a single generation while complicating model reuse, cancellation and progress reporting. The existing JSON Lines worker boundary isolates the native library and serves both UI and MCP.

## Integration

Build the Vulkan distribution independently of scala-torch-models, Python and CUDA. Select its worker without probing for an NVIDIA driver. Preserve existing settings and model paths. Expose compatible Vulkan adapters by name and allow explicit selection on multi-GPU machines. Record the adapter actually executing inference. Surface device or allocation failures to the caller. A CPU-only result must never count as Vulkan verification.

Pin the header, library and submodules together. Scope native access to the worker. Keep model handles alive across jobs, release result buffers according to the pinned ABI and clean up on failure, cancellation, model replacement and shutdown. Advertise only operations implemented and verified through the worker protocol. Inventory extension hooks before claiming migration parity.

## Verification and staging

First generate and inspect a 512×512 SD1.5 image via Java 25 FFM. Exercise the same request through the normal desktop and MCP routes. Add SDXL and inspect real-model output. Test the extracted package with both local adapters explicitly selected (Radeon 780M and RTX 3090), recording model, dimensions, adapter, time and any memory limitations. Record Intel as unverified until a compatible device runs the same test. Inspect resolved JVM dependencies, native imports and package contents for CUDA dependencies. Complete a clean-machine run without the CUDA toolkit. Broader extension coverage proceeds from the inventory.

The implementer selects the upstream revision, FFM binding generation method, resource-compatible test settings and reproducible build commands after inspecting the actual ABI and available drivers.
