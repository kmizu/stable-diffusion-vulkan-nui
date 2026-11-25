# Vulkan inference implementation plan

Goal: run SD1.5 and SDXL from the Windows desktop and MCP on selected Vulkan adapters using a Java 25 FFM worker.

Design: ../specs/2026-11-26-vulkan-worker-design.md

- [x] Bind the f296cb0 C ABI in vulkanWorker/src/main/java/sdfx/vulkan/VulkanInference.java. Validate device discovery, model/context lifetime, progress, cancellation, and PNG output on the 780M before moving to the RTX 3090.
- [x] Implement the existing JSON Lines protocol in vulkanWorker/src/main/scala/sdfx/vulkan/VulkanWorker.scala. Reject unsupported requests and verify model replacement, output paths and error recovery.
- [x] Add an isolated Vulkan build profile, SPI provider and packaged worker launch. Preserve conventional builds. Verify profile dependency resolution and extracted layout.
- [x] Assemble and exercise normal desktop and MCP generation. Inspect SD1.5 and SDXL outputs and summarize hardware limits and extension parity.

Verification results and remaining hardware/feature coverage are recorded in [the verification report](../../vulkan-worker-verification.md).
