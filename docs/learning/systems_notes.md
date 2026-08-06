# Systems and hardware notes

Interleaved into Q1–Q5, not a separate track. Each note is the result of doing one of the experiments, written down after the fact.

Required by the time Q5 closes:

- `transformer-math.md` — attention, MLPs, residual stream, RMSNorm/LayerNorm, position encodings derived from scratch.
- `gpu-memory.md` — what lives in VRAM during forward / training; KV cache scaling.
- `attention-at-the-hardware-level.md` — Q/K/V matmul → softmax → attention output, memory access patterns, what FlashAttention changes.
- `kv-cache.md` — growth, dominance at long contexts, KV-quantization implications.
- `kernels-and-deployment.md` — int4/int8 GEMM kernels (Marlin, CUTLASS), why some quantizers have first-class kernels, vLLM scheduler / PagedAttention.
- `model-storage.md` — safetensors, GGUF, HF format; quantized format encoding (scales, zero-points).
- `aws-deployment.md` — instance types, EBS vs instance store, multi-GPU networking.

The point of writing these is the act of writing them; reading the equivalent blog post is not a substitute.
