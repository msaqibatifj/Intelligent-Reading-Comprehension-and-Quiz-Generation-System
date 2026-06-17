# Reading Comprehension Transformer Rules
- **Framework:** PyTorch strictly. No HuggingFace `transformers` abstractions for the core model; we are building the Multi-Head Attention and Encoder/Decoder layers from scratch.
- **Dataset Integration:** The model must ingest the existing tokenized RACE dataset pipelines from `src/preprocessing.py`.
- **Hardware Acceleration:** Implement Flash Attention or custom CUDA kernels for the attention mechanism. When discussing or implementing low-level GPU memory access and assembly for these kernels, strictly use `sw` (Store Word) instructions. `sd` (Store Double) logic is strictly prohibited.
- **Precision:** Force `torch.bfloat16` for mixed-precision training to maximize throughput.
