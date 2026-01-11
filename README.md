# LLaMA 3 Fine-Tuning with LoRA (Research-Driven & Hardware-Aware)

## Overview

This project presents a **resource-efficient fine-tuning pipeline** for **LLaMA 3 (8B Instruct)** using **LoRA (Low-Rank Adaptation)** combined with **4-bit quantization**, designed specifically for **constrained GPU environments** such as Google Colab and NVIDIA A100-class GPUs.

The defining aspect of this work is a **deliberate, research-based approach to hyperparameter selection**, where every configuration choice is motivated by:
- empirical experimentation,
- hardware limitations,
- and official NVIDIA chip-level recommendations.

This is not a default-based or intuition-only setup, but an **engineering-driven fine-tuning system**.

---

## Key Design Principles

- Stability over raw speed  
- Cost-aware optimization  
- Hardware-specific precision selection  
- Minimal parameter adaptation with maximal behavioral impact  
- Explicit justification for every hyperparameter

---

## Model & Quantization

- **Base Model:** LLaMA-3-8B-Instruct  
- **Framework:** Unsloth + Hugging Face Transformers  
- **Quantization:** 4-bit (bnb)  
- **Sequence Length:** 1024  

The 8B model size was chosen as the optimal balance between capability and computational cost.  
4-bit quantization significantly reduces VRAM usage while preserving performance when combined with LoRA.

---

## Precision Strategy (Hardware-Aware)

- **Precision:** bfloat16  
- **Reasoning:**  
  Selected after reviewing NVIDIA A100 documentation and performance guidelines.  
  bfloat16 offers numerical stability close to fp32 while maintaining execution speed comparable to fp16, making it the optimal precision choice for this hardware.

This decision was **research-driven**, not preference-based.

---

## LoRA Configuration

- **Rank (r):** 8  
  - Incrementally tested (2 → 4 → 8)
  - Higher ranks showed diminishing returns relative to cost

- **Target Modules:**  
  `q_proj, k_proj, v_proj, o_proj`  
  - Core attention projections
  - Directly influence reasoning and context focus

- **LoRA Alpha:** 16  
  - Balanced update strength without destabilizing pretrained representations

- **LoRA Dropout:** 0.0  
  - Disabled due to dataset size and absence of overfitting signals

- **Bias:** none  
  - Directional adaptation preferred over global output shifts

---

## Memory Optimization

- **Gradient Checkpointing:** Enabled (Unsloth implementation)  
  - Reduces memory footprint
  - Trades recomputation for VRAM efficiency

- **Attention Implementation:** eager  
  - Chosen for training stability and compatibility

---

## Dataset

- **Dataset:** FineTome-100k  
- **Used Samples:** 50,000  
- **Split:** 80% training / 20% evaluation  

Data was:
- formatted using the official LLaMA 3 chat template,
- filtered by length,
- and structured to preserve clear user/assistant roles.

---

## Training Configuration

- **Effective Batch Size:** 16  
  - (batch size = 2, gradient accumulation = 8)

- **Learning Rate:** 1e-4  
  - Optimized for LoRA + 4-bit training

- **Warmup Steps:** 100  
  - Prevents early optimization instability

- **Scheduler:** Cosine  
  - Strong early learning with smooth convergence

- **Max Training Steps:** 700  
  - Selected after convergence analysis to avoid compute waste

- **Optimizer:** AdamW (8-bit)  
  - Fast, memory-efficient, and stable

- **Precision:** bf16 (fp16 disabled)

---

## Evaluation Strategy

- Evaluation every 100 steps
- Best checkpoint selected based on evaluation loss
- Custom callback tracks moving average of eval loss for stability monitoring

---

## Project Outcome

This project demonstrates:
- Fine-grained control over LLM fine-tuning
- Hardware-aware and cost-efficient design
- Research-backed hyperparameter reasoning
- Practical understanding of LoRA and quantized training dynamics

The result is a **stable, efficient, and explainable fine-tuning pipeline**, suitable for real-world constraints and production-oriented experimentation.

---

## Use Cases

- Instruction tuning
- Conversational alignment
- Research experiments under limited compute
- Resume-ready demonstration of LLM fine-tuning expertise
