# Evaluating Catastrophic Forgetting

This repository contains the reproduction script for GenAI-ML 2025 Homework 8.

## Model Details
- **Base Model:** Llama-3.2-1B-Instruct
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Hugging Face Path:** [o-mouse/hw8-llama-finetuned-v2](https://huggingface.co/o-mouse/hw8-llama-finetuned-v2)

## How to Reproduce Results
To install dependencies, download the datasets, and run the evaluation (GSM8K and Safety Rate), execute the following command:

```bash
export HF_TOKEN="your_huggingface_token_here"
bash run.sh