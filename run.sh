#!/bin/bash
# Install required dependencies
pip install -U datasets trl bitsandbytes transformers accelerate peft tqdm

# Download required datasets if not present
wget -q https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl
wget -q https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_public.jsonl
wget -q https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_private.jsonl
wget -q https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv

# Run the evaluation
python eval.py