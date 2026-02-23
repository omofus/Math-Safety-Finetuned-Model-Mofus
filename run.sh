#!/bin/bash
echo "Installing dependencies..."
pip install -q torch transformers peft bitsandbytes accelerate datasets

echo "Starting Evaluation Pipeline..."
python eval.py
