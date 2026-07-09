#!/usr/bin/env python3
"""One-off probe for VQA-Med HF datasets."""
from huggingface_hub import list_datasets

print("=== HF search: vqa-med ===")
for d in list_datasets(search="vqa-med", limit=30):
    print(d.id)

print("\n=== HF search: imageclef vqa ===")
for d in list_datasets(search="imageclef vqa med", limit=30):
    print(d.id)
