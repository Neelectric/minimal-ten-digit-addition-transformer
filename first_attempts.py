"""
Playing around with visualisations
"""

import argparse
from ast import mod
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs
from scipy.optimize import minimize

from train import make_batch, loss_fn, evaluate, INTERMEDIATE_SIZE, encode

parser = argparse.ArgumentParser()
parser.add_argument("--ff", type=int, default=INTERMEDIATE_SIZE)
parser.add_argument("--heads", type=int, default=1)
parser.add_argument("--kv", type=int, default=1)
args = parser.parse_args()

model_args = ModelArgs(
    model_type="qwen3", hidden_size=3, num_hidden_layers=1,
    intermediate_size=args.ff, num_attention_heads=args.heads, rms_norm_eps=1e-6,
    vocab_size=10, tie_word_embeddings=True, num_key_value_heads=args.kv,
    max_position_embeddings=64, rope_theta=3, head_dim=4,
)

model = Model(model_args)
mx.eval(model.parameters())
template = tree_flatten(model.parameters())
param_shapes = [(name, p.shape) for name, p in template]
n_params = sum(p.size for _, p in template)

checkpoint = f"checkpoint/best_{n_params}.npz"
print(f"checkpoint is {checkpoint}")
model.load_weights(list(mx.load(checkpoint).items()))
model.eval()
mx.eval(model.parameters())
template = tree_flatten(model.parameters())

sa, da = evaluate(model, 1000, random.Random(12345))
print(f"Loaded {checkpoint} ({n_params} params)")
print(f"Before: seq_acc={sa:.4f}  dig_acc={da:.4f}")

for name, module in model.named_modules():
    if not hasattr(module, "weight"):
        # print(f"Module {name} has no weight attr, skipping...")
        # print("\n"*3)
        continue
        
    print(f"Name: {name}, Module: {module}")
    print(f"weight: {module.weight}")
    print("\n"*2)
    

# OUTPUT_DIGITS is 11 as defined in train.py and submission.py
OUTPUT_DIGITS = 11

def predict_addition(model, a: int, b: int) -> int:
    """Uses the loaded model to predict the sum of a and b."""
    # Encode the inputs using the function imported from train.py
    seq = encode(a, b)
    
    digits = []
    # Auto-regressively predict the next digit
    for _ in range(OUTPUT_DIGITS):
        x = mx.array([seq], dtype=mx.int32)
        logits = model(x)
        # Get the argmax of the last token's logits
        d = int(mx.argmax(logits[0, -1, :]).item())
        seq.append(d)
        digits.append(d)
        
    # The digits are predicted Least-Significant-Digit first, 
    # so we reverse them and convert back to an integer.
    pred_str = "".join(str(d) for d in reversed(digits))
    return int(pred_str)

# Interactive loop for terminal testing
# if __name__ == "__main__":
#     print("\n" + "="*30)
#     print("🚀 Model Inference Ready")
#     print("="*30)
    
#     while True:
#         try:
#             user_input = input("\nEnter two integers separated by a space (or 'q' to quit): ")
#             if user_input.lower().strip() == 'q':
#                 break
                
#             parts = user_input.split()
#             if len(parts) != 2:
#                 print("Please enter exactly two numbers.")
#                 continue
                
#             a, b = int(parts[0]), int(parts[1])
            
#             # Prevent inputs larger than the model's 10-digit training max
#             if a > 10**10 - 1 or b > 10**10 - 1:
#                 print("Warning: Model was only trained on up to 10-digit numbers.")
            
#             prediction = predict_addition(model, a, b)
#             actual = a + b
            
#             print(f"Model Prediction: {prediction}")
#             print(f"Actual Math:      {actual}")
#             print(f"Match:            {'✅ Yes' if prediction == actual else '❌ No'}")
            
#         except ValueError:
#             print("Invalid input. Please enter valid integers.")
#         except KeyboardInterrupt:
#             print("\nExiting...")
#             break