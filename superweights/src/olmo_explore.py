# Implement the 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)

print("----------------------------------")
print(inputs)
# print("----------------------------------")
# print(olmo.model.layers)
print("----------------------------------")
# print(olmo.model.layers[0].mlp.down_proj.register_forward_hook)
# print(olmo.model.layers[0].mlp.down_proj)

'''
Y=XW^T
X   = L (row) x H (col)
W   = D (row) x H (col)
W^T = H (row) x D (col)
Y   = L (row) x D (col)
LxD

'''

handle_bucket = []
for i, layer in enumerate(olmo.model.layers):
    def hook(module, inp, out, i=i):
        W = module.weight
        Y = out
        X = inp[0]

        y_flat_idx = Y.abs().argmax().item()
        x_flat_idx = X.abs().argmax().item()
        token_x, k = divmod(x_flat_idx, X.shape[-1])
        token_y, j = divmod(y_flat_idx, Y.shape[-1])

        x_spike = X[0, token_x, k].item()
        y_spike = Y[0, token_y, j].item()
        sw_value = W[j, k].item()

        print("----------------------------------")
        print(f"Layer {i}")
        print(f"  max|X| = {X.abs().max().item():10.3f}  at (token {token_x}, ch {k})")
        print(f"  max|Y| = {Y.abs().max().item():10.3f}  at (token {token_y}, ch {j})")
        print(f"  W[{j},{k}] = {sw_value:.4f}   (max|W| in row {j}: "
              f"{W[j].abs().max().item():.4f}, whole matrix: {W.abs().max().item():.4f})")

        # dominance check: does the single product explain the output spike?
        print(f"  X*W = {x_spike * sw_value:10.3f}   vs   Y = {y_spike:10.3f}")

        # top-10s: how lonely are the spikes?
        y_top = torch.topk(Y.abs().flatten(), 10).values
        x_top = torch.topk(X.abs().flatten(), 10).values
        w_row_top = torch.topk(W[j].abs(), 10).values
        print(f"  top10|Y|:    {[round(v, 2) for v in y_top.tolist()]}")
        print(f"  top10|X|:    {[round(v, 2) for v in x_top.tolist()]}")
        print(f"  top10|W[{j}]|: {[round(v, 4) for v in w_row_top.tolist()]}")

    handle = layer.mlp.down_proj.register_forward_hook(hook)
    handle_bucket.append(handle)

olmo(**inputs)
[handle.remove() for handle in handle_bucket]

# print("----------------------------------")
# print(olmo)
# print(tokenizer)


# ------------------------- INFERENCE EXAMPLE -------------------------
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
# response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
