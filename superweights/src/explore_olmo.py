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
handle_bucket = []
for i, layer in enumerate(olmo.model.layers):
    def hook(module, inp, out, i=i):
        W = module.weight
        Y = out
        X = inp[0]
        max_activation = Y.abs().argmax().item()
        max_input = X.abs().argmax().item()
        token_x, k = divmod(max_input, X.shape[-1])
        token_y, j = divmod(max_activation, Y.shape[-1])
        print(token_x, k, token_y, j)
        # print(i, module.weight.shape, inp[0].shape, out.shape)
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
