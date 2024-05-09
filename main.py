# %%

print("hello")

# %%

from sae_lens import SparseAutoencoder

layer = 8 # pick a layer you want.
sparse_autoencoder = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre"
)

# %%

print(sparse_autoencoder)

# %%

import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Run the model and get logits and activations
logits_of_base_model, activations_of_base_model = model.run_with_cache("Hello World here is more of the sentence with even more text that follows afterwards, I will probably wait too long if it keeps going because I am on CPU")

# %%

activations_of_base_model['blocks.8.hook_resid_pre'].shape

# %%

print(f"{[key for key in activations_of_base_model.keys() if 'blocks.8' in key]=}")

# %%

import torch as t

input_to_sae = activations_of_base_model['blocks.8.hook_resid_pre']

sae_output, activations_of_sae_out = sparse_autoencoder.run_with_cache(
    input_to_sae
)

# %%

sae_output.mse_loss

# %%

dir(sae_output)

# %%

activations_of_base_model