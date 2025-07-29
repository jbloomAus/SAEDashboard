# %%

from transformer_lens import HookedTransformer
import transformer_lens.utils as utils


model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m")

# %%

utils.test_prompt("Michael Jordan plays the sport of", " basketball", model)
# %%
