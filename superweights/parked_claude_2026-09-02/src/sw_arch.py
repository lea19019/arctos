"""Architecture adapter: where are the decoder layers, and which module is
the FFN's output projection (the `down_proj` of Yu et al.)?

Every runner in this track hard-coded `model.model.layers[i].mlp.down_proj`
(Llama / Mistral / OLMo / Phi-3 / Qwen). RQ2 needs other families:

    family        layers                       FFN output projection
    llama-like    model.model.layers           mlp.down_proj
    gpt_neox      model.gpt_neox.layers        mlp.dense_4h_to_h   (Pythia)
    bloom         model.transformer.h          mlp.dense_4h_to_h
    gemma3 (VLM)  model.model.language_model.layers   mlp.down_proj
    cohere        model.model.layers           mlp.down_proj       (Aya)
    m2m_100       encoder.layers / decoder.layers     fc2          (NLLB) -- two stacks

Weight convention is the same everywhere: W has shape (d_model, d_ff), so
W[j, k] maps intermediate neuron k into residual channel j.

`down_proj(layer)` returns the module; `get_layers(model)` returns the layer
list for a decoder-only model. For encoder-decoder models use `get_stacks`.
"""

LAYER_PATHS = ["model.layers", "gpt_neox.layers", "transformer.h",
               "model.language_model.layers"]
FFN_OUT_ATTRS = ["mlp.down_proj", "mlp.dense_4h_to_h", "mlp.fc2", "mlp.c_proj",
                 "fc2"]


def _resolve(obj, dotted):
    for a in dotted.split("."):
        obj = getattr(obj, a, None)
        if obj is None:
            return None
    return obj


def down_proj(layer):
    for attr in FFN_OUT_ATTRS:
        m = _resolve(layer, attr)
        if m is not None:
            return m
    raise SystemExit(f"No FFN output projection found on {type(layer).__name__}; "
                     f"tried {FFN_OUT_ATTRS}. Extend sw_arch.FFN_OUT_ATTRS.")


def get_layers(model):
    """Decoder-only: the list of transformer blocks."""
    for p in LAYER_PATHS:
        layers = _resolve(model, p)
        if layers is not None:
            down_proj(layers[0])          # fail loudly here, not mid-run
            return layers
    raise SystemExit(f"Unsupported architecture ({type(model).__name__}): "
                     f"no layer list at any of {LAYER_PATHS}. Extend sw_arch.")


def get_stacks(model):
    """Encoder-decoder: {'encoder': layers, 'decoder': layers}."""
    enc = _resolve(model, "model.encoder.layers")
    dec = _resolve(model, "model.decoder.layers")
    if enc is None or dec is None:
        raise SystemExit(f"{type(model).__name__} is not a recognised "
                         f"encoder-decoder (need model.encoder.layers and "
                         f"model.decoder.layers).")
    return {"encoder": enc, "decoder": dec}


def is_encoder_decoder(model):
    return bool(getattr(model.config, "is_encoder_decoder", False))
