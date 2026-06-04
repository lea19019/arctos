"""Compatibility shims for the (now-deprecated) quantization libraries.

AutoAWQ 0.2.9 is the final release and was last tested on transformers 4.51;
it imports ``PytorchGELUTanh`` from ``transformers.activations``, which newer
transformers (>=4.54) removed. We run transformers 4.57 (needed for Qwen3
support), so we re-inject a functionally-identical activation before AutoAWQ
is imported. Import this module FIRST, then ``import awq``.
"""

from __future__ import annotations


def patch_transformers_activations() -> None:
    """Re-add ``PytorchGELUTanh`` removed from newer transformers."""
    import transformers.activations as A

    if not hasattr(A, "PytorchGELUTanh"):
        import torch.nn as nn
        import torch.nn.functional as F

        class PytorchGELUTanh(nn.Module):
            def forward(self, x):
                return F.gelu(x, approximate="tanh")

        A.PytorchGELUTanh = PytorchGELUTanh


def patch_awq_catcher() -> None:
    """Make AutoAWQ's calibration ``Catcher`` proxy attribute access.

    During calibration AutoAWQ 0.2.9 wraps decoder layer 0 in a local
    ``Catcher`` module and runs a forward pass to capture its inputs. But
    transformers >=4.5x (Qwen3, Llama, ...) reads ``decoder_layer.attention_type``
    on every layer *before* calling it, to pick the right mask from a
    per-attention-type mask mapping. The stock ``Catcher`` doesn't expose that
    attribute, so the calibration forward crashes with
    ``'Catcher' object has no attribute 'attention_type'``.

    We replace ``AwqQuantizer.init_quant`` with a byte-faithful copy whose
    ``Catcher`` delegates any missing attribute to the wrapped layer. AutoAWQ is
    deprecated/frozen, so this copy will not drift.
    """
    import awq.quantize.quantizer as qz

    if getattr(qz.AwqQuantizer, "_arctos_catcher_patched", False):
        return

    nn = qz.nn
    get_calib_dataset = qz.get_calib_dataset
    get_best_device = qz.get_best_device
    clear_memory = qz.clear_memory

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = qz.torch.cat(samples, dim=0)
        inps = []
        layer_kwargs = {}
        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def __getattr__(self, name):
                # nn.Module checks _parameters/_buffers/_modules first; on a
                # miss, delegate to the wrapped layer (e.g. attention_type).
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(super().__getattr__("module"), name)

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)
                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit

        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:
            pass
        modules[0] = modules[0].module  # restore

        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        layer_kwargs.pop("input_ids")
        del samples
        inps = inps[0]
        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")
        clear_memory()
        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(best_device)
        elif "qwen" in self.awq_model.model_type:
            layer_kwargs["attention_mask"] = None
        return modules, layer_kwargs, inps

    qz.AwqQuantizer.init_quant = init_quant
    qz.AwqQuantizer._arctos_catcher_patched = True


def apply_all() -> None:
    patch_transformers_activations()
    patch_awq_catcher()
