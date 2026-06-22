from comfy.comfy_types.node_typing import IO, ComfyNodeABC
import torch
from .defs import COSY_CATEGORY, COSY_HASH, _mk_hash_key

class CLIPTextEncodeScaled(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {"forceInput": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Multiplier applied to the whole conditioning.",
                }),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled",})
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    RETURN_NAMES = ("cond",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, clip, text, scale=1.0, enabled=True):
        if not enabled or text is None: return (None,)
        if clip is None: raise RuntimeError("ERROR: clip input is invalid: None")
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        scaled = []
        key = _mk_hash_key(text, scale)
        for cond, metadata in cond:
            scaled_metadata = metadata.copy()
            scaled_cond = cond * scale

            pooled_output = scaled_metadata.get("pooled_output", None)
            if torch.is_tensor(pooled_output):
                scaled_metadata["pooled_output"] = pooled_output * scale

            prev = scaled_metadata.get(COSY_HASH, None)
            scaled_metadata[COSY_HASH] = _mk_hash_key(prev, key) if prev else _mk_hash_key(key)

            scaled.append([scaled_cond, scaled_metadata])

        return (scaled,)

STRATEGIES = [
    # prompt A tokens, then prompt B tokens
    "concat A then B",
    # prompt B tokens, then prompt A tokens
    "concat B then A",
    "separate entries",
    "average padded",
    "add padded",
]

#"Primary/Secondary depends on what is A and what is B, as chosen from STRATEGIES."
POOLED_STRATEGIES = ["Avg", "Primary", "Secondary", "Add"]

class CondCombiner(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strategy": (STRATEGIES, {"default": STRATEGIES[0]}),
                "pooled_strategy": (POOLED_STRATEGIES, {"default": POOLED_STRATEGIES[0]}),
            },
            "optional": {
                "cond_a": (IO.CONDITIONING,),
                "cond_b": (IO.CONDITIONING,),
            },
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    RETURN_NAMES = ("cond",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, cond_a, cond_b, strategy, pooled_strategy):
        if cond_a is None or cond_b is None: return (cond_b if cond_a is None else cond_a,)

        # at this point forward, both A and B are defined.
        if strategy == "separate entries":
            return (self._combine_entries(cond_a, cond_b),)

        if len(cond_a) != 1 or len(cond_b) != 1:
            raise ValueError(
                f"Strategy '{strategy}' expects one conditioning entry from each input. "
                f"Got {len(cond_a)} and {len(cond_b)}."
            )

        tensor_a, meta_a = cond_a[0]
        tensor_b, meta_b = cond_b[0]

        self._validate_basic_compatibility(tensor_a, tensor_b)

        if strategy == "concat A then B":
            merged_tensor = torch.cat((tensor_a, tensor_b), dim=1)
            merged_meta = self._merge_metadata(meta_a, meta_b, pooled_strategy)

        elif strategy == "concat B then A":
            merged_tensor = torch.cat((tensor_b, tensor_a), dim=1)
            merged_meta = self._merge_metadata(meta_b, meta_a, pooled_strategy)

        elif strategy == "average padded":
            padded_a, padded_b = self._pad_to_same_token_length(tensor_a, tensor_b)
            merged_tensor = (padded_a + padded_b) / 2.0
            merged_meta = self._merge_metadata(meta_a, meta_b, pooled_strategy)

        elif strategy == "add padded":
            padded_a, padded_b = self._pad_to_same_token_length(tensor_a, tensor_b)
            merged_tensor = padded_a + padded_b
            merged_meta = self._merge_metadata(meta_a, meta_b, pooled_strategy)

        else:
            raise ValueError(f"Unknown conditioning combine strategy: {strategy}")

        return ([[merged_tensor, merged_meta]],)

    def _merge_metadata(self, meta_1, meta_2, pooled_strategy):
        merged_meta = meta_1.copy()

        _po = "pooled_output"
        pooled_1 = meta_1.get(_po, None)
        pooled_2 = meta_2.get(_po, None)

        has_1 = torch.is_tensor(pooled_1)
        has_2 = torch.is_tensor(pooled_2)

        if not has_1 and not has_2:
            raise RuntimeError("pooled_output is missing from both conditionings.")

        _ha = COSY_HASH
        hash_1 = meta_1.get(_ha, None)
        hash_2 = meta_2.get(_ha, None)
        if hash_1 is not None and hash_2 is not None: merged_meta[_ha] = _mk_hash_key(hash_1, hash_2)
        elif hash_1 is not None: merged_meta[_ha] = hash_1
        elif hash_2 is not None: merged_meta[_ha] = hash_2

        if pooled_strategy == POOLED_STRATEGIES[1]: #Primary
            if not has_1: raise RuntimeError("pooled_strategy='A' but A has no pooled_output.")
            merged_meta[_po] = pooled_1
            return merged_meta

        if pooled_strategy == POOLED_STRATEGIES[2]: #Secondary
            if not has_2:
                raise RuntimeError("pooled_strategy='B' but B has no pooled_output.")
            merged_meta[_po] = pooled_2
            return merged_meta

        if has_1 and has_2:
            if pooled_1.shape != pooled_2.shape:
                raise ValueError(f"pooled_output shape mismatch: {pooled_1.shape} vs {pooled_2.shape}")

            merged_meta[_po] = (pooled_1 + pooled_2) #add
            if pooled_strategy == POOLED_STRATEGIES[0]: #Average
                merged_meta[_po] /= 2.0
            elif pooled_strategy != POOLED_STRATEGIES[3]: #add
                raise ValueError(f"Unknown pooled_strategy: {pooled_strategy}")
            return merged_meta

        merged_meta[_po] = pooled_1 if has_1 else pooled_2
        return merged_meta

    def _combine_entries(self, cond_a, cond_b):
        output = []

        for tensor, meta in cond_a:
            output.append([tensor, meta.copy()])

        for tensor, meta in cond_b:
            output.append([tensor, meta.copy()])

        return output

    def _validate_basic_compatibility(self, tensor_a, tensor_b):
        if tensor_a.shape[0] != tensor_b.shape[0]:
            raise ValueError(f"Batch mismatch: {tensor_a.shape} vs {tensor_b.shape}")

        if tensor_a.shape[-1] != tensor_b.shape[-1]:
            raise ValueError(
                f"Embedding/channel mismatch: {tensor_a.shape} vs {tensor_b.shape}. "
                "Use conditionings produced by the same CLIP/text encoder."
            )

    def _pad_to_same_token_length(self, tensor_a, tensor_b):
        len_a = tensor_a.shape[1]
        len_b = tensor_b.shape[1]

        if len_a == len_b:
            return tensor_a, tensor_b

        if len_a < len_b:
            pad = torch.zeros(
                (
                    tensor_a.shape[0],
                    len_b - len_a,
                    tensor_a.shape[2],
                ),
                dtype=tensor_a.dtype,
                device=tensor_a.device,
            )
            tensor_a = torch.cat((tensor_a, pad), dim=1)
        else:
            pad = torch.zeros(
                (
                    tensor_b.shape[0],
                    len_a - len_b,
                    tensor_b.shape[2],
                ),
                dtype=tensor_b.dtype,
                device=tensor_b.device,
            )
            tensor_b = torch.cat((tensor_b, pad), dim=1)

        return tensor_a, tensor_b

