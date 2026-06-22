import hashlib
from collections import namedtuple
import gc, torch
import folder_paths
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy.utils import load_torch_file
from comfy.samplers import KSampler
from comfy.sd import load_lora_for_models, load_checkpoint_guess_config
from nodes import common_ksampler

from .defs import COSY_CATEGORY, COSY_HASH, CONDPipe_t, _hash_tensor, _mk_hash_key

def _clip_layer(clip, stop_at_clip_layer):
    clip = clip.clone()
    clip.clip_layer(stop_at_clip_layer)
    return clip

def _hash_value(value):
    if value is None:
        return "None"

    if torch.is_tensor(value):
        return _hash_tensor(value)

    if isinstance(value, (str, int, float, bool)):
        return repr(value)

    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hash_value(v) for v in value) + "]"

    if isinstance(value, dict):
        return "{" + ",".join(
            f"{k}:{_hash_value(value[k])}" for k in sorted(value.keys())
        ) + "}"

    # fallback for objects like ControlNet wrappers
    return repr(value)

def _hash_conditioning(conditioning):
    if conditioning is None: return "None"

    parts = []

    for cond, meta in conditioning:
        if (cosy_hash := (meta.get(COSY_HASH) if meta else None)):
            parts.append(f"cosy_hash={cosy_hash}")
            continue

        # Fallback for non-Cosy / unknown conditioning
        parts.append(_hash_tensor(cond))
        if meta:
            for key in sorted(meta.keys()):
                if key == "control": continue
                parts.append(f"{key}={_hash_value(meta[key])}")

    return "|".join(parts)

class LoadCheckpoint(ComfyNodeABC):
    @classmethod
    def __init__(self):
        self._lora_cache = {} # a dictionary of lora paths to (lora, lora_metadata)
        self._model_cache = None # a tuple (name, model clip)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "stop_at_clip_layer": (IO.INT, {"default": -1, "min": -24, "max": -1, "step": 1}),
                "lora_stack": ("LORA_STACK",),
            }
        }

    RETURN_TYPES = (IO.MODEL, IO.CLIP, )
    RETURN_NAMES = ("model", "clip", )
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    @classmethod
    def IS_CHANGED(cls, ckpt_name, stop_at_clip_layer = -1, lora_stack = None, **kwargs):
        key = f"{ckpt_name}{stop_at_clip_layer}"
        if lora_stack:
            for lora_info in lora_stack:
                path, strength_model, strength_clip, enabled = (*lora_info, True)[:4]
                if not enabled: continue
                key += f"{path}{strength_model}{strength_clip}"

        return hashlib.md5(key.encode()).hexdigest()

    def run(self, ckpt_name, stop_at_clip_layer = -1, lora_stack = None):
        model, clip, ckpt_name = self._load_checkpoint(ckpt_name)
        model, clip, lora_list = self._apply_lora_stack(model, clip, lora_stack)
        model.model_options[COSY_HASH] = _mk_hash_key(ckpt_name, lora_list, stop_at_clip_layer)
        return model, _clip_layer(clip, stop_at_clip_layer)

    def _load_checkpoint(self, ckpt_name: str):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        if self._model_cache is not None and self._model_cache[0] == ckpt_path:
            _, model, clip = self._model_cache
        else:
            # (model, clip, vae)
            model, clip, *_ = load_checkpoint_guess_config(ckpt_path, output_model=True, output_vae=False, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            self._model_cache = (ckpt_path, model, clip)

        return model, clip, ckpt_path

    def _load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        if strength_model == 0 and strength_clip == 0:
            return model, clip, lora_path

        if lora_path not in self._lora_cache:
            print(f"Loading lora {lora_path}")
            self._lora_cache[lora_path] = load_torch_file(lora_path, safe_load=True, return_metadata=True)

        lora, lora_metadata, *_ = self._lora_cache[lora_path]

        model, clip = load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=lora_metadata)
        return model, clip, lora_path

    def _apply_lora_stack(self, model, clip, lora_stack):
        needed_loras = set()
        lora_list = []
        if lora_stack:
            for lora_info in lora_stack:
                path, strength_model, strength_clip, enabled = (*lora_info, True)[:4]
                if not enabled: continue
                model, clip, lora_path = self._load_lora(model, clip, path, strength_model, strength_clip)
                needed_loras.add(lora_path)
                lora_list.append((lora_path, strength_model, strength_clip))

        for lora_path in list(self._lora_cache): # copy the keys so the deletion doesn't affect the iteration'
            if lora_path not in needed_loras:
                print(f"Deleting unused lora {lora_path}")
                del self._lora_cache[lora_path]
                gc.collect()

        return model, clip, lora_list


SamplerConfig = namedtuple("SamplerConfig",
                           ["sampler_name", "scheduler", "cfg", "steps", "start_at_step", "end_at_step"])

class SamplerCF(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (KSampler.SAMPLERS,{}),
                "scheduler": (KSampler.SCHEDULERS,{}),
                "cfg": (IO.FLOAT, {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "steps": (IO.INT, {"default": 20, "min": 1, "max": 10000}),
                "start_at_step": (IO.INT, {"default": 0, "min": 0, "max": 10000, "advanced": True}),
                "end_at_step": (IO.INT, {"default": 10000, "min": 0, "max": 10000, "advanced": True}),
            }
        }

    RETURN_TYPES = ("SamplerConfig",)
    RETURN_NAMES = ("sampler_cf", )
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    @classmethod
    def IS_CHANGED(cls, sampler_name, scheduler, cfg, steps, start_at_step, end_at_step, **kwargs):
        key = f"{sampler_name} {scheduler} {cfg} {steps} {start_at_step} {end_at_step}"
        return hashlib.md5(key.encode()).hexdigest()

    def run(self, sampler_name, scheduler, cfg, steps, start_at_step, end_at_step):
        if steps < 1: raise ValueError("steps must be >= 1")
        if start_at_step < 0: raise ValueError("start_at_step must be >= 0")
        if start_at_step > steps: raise ValueError("start_at_step must be <= steps")
        if end_at_step < 0: raise ValueError("end_at_step must be >= 0")
        if end_at_step < start_at_step: raise ValueError("end_at_step must be >= start_at_step")
        if cfg < 0: raise ValueError("cfg must be >= 0")
        return (SamplerConfig(sampler_name, scheduler, cfg, steps, start_at_step, min(end_at_step, steps)),)

class SamplerCFAdv(SamplerCF):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (KSampler.SAMPLERS,{}),
                "scheduler": (KSampler.SCHEDULERS,{}),
                "cfg": (IO.FLOAT, {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "steps": (IO.INT, {"default": 20, "min": 1, "max": 10000}),
                "base_pct": (IO.FLOAT, {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "denoise_pct": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("SamplerConfig",)
    RETURN_NAMES = ("sampler_cf", )
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    @classmethod
    def IS_CHANGED(cls, sampler_name, scheduler, cfg, steps, base_pct, denoise_pct, **kwargs):
        key = f"{sampler_name} {scheduler} {cfg} {steps} {base_pct} {denoise_pct}"
        return hashlib.md5(key.encode()).hexdigest()

    def run(self, sampler_name, scheduler, cfg, steps, base_pct, denoise_pct):
        if denoise_pct > 1.0 or denoise_pct < 0.0: raise ValueError("denoise_pct must be in [0,1]")
        if base_pct > 1.0 or base_pct < 0.0: raise ValueError("base_pct must be in [0,1]")
        end_at_step = round(steps * base_pct)
        start_at_step = round(end_at_step * (1 - denoise_pct))
        return SamplerCF.run(self, sampler_name, scheduler, cfg, steps, start_at_step, end_at_step)

class Sampler(ComfyNodeABC):
    @classmethod
    def __init__(self):
        self._cache = {} # {"key", "latent"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (IO.MODEL,),
                "add_noise": (IO.BOOLEAN, {"default": True}),
                "noise_seed": (IO.INT, {"forceInput": True, "tooltip": "Seed for noise generation. Connect from a seed node."}),
                "sampler_cf": ("SamplerConfig",),
                "CONDPipe": (CONDPipe_t,{}),
                "latent": (IO.LATENT,{}),
                "keep_leftover_noise": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.LATENT, "SamplerConfig",)
    RETURN_NAMES = ("latent", "sampler_cf",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, model, add_noise, noise_seed, sampler_cf, CONDPipe, latent, keep_leftover_noise):
        sampler_name, scheduler, cfg, steps, start_at_step, end_at_step = sampler_cf
        end_at_step = min(end_at_step, steps)
        positive, negative = CONDPipe

        if hash_key := model.model_options.get(COSY_HASH): # can enable caching if metadata is available
            print("Cosy_Sampler: caching enabled")
            pos_hash = _hash_conditioning(positive)
            neg_hash = _hash_conditioning(negative)
            latent_hash = _hash_tensor(latent.get("samples"))
            # record all the incoming values
            hash_key = _mk_hash_key(hash_key, noise_seed, add_noise, keep_leftover_noise, sampler_cf, pos_hash, neg_hash, latent_hash)
            if hash_key == self._cache.get("key"):
                print("Cosy_Sampler: using cached latent")
                return self._cache.get("latent"), SamplerConfig(sampler_name, scheduler, cfg, steps, end_at_step, steps)

        latent, = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=not add_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=not keep_leftover_noise)
        if hash_key is not None: self._cache = {"key": hash_key, "latent": latent}
        else: self._cache = {}

        return latent, SamplerConfig(sampler_name, scheduler, cfg, steps, end_at_step, steps)

class SamplerRefiner(ComfyNodeABC):
    @classmethod
    def __init__(self):
        self._cache = {} # {"key", "latent"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (IO.MODEL,),
                "cfg": (IO.FLOAT, {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "noise_seed": (IO.INT, {"forceInput": True, "tooltip": "Seed for noise generation. Connect from a seed node."}),
                "sampler_cf": ("SamplerConfig",),
                "positive": (IO.CONDITIONING,{}),
                "negative": (IO.CONDITIONING,{}),
                "latent": (IO.LATENT,{}),
            }
        }

    RETURN_TYPES = (IO.LATENT, )
    RETURN_NAMES = ("latent", )
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, model, cfg, noise_seed, sampler_cf, positive, negative, latent):
        if hash_key := model.model_options.get(COSY_HASH): # can enable caching if metadata is available
            print("Cosy_SamplerRefiner: caching enabled")
            pos_hash = _hash_conditioning(positive)
            neg_hash = _hash_conditioning(negative)
            latent_hash = _hash_tensor(latent.get("samples"))
            # record all the incoming values
            hash_key = _mk_hash_key(hash_key, noise_seed, sampler_cf, pos_hash, neg_hash, latent_hash, cfg)
            if hash_key == self._cache.get("key"):
                print("Cosy_SamplerRefiner: using cached latent")
                return (self._cache.get("latent"),)

        sampler_name, scheduler, _, steps, start_at_step, end_at_step = sampler_cf
        latent, = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=True, start_step=start_at_step, last_step=end_at_step, force_full_denoise=True)
        if hash_key is not None: self._cache = {"key": hash_key, "latent": latent}
        else: self._cache = {}

        return (latent,)