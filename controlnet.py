import folder_paths
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy.controlnet import load_controlnet as cn_load
from .defs import COSY_CATEGORY, COSY_HASH, CONDPipe_t, _mk_hash_key, _hash_tensor
from nodes import ControlNetApplyAdvanced as ContN
import comfy.utils

def load_controlnet(control_net_name):
    controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
    controlnet = cn_load(controlnet_path)
    if controlnet is None:
        raise RuntimeError("ERROR: controlnet file is invalid and does not contain a valid controlnet model.")
    return controlnet

class ControlNetRC(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_pct": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_pct": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "enabled": (IO.BOOLEAN, {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }
    RETURN_TYPES = ("COSY_CONTROLNET_RC",)
    RETURN_NAMES = ("controlnet_cf",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, strength, start_pct, end_pct, enabled=True):
        return (strength, start_pct, end_pct, enabled),

class ControlNet_enabled(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_cf": ("COSY_CONTROLNET_RC",{}),
            },
        }
    RETURN_TYPES = (IO.BOOLEAN,)
    RETURN_NAMES = ("enabled",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, controlnet_cf):
        _, _, _, enabled = controlnet_cf
        return enabled,

class MaybeApplyControlNet(ComfyNodeABC):
    @classmethod
    def __init__(self):
        self._cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CONDPipe": (CONDPipe_t, {"tooltip": "Duh."}),
                "control_net": (folder_paths.get_filename_list("controlnet"),),
                "image": (IO.IMAGE, {"lazy": True}),
                "controlnet_cf": ("COSY_CONTROLNET_RC",{}),
            },
            "optional": {
                "mask": (IO.MASK,{"tooltip": "Optional mask, stretched to image size, applied to image before ControlNet processing. Black=IN"}),
                "vae": (IO.VAE, {"lazy": True}),
            }

        }
    RETURN_TYPES = (CONDPipe_t, IO.IMAGE,)
    RETURN_NAMES = ("CONDPipe", "image",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def check_lazy_status(self, CONDPipe, control_net, image, controlnet_cf, mask=None, vae=None):
        if controlnet_cf is None: return []  # Safety
        _, _, _, enabled = controlnet_cf

        ret = []
        if enabled:
            ret.append("image")

        return ret

    def run(self, CONDPipe, control_net, image, controlnet_cf, mask = None, vae=None):
        positive, negative = CONDPipe
        strength, s_pct, e_pct, enabled = controlnet_cf

        if enabled:
            print(f"Applying controlnet {control_net} to image {image.shape}")
            if self._cache.get("name") != control_net:
                self._cache = {} #clear old one first so exception will not leave it half-baked.
                self._cache = {
                    "name": control_net,
                    "model": load_controlnet(control_net),
                }

            extra_concat = []
            if mask is not None:
                print(f"Applying mask {mask.shape} to image {image.shape}")
                m = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
                m_apply = comfy.utils.common_upscale(m, image.shape[2], image.shape[1], "bilinear", "disabled").round()
                image = image * m_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])
                # print(f"Mask applied {m_apply.shape}, image now {image.shape}")
                if getattr(self._cache["model"], "concat_mask", False): extra_concat = [m]

            positive, negative = ContN().apply_controlnet(positive, negative, self._cache["model"], image, strength, s_pct, e_pct, vae=vae, extra_concat=extra_concat)
            cnet_hash = _mk_hash_key(self._cache["name"], _hash_tensor(image), _hash_tensor(mask), strength, s_pct, e_pct)
            positive = self._add_control_hash(positive, cnet_hash)
            negative = self._add_control_hash(negative, cnet_hash)

        else:
            self._cache = {}

        return (positive, negative), image,

    def _add_control_hash(self, conditioning, cnet_hash):
        out = []
        for cond, meta in conditioning:
            meta = meta.copy()
            previous = meta.get(COSY_HASH)
            meta[COSY_HASH] = _mk_hash_key(previous, cnet_hash) if previous else cnet_hash
            out.append([cond, meta])

        return out