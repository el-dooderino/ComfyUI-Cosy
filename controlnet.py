from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from .defs import COSY_CATEGORY, CONDPipe_t
from nodes import ControlNetApplyAdvanced as ContN

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
    RETURN_TYPES = ("ControlNetCf",)
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
                "controlnet_cf": ("ControlNetCf",{}),
            },
        }
    RETURN_TYPES = (IO.BOOLEAN,)
    RETURN_NAMES = ("enabled",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, controlnet_cf):
        _, _, _, enabled = controlnet_cf
        return enabled,

class MaybeApplyControlNet(ContN):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CONDPipe": (CONDPipe_t, {"tooltip": "Duh."}),
                "control_net": ("CONTROL_NET", {"lazy": True}),
                "image": (IO.IMAGE, {"lazy": True}),
                "controlnet_cf": ("ControlNetCf",{}),
            },
            "optional": {
                "vae": (IO.VAE, {"lazy": True}),
            }

        }
    RETURN_TYPES = (CONDPipe_t,)
    RETURN_NAMES = ("CONDPipe",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def check_lazy_status(self, CONDPipe, control_net, image, controlnet_cf, vae=None):
        if controlnet_cf is None: return []  # Safety
        _, _, _, enabled = controlnet_cf

        if enabled:
            needed = ["control_net", "image"]
            if vae is not None: needed.append("vae")
            return needed
        else:
            return []  # Don't load anything when disabled

    def run(self, CONDPipe, control_net, image, controlnet_cf, vae=None):
        positive, negative = CONDPipe
        strength, start_percent, end_percent, enabled = controlnet_cf
        if enabled:
            positive, negative = ContN.apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=vae)
        return (positive, negative),
