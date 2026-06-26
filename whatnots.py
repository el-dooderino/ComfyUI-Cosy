from comfy.comfy_types.node_typing import IO, ComfyNodeABC
import comfy.model_management as mm
from .defs import WHPipe_t, CONDPipe_t, COSY_CATEGORY

class WHPipeIn(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "W": (IO.INT, {"tooltip": "Duh."}),
                "H": (IO.INT, {"tooltip": "Duh."}),
            }
        }

    RETURN_TYPES = (WHPipe_t,)
    RETURN_NAMES = ("WHPipe",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, W, H):
        return ((W,H,),)

class WHPipeOut(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "WHPipe": (WHPipe_t, {"tooltip": "Duh."}),
            }
        }

    RETURN_TYPES = (IO.INT, IO.INT,)
    RETURN_NAMES = ("W", "H")
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, WHPipe):
        W, H = WHPipe
        return W,H

class ResolutionPicker(ComfyNodeABC):
    DEF_RESOLUTION = "1024x1024 (1.0)"
    RESOLUTIONS = {
        "704x1408 (0.5)": (704, 1408, 0.5),
        "704x1344 (0.52)": (704, 1344, 0.52),
        "768x1344 (0.57)": (768, 1344, 0.57),
        "768x1280 (0.6)": (768, 1280, 0.6),
        "832x1216 (0.68)": (832, 1216, 0.68),
        "832x1152 (0.72)": (832, 1152, 0.72),
        "896x1152 (0.78)": (896, 1152, 0.78),
        "896x1088 (0.82)": (896, 1088, 0.82),
        "960x1088 (0.88)": (960, 1088, 0.88),
        "960x1024 (0.94)": (960, 1024, 0.94),
        DEF_RESOLUTION: (1024, 1024, 1.0),
        "1024x960 (1.07)": (1024, 960, 1.07),
        "1088x960 (1.13)": (1088, 960, 1.13),
        "1088x896 (1.21)": (1088, 896, 1.21),
        "1152x896 (1.29)": (1152, 896, 1.29),
        "1152x832 (1.38)": (1152, 832, 1.38),
        "1216x832 (1.46)": (1216, 832, 1.46),
        "1280x768 (1.67)": (1280, 768, 1.67),
        "1344x768 (1.75)": (1344, 768, 1.75),
        "1344x704 (1.91)": (1344, 704, 1.91),
        "1408x704 (2.0)": (1408, 704, 2.0),
        "1472x704 (2.09)": (1472, 704, 2.09),
        "1536x640 (2.4)": (1536, 640, 2.4),
        "1600x640 (2.5)": (1600, 640, 2.5),
        "1664x576 (2.89)": (1664, 576, 2.89),
        "1728x576 (3.0)": (1728, 576, 3.0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (list(cls.RESOLUTIONS.keys()), {"default": cls.DEF_RESOLUTION}),
            }
        }

    RETURN_TYPES = (WHPipe_t,)
    RETURN_NAMES = ("WHPipe",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, resolution):
        width, height, _ = self.RESOLUTIONS.get(resolution, self.DEF_RESOLUTION)
        return ((width, height,),)

class CONDPipeIn(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": (IO.CONDITIONING, {"tooltip": "Duh."}),
                "negative": (IO.CONDITIONING, {"tooltip": "Duh."}),
            }
        }

    RETURN_TYPES = (CONDPipe_t,)
    RETURN_NAMES = ("CONDPipe",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, positive, negative):
        return ((positive,negative,),)

class CONDPipeOut(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CONDPipe": (CONDPipe_t, {"tooltip": "Duh."}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING, IO.CONDITIONING,)
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, CONDPipe):
        p, n = CONDPipe
        return p, n

class RGBIn(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "R": (IO.INT, {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Red."}),
                "G": (IO.INT, {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Green."}),
                "B": (IO.INT, {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Blue."}),
            }
        }

    RETURN_TYPES = ("RGB",)
    RETURN_NAMES = ("rgb",)
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    def run(self, R, G, B):
        return [R,G,B],