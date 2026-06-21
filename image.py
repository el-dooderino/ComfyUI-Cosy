import hashlib
import os
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from .defs import WHPipe_t, COSY_CATEGORY
import comfy.utils
import comfy.model_management as mm
import folder_paths
from nodes import LoadImage, LoadImageMask
import torch.nn.functional as F
import torch

def resize_image(image, width, height, method, condition, interpolation="nearest"):
    _, oh, ow, _ = image.shape
    x = y = x2 = y2 = 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    if method == 'keep proportion' or method == 'pad':
        ratio = min(width / ow, height / oh)
        new_width = round(ow * ratio)
        new_height = round(oh * ratio)

        if method == 'pad':
            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

        width = new_width
        height = new_height
    elif method.startswith('fill'):
        ratio = max(width / ow, height / oh)
        new_width = round(ow * ratio)
        new_height = round(oh * ratio)
        x = (new_width - width) // 2
        y = (new_height - height) // 2
        x2 = x + width
        y2 = y + height
        if x2 > new_width:
            x -= (x2 - new_width)
        if x < 0:
            x = 0
        if y2 > new_height:
            y -= (y2 - new_height)
        if y < 0:
            y = 0
        width = new_width
        height = new_height

    # else keep W/H as-is

    if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or (
            "upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or (
            "smaller area" in condition and (oh * ow < height * width)):

        outputs = image.permute(0, 3, 1, 2)

        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if method == 'pad':
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0, 2, 3, 1)

        if method.startswith('fill'):
            if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                outputs = outputs[:, y:y2, x:x2, :]
    else:
        outputs = image

    outputs = torch.clamp(outputs, 0, 1)

    return outputs, (outputs.shape[2], outputs.shape[1],)

def empty_latent(width, height, batch_sz):
    latent = torch.zeros([batch_sz, 4, height // 8, width // 8], device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    return {"samples": latent, "downscale_ratio_spacial": 8}


def vae_encode(vae, image, batch_size: int = 1):
    if batch_size > 1:
        # Repeat the image along the batch dimension
        image = image.repeat(batch_size, 1, 1, 1)
    t = vae.encode(image)
    return {"samples": t}

class MaybeLoadImage(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ([], {"image_upload": True})
            },
            "optional": {
                "enabled": (IO.BOOLEAN, {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "input_image_path": (IO.STRING,),
            }
        }

    CATEGORY = COSY_CATEGORY
    RETURN_TYPES = (IO.IMAGE, IO.MASK, IO.BOOLEAN, IO.STRING, WHPipe_t,)
    RETURN_NAMES = ("image", "mask", "enabled", "filename", "WHPipe",)
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, image, enabled=True, input_image_path="", **kwargs):
        if not enabled: return False

        try:
            image_path = folder_paths.get_annotated_filepath(image) or input_image_path.strip()
            if os.path.exists(image_path):
                mtime = os.path.getmtime(image_path)
                return f"{image_path}:{mtime}"
            else:
                return float("NaN")  # Force re-run if file missing
        except Exception:
            return float("NaN")  # Force re-run on any error

    def run(self, image, enabled=True, input_image_path=""):
        if not enabled: return None, None, enabled, "", (0, 0,)
        image_path = folder_paths.get_annotated_filepath(image) or input_image_path.strip()
        output_image, output_mask = self.load_image(image_path)
        return output_image, output_mask, enabled, image, (output_image.shape[2], output_image.shape[1],)

class MaybeImgToLatent(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "WHPipe": (WHPipe_t, {"tooltip": "Duh."}),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
            },
            "optional": {
                "image": (IO.IMAGE, {}),
                "VAE": (IO.VAE, {"tooltip": "Duh."}),
                "batch_size": (IO.INT, {"default": 1, "min": 1, "max": 100}),
            }
        }

    CATEGORY = COSY_CATEGORY
    RETURN_TYPES = (IO.IMAGE, IO.LATENT, WHPipe_t,)
    RETURN_NAMES = ("image", "LATENT", "WHPipe",)
    FUNCTION = "run"
    DESCRIPTION = """If image is given:
       - Will resize as specified and return resized image.
       - If VAE is given: will also generate latent of this image.
    If image not given: will return an empty latent of specified size."""

    @classmethod
    def IS_CHANGED(cls, WHPipe, interpolation, method, condition, image=None, VAE=None, batch_size=1, **kwargs):
        try:
            if image is not None:
                if isinstance(image, torch.Tensor):
                    img_sig = (image.shape, round(float(image.mean()), 6), round(float(image.std()), 6),)
                else:
                    img_sig = str(image)

                key = f"img|{WHPipe}|{interpolation}|{method}|{condition}|{batch_size}|{img_sig}"
            else:
                key = f"empty|{WHPipe}|{batch_size}"

            return hashlib.md5(key.encode()).hexdigest()

        except Exception:
            return float("NaN")

    def run(self, WHPipe, interpolation="nearest", method="stretch", condition="always", image= None, VAE=None, batch_size=1):
        width, height = WHPipe
        if width <= 0 or height <= 0:
            raise ValueError("MaybeImgToLatent: bail on invalid W/H")

        if image is None:
            return None, empty_latent(width, height, batch_size), WHPipe

        image, WHPipe = resize_image(image, width, height, method, condition, interpolation)

        return image, vae_encode(VAE, image, batch_size) if VAE else None, WHPipe
