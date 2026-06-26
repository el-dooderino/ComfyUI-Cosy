import os
import numpy
import scipy
import torchvision.transforms.v2 as T2
from PIL import Image, ImageFilter

from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from .defs import WHPipe_t, COSY_CATEGORY, _mk_hash_key, _hash_tensor
import comfy.utils
import comfy.model_management as mm
import folder_paths
from nodes import LoadImage
import torch.nn.functional as F
import torch

METHODS = ["stretch", "keep proportion", "fill / crop", "pad"]
INTERPOLATIONS = ["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"]
CONDITIONS = ["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"]
FLIPS = ["don't", "vertical", "horizontal"]
ROTATOES = ["0", "90", "180", "270"]

def resize_image(image, width, height, method, condition, interpolation="nearest"):
    _, oh, ow, _ = image.shape
    x = y = x2 = y2 = 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    if method == METHODS[1] or method == METHODS[3]: # keep proportion or pad
        ratio = min(width / ow, height / oh)
        nw, nh = round(ow * ratio), round(oh * ratio)

        if method == METHODS[3]: # pad
            pad_left = (width - nw) // 2
            pad_right = width - nw - pad_left
            pad_top = (height - nh) // 2
            pad_bottom = height - nh - pad_top

        width, height = nw, nh

    elif method == METHODS[2]: # fill / crop
        ratio = max(width / ow, height / oh)
        nw, nh = round(ow * ratio), round(oh * ratio)
        x, y = (nw - width) // 2, (nh - height) // 2
        x2, y2 = x + width, y + height
        if x2 > nw: x -= (x2 - nw)
        if x < 0: x = 0
        if y2 > nh: y -= (y2 - nh)
        if y < 0: y = 0

        width, height = nw, nh

    # else keep W/H as-is

    if (condition == CONDITIONS[0] #always
        or (condition == CONDITIONS[1] and (oh > height or ow > width)) # downscale if bigger
        or (condition == CONDITIONS[2] and (oh < height or ow < width)) # upscale if smaller
        or (condition == CONDITIONS[3] and (oh * ow > height * width))  # bigger area
        or (condition == CONDITIONS[4] and (oh * ow < height * width))): # smaller area

        outputs = image.permute(0, 3, 1, 2)

        if interpolation == INTERPOLATIONS[5]: # lanczos
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if method == METHODS[3]: # pad
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0, 2, 3, 1)

        if method == METHODS[2]: # fill/crop
            if x > 0 or y > 0 or x2 > 0 or y2 > 0: outputs = outputs[:, y:y2, x:x2, :]
    else:
        outputs = image

    outputs = torch.clamp(outputs, 0, 1)

    return outputs, (outputs.shape[2], outputs.shape[1],)

# rotate by 90 degrees * rotation.
def rotate_hw_by_90(tensor, rotation: str | int):
    if isinstance(rotation, str):
        try:
            rotation = ROTATOES.index(rotation)
        except ValueError:
            return tensor

    return torch.rot90(tensor, k=rotation, dims=[2, 1])

def flip_hw(tensor, flip_method: str):
    if flip_method == FLIPS[1]: return torch.flip(tensor, dims=[1]) # vertical
    if flip_method == FLIPS[2]: return torch.flip(tensor, dims=[2]) # horizontal
    return tensor

def empty_latent(width, height, batch_sz):
    latent = torch.zeros([batch_sz, 4, height // 8, width // 8], device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    return {"samples": latent, "downscale_ratio_spacial": 8}

def vae_encode(vae, image, batch_size: int = 1):
    if batch_size > 1: image = image.repeat(batch_size, 1, 1, 1)
    return {"samples": vae.encode(image)}


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class MaybeLoadImage(LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ([], {"image_upload": True}),
                "flip": (FLIPS, {"default": FLIPS[0]}),
                "rotate": (ROTATOES, {"default": ROTATOES[0]}),
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
    def IS_CHANGED(cls, image, flip, rotate, enabled=True, input_image_path="", **kwargs):
        if not enabled: return False

        try:
            image_path = folder_paths.get_annotated_filepath(image) or input_image_path.strip()
            if os.path.exists(image_path):
                mtime = os.path.getmtime(image_path)
                return _mk_hash_key(image_path,mtime,flip,rotate)
            else:
                return float("NaN")  # Force re-run if file missing
        except Exception:
            return float("NaN")  # Force re-run on any error

    def run(self, image, flip, rotate, enabled=True, input_image_path=""):
        if not enabled: return None, None, enabled, "", (0, 0,)
        image_path = folder_paths.get_annotated_filepath(image) or input_image_path.strip()
        output_image, output_mask = self.load_image(image_path)

        output_image = flip_hw(output_image, flip)
        output_image = rotate_hw_by_90(output_image, rotate)

        output_mask = flip_hw(output_mask, flip)
        output_mask = rotate_hw_by_90(output_mask, rotate)

        return output_image, output_mask, enabled, image, (output_image.shape[2], output_image.shape[1],)

class MaybeImgToLatent(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "WHPipe": (WHPipe_t, {"tooltip": "Duh."}),
                "flip": (FLIPS, {"default": FLIPS[0], "tooltip": "happens before resizing"}),
                "rotate": (ROTATOES, {"default": ROTATOES[0], "tooltip": "happens before resizing"}),
                "resize_with": (INTERPOLATIONS, {"default": INTERPOLATIONS[0]}),
                "resize_how": (METHODS, {"default": METHODS[0]}),
                "resize_if": (CONDITIONS,{"default": CONDITIONS[0]}),
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
    def IS_CHANGED(cls, WHPipe, flip, rotate, resize_with, resize_how, resize_if, image=None, VAE=None, batch_size=1, **kwargs):
        try:
            if image is not None:
                if isinstance(image, torch.Tensor): img_sig = _hash_tensor(image)
                else: img_sig = str(image)
                return _mk_hash_key(WHPipe, flip, rotate, resize_with, resize_how, resize_if, batch_size, img_sig)
            else:
                return _mk_hash_key(WHPipe,batch_size)

        except Exception:
            return float("NaN")

    def run(self, WHPipe, flip:str, rotate:str, resize_with:str, resize_how:str, resize_if:str, image= None, VAE=None, batch_size=1):
        width, height = WHPipe
        if width <= 0 or height <= 0:
            raise ValueError("MaybeImgToLatent: bail on invalid W/H")

        if image is None:
            return None, empty_latent(width, height, batch_size), WHPipe

        image = flip_hw(image, flip)
        image = rotate_hw_by_90(image, rotate)
        image, WHPipe = resize_image(image, width, height, resize_how, resize_if, resize_with)

        return image, vae_encode(VAE, image, batch_size) if VAE else None, WHPipe

class MaskRC(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expand": (IO.INT, {"default": 0, "min": -256, "max": 256, "step": 1, }),
                "blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, }),
                "use_alpha": (IO.BOOLEAN, {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Use alpha channel for mask if available."}),
            },
        }

    CATEGORY = COSY_CATEGORY
    RETURN_TYPES = ("COSY_MASK_RC",)
    RETURN_NAMES = ("mask_cf",)
    FUNCTION = "run"

    def run(self, expand, blur, use_alpha):
        return (expand, blur, use_alpha),

class MaybeMask(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"tooltip": "Duh.", "lazy": True}),
                "threshold": (IO.INT, {"default": 0, "min": 0, "max": 127, "step": 1, "tooltip": "Threshold is +-"}),
                "enabled": (IO.BOOLEAN, {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
            "optional": {
                "rgb": ("RGB", {"tooltip": "Default is W."}),
                "mask_cf": ("COSY_MASK_RC", {"tooltip": "Duh."}),
            }
        }

    CATEGORY = COSY_CATEGORY
    RETURN_TYPES = (IO.MASK, IO.MASK,)
    RETURN_NAMES = ("mask", "inv_mask",)
    FUNCTION = "run"

    def check_lazy_status(self, image, threshold, enabled=True, rgb=None, mask_cf=None):
        ret = []
        if enabled:
            ret.append("image")
            ret.append("threshold")

        return ret

    @classmethod
    def IS_CHANGED(cls, image, threshold, enabled=True, rgb=None, mask_cf = None, **kwargs):
        if not enabled: return False
        return _mk_hash_key(image, threshold, enabled, rgb, mask_cf)

    def run(self, image, threshold, enabled=True, rgb=None, mask_cf = None):
        if not enabled: return None, None
        if not isinstance(image, torch.Tensor): return None, None

        cf_expand, cf_blur, cf_use_alpha = mask_cf if mask_cf is not None else (0, 0, True)
        # Check for alpha channel (C == 4)
        if image.shape[-1] == 4 and cf_use_alpha:
            mask = image[..., 3].clone()
            # Normalize to [0, 1] if needed
            if mask.max() > 1.0: mask = mask / 255.0
            mask = mask.float()
        else:
            # Default to white if no rgb provided
            color = torch.tensor(rgb, dtype=torch.uint8) if rgb is not None else torch.tensor([255, 255, 255], dtype=torch.uint8)
            # Convert image to 0-255 int
            ti = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)

            # Create per-channel boolean mask
            lb = (color - threshold).clamp(min=0).view(1, 1, 1, 3)
            ub = (color + threshold).clamp(max=255).view(1, 1, 1, 3)
            mask = (lb <= ti) & (ti <= ub)  # shape: [B, H, W, 3]
            # Reduce to single channel mask (True where all 3 channels match)
            mask = mask.all(dim=-1).float()  # shape: [B, H, W]

        if mask_cf is not None:
            result = []
            for m in mask: # m is [H, W]
                if cf_expand != 0:
                    sz = abs(cf_expand)
                    m = m.cpu().numpy()
                    if cf_expand > 0: #expand the black area by erosion.
                        m = scipy.ndimage.grey_erosion(m, size=(sz, sz))
                    else: # collapse the black area by dilation.
                        m = scipy.ndimage.grey_dilation(m, size=(sz, sz))

                    m = torch.from_numpy(m).float()

                if cf_blur > 0:
                    m = pil2tensor(tensor2pil(m.cpu().detach()).filter(ImageFilter.GaussianBlur(cf_blur))).float()

                result.append(m.float())

            mask = torch.stack(result, dim=0).to(image.device).float()

        return mask, 1 - mask
