from . import qwen3_vl_node, qwen3_vl_prompt_select, clip, whatnots, image, sampling, controlnet
from .defs import COSY_CATEGORY

NODE_CLASS_MAPPINGS = {
    f"{COSY_CATEGORY} Qwen3VL_Node": qwen3_vl_node.Qwen3VL_Node,
    f"{COSY_CATEGORY} Qwen3VL_Prompt": qwen3_vl_prompt_select.Qwen3VL_PromptSelect,
    f"{COSY_CATEGORY} CLIP_Text_Enc_Scaled": clip.CLIPTextEncodeScaled,
    f"{COSY_CATEGORY} COND_combine": clip.CondCombiner,
    f"{COSY_CATEGORY} WHPipe_In": whatnots.WHPipeIn,
    f"{COSY_CATEGORY} WHPipe_Out": whatnots.WHPipeOut,
    f"{COSY_CATEGORY} CONDPipe_In": whatnots.CONDPipeIn,
    f"{COSY_CATEGORY} CONDPipe_Out": whatnots.CONDPipeOut,
    f"{COSY_CATEGORY} Maybe_LoadImage": image.MaybeLoadImage,
    f"{COSY_CATEGORY} Maybe_ImgToLatent": image.MaybeImgToLatent,
    f"{COSY_CATEGORY} SDXL Resolution": whatnots.SDXL_ResolutionPicker,
    f"{COSY_CATEGORY} Load_Checkpoint": sampling.LoadCheckpoint,
    f"{COSY_CATEGORY} Sampler": sampling.Sampler,
    f"{COSY_CATEGORY} Sampler_Ref": sampling.SamplerRefiner,
    f"{COSY_CATEGORY} Sampler_CF": sampling.SamplerCF,
    f"{COSY_CATEGORY} Sampler_CF_Adv": sampling.SamplerCFAdv,
    f"{COSY_CATEGORY} ControlNet_CF": controlnet.ControlNetRC,
    f"{COSY_CATEGORY} ControlNet_Enabled": controlnet.ControlNet_enabled,
    f"{COSY_CATEGORY} Maybe_ControlNet": controlnet.MaybeApplyControlNet,
    f"{COSY_CATEGORY} RGB_In": whatnots.RGBIn,
    f"{COSY_CATEGORY} Mask_CF": image.MaskRC,
    f"{COSY_CATEGORY} Maybe_Mask": image.MaybeMask,
}

#not sure what the purpose of double-remapping is except spending more time guessing what the final name of the node might be.
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


