from . import qwen3_vl_node, qwen3_vl_prompt_select, clip, whatnots, image, sampling, controlnet

NODE_CLASS_MAPPINGS = {
    "Qwen3VL_Node": qwen3_vl_node.Qwen3VL_Node,
    "Qwen3VL_Prompt": qwen3_vl_prompt_select.Qwen3VL_PromptSelect,
    "CLIP_Text_Enc_Scaled": clip.CLIPTextEncodeScaled,
    "COND_combine": clip.CondCombiner,
    "WHPipe_In": whatnots.WHPipeIn,
    "WHPipe_Out": whatnots.WHPipeOut,
    "CONDPipe_In": whatnots.CONDPipeIn,
    "CONDPipe_Out": whatnots.CONDPipeOut,
    "Maybe_LoadImage": image.MaybeLoadImage,
    "Maybe_ImgToLatent": image.MaybeImgToLatent,
    "Resolution_Picker": whatnots.ResolutionPicker,
    "Cosy_Load_Checkpoint": sampling.LoadCheckpoint,
    "Cosy_Sampler": sampling.Sampler,
    "Cosy_Sampler_Ref": sampling.SamplerRefiner,
    "Cosy_Sampler_CF": sampling.SamplerCF,
    "Cosy_Sampler_CF_Adv": sampling.SamplerCFAdv,
    "ControlNet_CF": controlnet.ControlNetRC,
    "ControlNet_Enabled": controlnet.ControlNet_enabled,
    "Maybe_ApplyControlNet": controlnet.MaybeApplyControlNet,
    "RGB_In": whatnots.RGBIn,
    "Mask_CF": image.MaskRC,
    "Maybe_Mask": image.MaybeMask,
}

#not sure what the purpose of double-remapping is except spending more time guessing what the final name of the node might be.
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


