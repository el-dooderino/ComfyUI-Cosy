from .defs import COSY_CATEGORY, _mk_hash_key, _hash_tensor
import os
from os.path import basename
import uuid
import re
import folder_paths
import numpy as np
import torch
import gc
from comfy.comfy_types.node_typing import IO, ComfyNodeABC

from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from pathlib import Path

# Model directory
model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)

def scan_local_vlm_models():
    available_models = []
    try:
        if os.path.exists(model_directory):
            local_items = os.listdir(model_directory)
            for item in local_items:
                available_models.append(item)
    finally:
        return available_models if len(available_models) > 0 else ["Huihui-Qwen3-VL-8B-Instruct-abliterated"]

DEF_SYS_PROMPT = "You are a lazy but friendly deadbeat who always have a reason or 2 to refuse the user's request."
DEF_USER_PROMPT = "Find a reason to refuse this request."

# ============================================================================
# Helper Functions
# ============================================================================

def temp_image(image):
    unique_id = uuid.uuid4().hex
    image_path = Path(folder_paths.temp_directory) / f"temp_image_{unique_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    img.save(str(image_path))
    return f"file://{image_path.as_posix()}", image_path

class Qwen3VL_Node(ComfyNodeABC):
    def __init__(self):
        # Instance-level cache
        self._cache = {}   # Will store: {"key": ..., "result": ..., "u_prompt": ..., "s_prompt": ...}
        self._model = None
        self._model_params = None # {"path", "attention"}

    @classmethod
    def INPUT_TYPES(cls):
        models = scan_local_vlm_models()

        return {
            "required": {
                "image": (IO.IMAGE,),
                "model_name": (models, {"default": models[0]}),
                "attention": (["flash_attention_2", "sdpa", "eager"], {"default": "sdpa"}),
                "max_out_tokens": (IO.INT, {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64
                }),
                # VRAM Management
                "unload_when_done": (IO.BOOLEAN, {
                    "default": True,
                    "label_on": "Unload",
                    "label_off": "Keep",
                }),
                # Internal cache toggle
                "use_cache": (IO.BOOLEAN, {
                    "default": True,
                    "label_on": "Use",
                    "label_off": "Unuse",
                }),
            },
            "optional": {
                "user": (IO.STRING, {"forceInput": True,}),
                "system": (IO.STRING, {"forceInput": True,}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING, IO.STRING)
    RETURN_NAMES = ("result", "user", "system")
    FUNCTION = "run"
    CATEGORY = COSY_CATEGORY

    @classmethod
    def IS_CHANGED(cls, image, model_name, attention, max_out_tokens, unload_when_done, use_cache, user=None, system=None, **kwargs):
        img_sig = None
        if image is not None:
            if isinstance(image, torch.Tensor): img_sig = _hash_tensor(image)
            else: img_sig = str(image)
        return _mk_hash_key(model_name, attention, max_out_tokens, user, system, img_sig)

    def run(self, image, model_name, attention, max_out_tokens, unload_when_done, use_cache, user=None, system=None,):
        """The orchestrator """
        try:
            user = user.strip() if user else ""
            system = system.strip() if system else ""
            user, system = user or DEF_USER_PROMPT, system or DEF_SYS_PROMPT

            if image is None:
                self._cache = {}
                return None, user, system

            # === CACHING ===
            cache_key = None
            if use_cache:
                cache_key = _mk_hash_key(_hash_tensor(image), user, system, model_name, attention, max_out_tokens)
                if self._cache.get("key") == cache_key:
                    print("✅ Cache hit")
                    return self._cache["result"], self._cache["u_prompt"], self._cache["s_prompt"]


            # === CACHE MISS → Run model ===
            print("⚡ Running Qwen3-VL model...")
            self._maybe_load_model(model_name, attention)
            result = self._do_inference(image, user, system, max_out_tokens)

            # === STORE IN CACHE ===
            if cache_key is not None:
                self._cache = {
                    "key": cache_key,
                    "result": result,
                    "u_prompt": user,
                    "s_prompt": system,
                }

            return result, user, system

        finally:
            if unload_when_done:
                self._maybe_unload_model()
            else:
                print("ℹ️  Model kept in VRAM")


    def _do_inference(self, image, user_prompt, system_prompt, max_out_tokens):
        if image.shape[0] != 1: raise ValueError("Qwen3VL_Node currently supports only a single image.")
        from qwen_vl_utils import process_vision_info

        # Pixel calculations
        min_px = 256 * 28 * 28
        max_px = 1280 * 28 * 28
        total_px = 20480 * 28 * 28

        processor = AutoProcessor.from_pretrained(self._model_params["path"])

        img_path = None
        try: # try/finally for the temp image file deletion.
            uri, img_path = temp_image(image)
            content = [
                {"type": "image", "image": uri, "min_pixels": min_px, "max_pixels": max_px, "total_pixels": total_px},
                {"type": "text", "text": user_prompt}
            ]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]


            modeltext = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Handle different API versions
            video_kwargs = {}
            try:
                image_inputs, video_inputs = process_vision_info(messages)
            except TypeError:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            inputs = processor(
                text=[modeltext],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(next(self._model.parameters()).device)

            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_out_tokens,
    #            no_repeat_ngram_size=4,
                do_sample=False,
                repetition_penalty=1.1,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            result = str(output_text[0])
            if "</think>" in result:
                result = result.split("</think>")[-1]
            result = re.sub(r"^[\s\u200b\xa0]+", "", result)

            print(f"📝 Generated prompt: {result[:100]}...")

            return result

        finally:
            if img_path is not None:
                try:
                    img_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"⚠️ Failed to delete temp image {img_path}: {e}")

    def _maybe_load_model(self, model: str, attention: str):
        path_list = model.rsplit("/", 1)
        if len(path_list) < 1:
            raise RuntimeError(f"bad model name: {model}")

        model_name = path_list[-1]

        model_path = os.path.join(model_directory, model_name)
        if self._model is not None and self._model_params is not None:
            ep = self._model_params.get("path")
            if (ep == model_path and
                self._model_params.get("attention") == attention):
                print(f"✅ Reusing already loaded model {basename(ep)}")
                return

            # Different model/attention requested → unload old one
            print(f"🔄 Switching model, unloading {basename(ep)}")
            self._maybe_unload_model()


        # Download if not exists
        if not os.path.exists(model_path):
            print(f"📥 Downloading model: {model}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )

        # Load the model (new or after unload)
        print(f"🔧 Loading model: {model_name} | Attention: {attention}")
        try:
            loaded_model = self._load_qwen_model(model_path, attention)
        except Exception:
            self._model = None
            self._model_params = None
            raise

        self._model = loaded_model
        self._model_params = {"path": model_path, "attention": attention}

    def _load_qwen_model(self, model_path: str, attention: str):
        print(f"Loading from: {model_path}")
        try:
            return Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation=attention,
            )
        except Exception as e:
            print(f"❌ Failed to load Qwen3-VL model from {model_path}: {e}")
            raise

    def _maybe_unload_model(self):
        if self._model is None: return
        try:
            self._model = None
            self._model_params = None
            # Release CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        finally:
            gc.collect()
