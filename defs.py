import hashlib
import torch

# For CU
WHPipe_t = "WHPipe"
CONDPipe_t = "CONDPipe"
COSY_CATEGORY = "Cosy"

def _mk_hash_key(*args):
    key = "|"
    for _, arg in enumerate(args): key += f"{arg}|"
    return hashlib.md5(key.encode()).hexdigest()

def _hash_tensor(tensor: torch.Tensor) -> str:
    if tensor is None: return "None"
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

