# test.py
import torch
from PIL import Image
# import deepspeed
from transformers import AutoModel, AutoTokenizer

try:
    import deepspeed
except ImportError:
    raise ImportError("This modeling file requires the following packages that were not found in your environment: deepspeed. Run `pip install deepspeed`")

model = AutoModel.from_pretrained(r'D:\ollama_models\MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
model = model.to(device='cuda', dtype=torch.bfloat16)
# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
#model = model.to(device='cuda', dtype=torch.float16)
# For Mac with MPS (Apple silicon or AMD GPUs).
# Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py`
#model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(r'D:\ollama_models\MiniCPM-V-2', trust_remote_code=True)
model.eval()

image = Image.open('./image1.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]

res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)
print(res)
