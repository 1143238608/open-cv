# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(r'D:\ollama_models\MiniCPM-Llama3-V-2_5', trust_remote_code=True,
                                  torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained(r'D:\ollama_models\MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

image = Image.open('./image1.jpg').convert('RGB')
question = '这张图片里边有什么?'
msgs = [{'role': 'user', 'content': question}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,  # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
