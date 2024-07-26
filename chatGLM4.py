import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import MyGLM4

device = "cuda"

system_prompt = "你是一个人工智能助手，请认真回答下面的问题"

model = MyGLM4(
    model_path = "./glm-4-9b-chat-int4",
    max_new_tokens = 512,
    do_sample = True,
    top_k = 5,
    system_prompt = system_prompt,
    device = device
)



# while no keyboard interrupt
try:
    while True:
        query = input(">>> ")
        response = model.get_response(query)
        print("GLM4: ", response[1:])
except KeyboardInterrupt:
    pass