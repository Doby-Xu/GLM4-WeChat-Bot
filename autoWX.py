# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from wxauto import *
import time

from model import MyGLM4, MyDeepSeek

from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Auto WeChat')
parser.add_argument('--v', action='store_true', help='use vision model')
parser.add_argument('--top_k', type=int, default=5, help='top k')

args = parser.parse_args()

if args.v:
    print("use vision model")
else:
    print("not use vision model")

model_path = "./glm-4-9b-chat-int4" if not args.v else "./glm-4v-9b-chat-int4"

system_prompt = "你是Doby，刚刚从上海交通大学毕业，即将到复旦大学读研究生，从事可信人工智能的研究。下面的对话来自于你的小伙伴，你要好好地回复他，最好保持精简、一句话回复。"

listen_list = [
    '文件传输助手',
    '干拉 白给 刷抖音',
]
multi_user_system_prompt = {
    '文件传输助手': "你是人工智能助手，你的任务是帮助用户解决问题，你可以回答用户的问题，也可以和用户聊天。",
    
}
device = "cuda" 

print("system_prompt:", system_prompt)
# model = MyGLM4(
#     model_path = model_path,
#     max_new_tokens = 512,
#     do_sample = True,
#     top_k = 5,
#     system_prompt = system_prompt,
#     device = device,
#     multi_user_list = listen_list,
#     multi_user_system_prompt = multi_user_system_prompt,
#     vision = args.v
# )
model = MyDeepSeek(
    system_prompt=system_prompt,
    multi_user_list=listen_list,
    multi_user_system_prompt=multi_user_system_prompt,
)




wx = WeChat()


# 循环添加监听对象
for i in listen_list:
    print(f"添加监听对象{i}")
    wx.AddListenChat(who=i, savepic=True if args.v else False)

# 持续监听消息，并且收到消息后回复“收到”
wait = 1  # 设置1秒查看一次是否有新消息0

# 1分钟没有新的对话则释放对话内存
# 为每个用户维护一个计数器
chat_release_time = 60
release_count_list = {}

for user_id in listen_list:
    release_count_list[user_id] = 0

def memory_check():
    for user_id in listen_list:
        if release_count_list[user_id] >= chat_release_time:
            print(f"释放{user_id}的对话内存")
            model.release_chat_memory(user_id)
            release_count_list[user_id] = 0


print("开始监听")

img = None


while True:
    msgs = wx.GetListenMessage()

    for chat in msgs:
        who = chat.who              # 获取聊天窗口名（人或群名）
        one_msgs = msgs.get(chat)   # 获取消息内容
        # 回复收到
        for msg in one_msgs:
            # print("收到消息")

            msgtype = msg.type       # 获取消息类型
            content = msg.content    # 获取消息内容，字符串类型的消息内容
            print(f'【{who}】：{content}')
            if content.startswith('D:\LLM\wxauto文件\微信图片'):
                if args.v:
                    print("收到图片")
                    img = Image.open(content).convert('RGB')
                    # 将图片给入模型
                    # model.add_image(img, who)
                continue

            # 清除who的计数器
            release_count_list[who] = 0

            if img is not None:
                print("该次访问包含一张图片")
                # 由于GLM4只只支持一张图片，所以读入图片时要清空对话缓存
                model.release_chat_memory(who)

            if who ==  "干拉 白给 刷抖音":
                # print("收到"+who+"的消息:"+content)
                if "有无" in content and msgtype == 'friend':
                    chat.SendMsg("🤣包有的")    
                if "何时" in content and msgtype == 'friend':
                    chat.SendMsg("🤣等不了了，就现在")
                if "doby" not in content and "Doby" not in content:
                    continue
                else:
                    if msgtype == 'friend' or (msgtype == 'self' and "🤣" not in content):
                        query = content
                        response = model.get_response(query, who, img=img)
                        chat.SendMsg("🤣"+response)
            elif who == "Framehehe":
                if msgtype == 'friend':
                    query = content
                    response = model.get_response(query, who, img=img)
                    chat.SendMsg("🤣"+response)
            elif who == "文件传输助手":
                print("收到"+who+"的消息:"+content)
                if "🤣" in content or msgtype == 'time':
                    continue
                query = content
                response = model.get_response(query, who, img=img)
                chat.SendMsg("🤣"+response)
            else:
                if msgtype == 'friend':
                    query = content
                    response = model.get_response(query, who, img=img)
                    chat.SendMsg(response)
            img = None

           

    time.sleep(wait)
    # 维护计数器
    for user_id in listen_list:
        release_count_list[user_id] += wait
    memory_check()
    # 显存检查，获取以GB为单位的显存使用情况
    # gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
    # # release memory if gpu memory is over 14GB
    # if gpu_memory > 14:
    #     print(f"显存使用超过14GB，释放所有对话内存")
    #     for user_id in listen_list:
    #         model.release_chat_memory(user_id)
    #         release_count_list[user_id] = 0