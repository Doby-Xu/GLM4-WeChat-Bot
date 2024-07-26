import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wxauto import *
import time

from model import MyGLM4

system_prompt = "你是Doby，刚刚从上海交通大学毕业，即将到复旦大学读研究生，从事可信人工智能的研究。下面的对话来自于你的小伙伴，你要好好地回复他，最好保持精简、一句话回复。"

listen_list = [
    '文件传输助手',
    '干拉 白给 刷抖音',
]

device = "cuda" 

print("system_prompt:", system_prompt)
model = MyGLM4(
    model_path = "./glm-4-9b-chat-int4",
    max_new_tokens = 512,
    do_sample = True,
    top_k = 5,
    system_prompt = system_prompt,
    device = device,
    multi_user_list = listen_list
)



wx = WeChat()


# 循环添加监听对象
for i in listen_list:
    print(f"添加监听对象{i}")
    wx.AddListenChat(who=i, savepic=False)

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

            # 清除who的计数器
            release_count_list[who] = 0

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
                        response = model.get_response(query, who)
                        chat.SendMsg("🤣"+response[1:])
            elif who == "Framehehe":
                if msgtype == 'friend':
                    query = content
                    response = model.get_response(query, who)
                    chat.SendMsg("🤣"+response[1:])
            elif who == "文件传输助手":
                print("收到"+who+"的消息:"+content)
                if "🤣" in content:
                    continue
                query = content
                response = model.get_response(query, who)
                chat.SendMsg("🤣"+response[1:])
            else:
                if msgtype == 'friend':
                    query = content
                    response = model.get_response(query, who)
                    chat.SendMsg(response[1:])

           

    time.sleep(wait)
    # 维护计数器
    for user_id in listen_list:
        release_count_list[user_id] += wait
    memory_check()
    # 显存检查，获取以GB为单位的显存使用情况
    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
    # release memory if gpu memory is over 14GB
    if gpu_memory > 14:
        print(f"显存使用超过14GB，释放所有对话内存")
        for user_id in listen_list:
            model.release_chat_memory(user_id)
            release_count_list[user_id] = 0