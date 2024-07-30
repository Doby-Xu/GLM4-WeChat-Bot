# 微信GLM4

一个简单的调用GLM4的微信自动回复机器人。

## 使用前，你需要...

- have some knowledge about 中文. But since you are using WeChat, I assume you are a Chinese speaker or at least you can somehow read Chinese.

- 是一个想入门LLM的小白，否则这个仓库对于你来说过于简单了

- 有一台配置了基本Python环境的电脑，以及12G以上显存的显卡，并且确保C盘有几十G可以折腾的空间

- 根据以下两个仓库配置环境：[wxauto](https://github.com/cluic/wxauto)和[GLM4](https://github.com/THUDM/GLM-4)。事实上，本仓库是在这两个仓库的基础上的简单融合。至于有多简单，这里都没有文件结构。你可以直接使用这两个库来搭建自己酷酷的项目，当然也可以参考本仓库作为入门demo

## Run

### 模型量化

搭建好环境后，首先运行：
    
```shell
    python model.py

```

会自行下载GLM4模型，将其量化为int4，并存储在当前目录下。

这个过程会在你的C盘/用户/.cache/huggingface下生成一个glm4文件夹，大小约为20G，完事后可以删掉他，因为有一个6.58G的int4量化模型存在了当前目录下。

如果成功下载了模型，却显示以下报错：
```shell
DefaultCPUAllocator: not enough memory: you tried to allocate xxxxxxxxx bytes.
```
那可能是C盘要爆了（当然也可能是内存真的不够）。你可以把C盘的glm4文件夹移动到当前文件夹下，然后把`model.py`中101行取消注释：
    
```python
    model = AutoModelForCausalLM.from_pretrained(
        "./glm-4-9b-chat",
101 ===># cache_dir="D:\\LLM",
        # torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=True
    ).eval()
```
再次运行此文件，成功载入模型将其量化为int4。


### 微信机器人

在`autoWX.py`中，你需要修改以下内容：

- `system_prompt`：这是LLM的系统提示词，他一定程度上决定了模型的人格。你可以自己定义一个，或者直接使用`"你是GLM4，一个人工智能助手"`。
- `listen_list`：这是一个列表，里面存放了你想让机器人监听的好友的昵称。你可以在这里添加你的好友昵称，也可以是“文件传输助手”或任意群。只填写你在你微信中看到的好友/群备注即可。
- `multi_user_system_prompt`：这是一个字典，可以为不同的微信用户设置不同的系统提示词。如果你只有一个微信用户。不设置的话，则会直接使用`system_prompt`。

然后运行：

```shell
    python autoWX.py
```

### 个性化

在`autoWX.py`的73行处，有大量if else语句，可以根据其示例，自行添加更多的回复规则，进行提示词工程等等。这里只是一个简单的示例，可以根据自己的需求，添加更多的回复规则。

### GLM4聊天

运行：
    
```shell
    python chatGLM4.py
```
可以直接和GLM4进行对话。

## 关于wxauto

微信陆续封掉了几乎所有API，现在微信自动化的合法途径貌似只有企业微信机器人，和键鼠输入。这个库是貌似是基于键盘输入的，他会将回复拷贝到剪切板里，粘贴到聊天框里，按enter发送出去。如果在这个关键时刻你操纵了键盘，会有不可预知的后果。所以请不要在运行时操纵键盘。

另外，长时间自动回复可能会被微信强制下线。

## 微调

推荐使用LLAMA-Factory进行微调。

如果有24G显存，可以进行LoRA微调。如果只有16G显存，可以进行量化的QLoRA微调

微调结束后，使用`merge_GLM.py`将微调后的模型和GLM4模型合并，得到一个新的模型

推荐使用[MemoTrace](https://memotrace.cn/doc/)来导出微信聊天记录，构建数字分身。但是诚挚建议在微调之前，优先考虑使用提示词工程。