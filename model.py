import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from volcenginesdkarkruntime import Ark
import os

class MyGLM4():
    '''
    GLM4 model
    '''
    def __init__(
            self,
            model_path: str = "./glm-4-9b-chat-int4",
            max_new_tokens: int = 512,
            do_sample: bool = True,
            top_k: int = 5,
            vision: bool = False,
            system_prompt = "你是一个人工智能助手，请认真回答下面的问题",
            device: str = "cuda" ,
            multi_user_list: list = [],
            multi_user_system_prompt: dict = {}
            ) -> None:
        
        # Build model and tokenizer
        print("Loading model and tokenizer...")
        # tokenizer_path = "THUDM/glm-4-9b-chat" if not vision else "THUDM/glm-4v-9b"
        tokenizer_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("Building Model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True,
            torch_dtype=torch.float16
        ).eval()
        
        

        # Set generation kwargs
        self.device = device
        self.gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "top_k": top_k}
        self.system_prompt = system_prompt

        # Multi user setting. For WeChat listening, message from different friends should be stored separately
        self.multi_user_flag = False if len(multi_user_list) == 0 else True
        self.multi_user_list = multi_user_list
        self.list_memory = {}

        if self.multi_user_flag:
            for user_id in multi_user_list:
                self.list_memory[user_id] = []
                if user_id in multi_user_system_prompt:
                    self.list_memory[user_id].append({"role": "system", "content": multi_user_system_prompt[user_id]})
                    print("assign custom system prompt for user:", user_id)
                else:
                    self.list_memory[user_id].append({"role": "system", "content": system_prompt})
        else:
            self.list_memory = []
            self.list_memory.append({"role": "system", "content": system_prompt})


    def get_response(self, query, user_id = None, img = None):
        # For WeChat listening, message from different friends should be stored separately
        if self.multi_user_flag:
            if img is not None:
                self.list_memory[user_id].append({"role": "user", "image": img, "content": query})
            else: 
                self.list_memory[user_id].append({"role": "user", "content": query})
            inputs = self.tokenizer.apply_chat_template(self.list_memory[user_id],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True)

            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # add response to inputs_querys
                self.list_memory[user_id].append({"role": "assistant", "content": response})
                return response
        # For single user
        else:
            if img is not None:
                self.list_memory.append({"role": "user", "image": img, "content": query})
            else:
                self.list_memory.append({"role": "user", "content": query})
            inputs = self.tokenizer.apply_chat_template(self.list_memory,
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True)

            inputs = inputs.to(self.device)

            with torch.no_grad():
                if img is not None:
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        position_ids=inputs['position_ids'],
                        images=inputs['images'],
                        **self.gen_kwargs
                    )
                else:
                    outputs = self.model.generate(**inputs, **self.gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # add response to inputs_querys
                self.list_memory.append({"role": "assistant", "content": response})
                return response
    # def add_image(self, image, user_id = None):
    #     if self.multi_user_flag:
    #         self.list_memory[user_id].append({"image": image})
    #     else:
    #         self.list_memory.append({"image": image})

    # In case the chat memory accumulates too much, release the memory when necessary
    def release_chat_memory(self, user_id):
        self.list_memory[user_id] = []
        self.list_memory[user_id].append({"role": "system", "content": self.system_prompt})


class MyDoubao():
    '''
    Doubao model
    '''
    def __init__(
            self,
            system_prompt = "你是一个人工智能助手，请认真回答下面的问题",
            temperature: float = 0.5,
            # device: str = "cuda" ,
            multi_user_list: list = [],
            multi_user_system_prompt: dict = {}
            ) -> None:
        
        # Build model and tokenizer
        print("Loading model and tokenizer...")
        self.client = Ark(api_key=os.environ.get("ARK_API_KEY"))
    
        self.model = "<input your model number here>" # something like "ep-20240817xxxxxx-xxxxxx"
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.multi_user_flag = False if len(multi_user_list) == 0 else True
        self.multi_user_list = multi_user_list
        self.list_memory = {}

        if self.multi_user_flag:
            for user_id in multi_user_list:
                self.list_memory[user_id] = []
                if user_id in multi_user_system_prompt:
                    self.list_memory[user_id].append({"role": "system", "content": multi_user_system_prompt[user_id]})
                    print("assign custom system prompt for user:", user_id)
                else:
                    self.list_memory[user_id].append({"role": "system", "content": system_prompt})
        else:
            self.list_memory = []
            self.list_memory.append({"role": "system", "content": system_prompt})
    
    def update_memory_with_chat_history(self, chat_history, who = None):
        # Chat history is a list of list, each item is [role, content]

        

        # Clear list_memory

        
        # Find last "Time" message
        time_index = -1
        for i in range(len(chat_history)-1, -1, -1):
            if chat_history[i][0] == "Time":
                time_index = i
                break
        if time_index == -1:
            print("No Time message found in chat history")
            return
        

        if self.multi_user_flag == False or who is None:
            # Clear list_memory
            self.list_memory = []
            # Set system prompt
            self.list_memory.append({"role": "system", "content": self.system_prompt})

            # Update list_memory with all the messages after the last "Time" message
            for i in range(time_index+1, len(chat_history)):
                if chat_history[i][0] == "Self":
                    self.list_memory.append({"role": "assistant", "content": chat_history[i][1]})
                else:
                    self.list_memory.append({"role": "user", "content": chat_history[i][1]})

        else:
            # Clear list_memory
            self.list_memory[who] = []
            # Set system prompt
            if who in self.multi_user_system_prompt:
                self.list_memory[who].append({"role": "system", "content": self.multi_user_system_prompt[who]})
            else:
                self.list_memory[who].append({"role": "system", "content": self.system_prompt})

            # Update list_memory with all the messages after the last "Time" message
            for i in range(time_index+1, len(chat_history)):
                if chat_history[i][0] == "Self":
                    self.list_memory[who].append({"role": "assistant", "content": chat_history[i][1]})
                else:
                    self.list_memory[who].append({"role": "user", "content": chat_history[i][1]})

        

    def get_response(self, query, user_id = None, img = None):
        # For WeChat listening, message from different friends should be stored separately
        if self.multi_user_flag:
            if img is not None:
                self.list_memory[user_id].append({"role": "user", "image": img, "content": query})
            else: 
                self.list_memory[user_id].append({"role": "user", "content": query})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages = self.list_memory[user_id],
                temperature=0.5,
            )
            response = completion.choices[0].message.content
            self.list_memory[user_id].append({"role": "assistant", "content": response})
            return response

        # For single user
        else:
            if img is not None:
                self.list_memory.append({"role": "user", "image": img, "content": query})
            else:
                self.list_memory.append({"role": "user", "content": query})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages = self.list_memory,
                temperature=0.5,
            )
            response = completion.choices[0].message.content
            self.list_memory.append({"role": "assistant", "content": response})
            return response

            

    def release_chat_memory(self, user_id):
        self.list_memory[user_id] = []
        self.list_memory[user_id].append({"role": "system", "content": self.system_prompt})



class MyBianque():
    '''
    BianQue Model
    '''
    def __init__(
            self,
            system_prompt = "None",
            top_p: float = 0.75,
            temperature: float = 0.95,
            # device: str = "cuda" ,
            multi_user_list: list = [],
            multi_user_system_prompt: dict = {}  
        ):
        # System prompt is not used in BianQue model
        print("Loading model and tokenizer...")
        model_name_or_path = 'scutcyr/BianQue-2'
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.temperature = temperature
        self.top_p = top_p
        self.multi_user_flag = False if len(multi_user_list) == 0 else True
        self.multi_user_list = multi_user_list

        device = "cuda:0"
        self.model.to(device)

        self.list_memory = {}
        if self.multi_user_flag:
            # BianQue model's chat history goes like:
            ''' python
            # 多轮对话调用模型的chat函数
            # 注意：本项目使用"\n病人："和"\n医生："划分不同轮次的对话历史
            # 注意：user_history比bot_history的长度多1
            user_history = ['你好', '我最近失眠了']
            bot_history = ['我是利用人工智能技术，结合大数据训练得到的智能医疗问答模型扁鹊，你可以向我提问。']
            # 拼接对话历史
            context = "\n".join([f"病人：{user_history[i]}\n医生：{bot_history[i]}" for i in range(len(bot_history))])
            input_text = context + "\n病人：" + user_history[-1] + "\n医生："
            '''
            for user_id in multi_user_list:
                self.list_memory[user_id] = ""

        else:
            self.list_memory = ""

    def get_response(self, query, user_id = None, img = None):  
        # query is the user's input
        # user_id is the user's id, used for multi-user chat history
        # return the bot's response
        print("generating response...")
        if self.multi_user_flag:
            self.list_memory[user_id] += "\n病人：" + query + "\n医生："
            print("history: ", self.list_memory[user_id])
            response, history = self.model.chat(self.tokenizer, query=self.list_memory[user_id], history=None, max_length=2048, num_beams=1, do_sample=True, top_p=self.top_p, temperature=self.temperature, logits_processor=None)
            # self.list_memory[user_id] = history
            # add the response to the chat history
            self.list_memory[user_id] += response
            return response
        else:
            self.list_memory += "\n病人：" + query + "\n医生："
            response, history = self.model.chat(self.tokenizer, query=self.list_memory, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=self.top_p, temperature=self.temperature, logits_processor=None)
            # self.list_memory = history
            # add the response to the chat history
            self.list_memory += response
            return response
        
    def release_chat_memory(self, user_id = None):
        if self.multi_user_flag:
            self.list_memory[user_id] = ""
        else:
            self.list_memory = ""


if __name__ == "__main__":

    device = "cpu"

    # tokenizer = AutoTokenizer.from_pretrained("./glm-4-9b-chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        # "THUDM/glm-4v-9b",
        "THUDM/glm-4-9b-chat",
        # cache_dir="D:\\LLM",
        # torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=True
    ).eval()
    # model.save_pretrained("glm-4v-9b-chat-int4")
    model.save_pretrained("glm-4-9b-chat-int4")
    # tokenizer.save_pretrained("glm-4-9b-chat-int4")