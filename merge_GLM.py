import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForSequenceClassification,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="chatglm", type=str, required=False)
    # parser.add_argument('--tokenizer_path', default="D:\\LLM\\LLaMA-Factory\\saves\\GLM-4-9B-Chat\lora\\train_2024-07-20-11-23-12\\tokenizer.model", type=str,
    #                     help="Please specify tokenization path.")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Please specify tokenization path.")
    parser.add_argument('--output_dir', default='D:\\LLM', type=str)
    parser.add_argument('--base_model_path', type=str, default="D:\\LLM\\glm-4-9b-chat")
    parser.add_argument('--lora_model_path', type=str, default="D:\\LLM\\LLaMA-Factory\\saves\\GLM-4-9B-Chat\\lora\\train_2024-07-20-11-23-12\\checkpoint-2800")
    args = parser.parse_args()

    base_model_path = args.base_model_path
    lora_model_path = args.lora_model_path
    output_dir = args.output_dir
    peft_config = PeftConfig.from_pretrained(lora_model_path)
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # 模型加载
    if peft_config.task_type == "SEQ_CLS":
        if args.model_type == "chatglm":
            raise ValueError("chatglm does not support sequence classification")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True
        )
    else:
        base_model = model_class.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True
        )

    # 分词器加载
    if args.tokenizer_path:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = tokenizer_class.from_pretrained(base_model_path, trust_remote_code=True)

    # 修改词表大小
    # if args.resize_emb:
    #     base_model_token_size = base_model.get_input_embeddings().weight.size(0)
    #     if base_model_token_size != len(tokenizer):
    #         base_model.resize_token_embeddings(len(tokenizer))

    # 初始化Peft新模型
    new_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_dir=output_dir,
        load_in_4bit=True
    )
    new_model.eval()
    new_base_model = new_model.merge_and_unload()

    tokenizer.save_pretrained(output_dir)
    new_base_model.save_pretrained(output_dir, safe_serialization=False, max_shard_size='2GB')


if __name__ == '__main__':
    main()