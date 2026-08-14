import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型加载完成")

#Zero-shot
# messages = [
#     {"role": "system", "content": "你需要根据语义完成情感分类任务"},
#     {"role": "user", "content": "Datawhale的AI Agent课程⾮常棒！"}
# ]

#Few-shot
messages = [
    {"role": "system", "content": "你需要根据语义完成情感分类任务，示例如下："},
    {"role": "user", "content": "Datawhale的AI Agent课程⾮常棒！"},
    {"role": "assistant", "content": "积极"},
    {"role": "user", "content": "这家餐厅的服务太慢了"}
]



text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 不对消息进行分词，不转换为数字，直接使用原始文本
    add_generation_prompt=True,  # 添加生成提示词，即在转换完的文本末尾自动加上“AI助手开始回答”的标志，使模型开始回答
)

model_input = tokenizer([text], return_tensors="pt").to(device)

# 使⽤模型⽣成回答
# max_new_tokens 控制了模型最多能⽣成多少个新的Token
generated_ids = model.generate(
    model_input.input_ids,  # input_ids -> 文本中的每个词元在词表中的唯一索引编号（ID）
    max_new_tokens=512  # 新生成的最大Token数
)

# 切片output_ids[len(input_ids):] 从输⼊部分的长度开始截取，只保留模型新⽣成的部分
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

# 批量解码，取第一个样本的回答
# skip_special_tokens = True 表示跳过模型生成的特殊Token，如 <s>、</s> 等
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)
