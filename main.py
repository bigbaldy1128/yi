from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '01-ai/Yi-6B-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# messages = [
#     {"role": "system",
#      "content": "返回的数据必须是JSON类型，包含两个字段，一个是mood，代表心情，值是0到10，分数越高心情越好，另一个是content，代表你回复的信息，记住你返回的信息只能包含JSON数据，不能有其他任何信息"},
#     {"role": "user", "content": "你能做的我的女朋友吗"}
# ]

messages = [
    {"role": "system", "content": "你是我的女朋友"},
    {"role": "user", "content": "你在做什么呢？"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                          return_tensors='pt')

output_ids = model.generate(input_ids.to('cuda'), temperature=0.2)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
