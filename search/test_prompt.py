from openai import OpenAI
import os
import json

client = OpenAI(api_key="sk-xxx")
system_content = "You are an expert in the field of neural architecture search."
with open("search/search_prompt.json", "r") as f:
    search_prompt = json.loads(f.read())
user_input = search_prompt['vit4cls']['user_input']  # Note for this, load search_prompt as a json file

experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a channel list that can improve the model's performance on ImageNet-1k beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix = search_prompt['vit4cls']['suffix']


messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_input + suffix},
]
messages_history = []

res = client.chat.completions.create(model='gpt-4-1106-preview', messages=messages, temperature=0, n=1).choices[0].message  # request prompt and response
messages.append(res)
messages_history.append(messages)

# print(messages)
gpt_depth = json.loads(res.content.split('\n')[0])
print(gpt_depth)
gpt_heads = json.loads(res.content.split('\n')[1])
print(gpt_heads)
gpt_mlp_ratios = json.loads(res.content.split('\n')[2])
print(gpt_mlp_ratios)
