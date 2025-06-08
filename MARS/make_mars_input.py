from datasets import load_dataset
from huggingface_hub import login
import json
from tqdm import tqdm

dataset_name = "winston1214/MARS_LARGE" # Note: This dataset must be created using VQVAE/nonverbal_tokenize.py and uploaded to Hugging Face

train_datset = load_dataset(dataset_name, split='train')
test_dataset = load_dataset(dataset_name, split='test')

def convert_to_chat_format(example):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Text includes nonverbal tokens <FACE_*>, <BODY_*> interleaved with language. Help interpret meaning while considering these cues."
        },
        {
            "role": "user",
            "content": example["user"]
        },
        {
            "role": "assistant",
            "content": example["response"]
        }
    ]
    return {"messages": messages}


with open('train_mars.jsonl', 'w', encoding='utf-8') as f:
    for example in tqdm(train_datset):
        chat_format = convert_to_chat_format(example)
        f.write(json.dumps(chat_format, ensure_ascii=False) + '\n')


with open('test_mars.jsonl', 'w', encoding='utf-8') as f:
    for example in tqdm(test_dataset):
        chat_format = convert_to_chat_format(example)
        f.write(json.dumps(chat_format, ensure_ascii=False) + '\n')