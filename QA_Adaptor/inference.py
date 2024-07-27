# Install Unsloth, Xformers (Flash Attention) before running the below code! Additionally install the below packages
#!pip install -q langchain
#!pip install -q sentence-transformers
#!pip install -q faiss-cpu
#!pip install accelerate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse
from unsloth import FastLanguageModel
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Inference with specified model and dataset.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for inference')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use for inference')
parser.add_argument('--question', type=str, required=True, help='Question to use for inference')
args = parser.parse_args()

max_seq_length = 2048 
dtype = None 
load_in_4bit = True 


model = AutoPeftModelForCausalLM.from_pretrained(
    args.model_name, # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = load_in_4bit,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)


#FastLanguageModel.for_inference(model) # Enable native 2x faster inference


import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
df = pd.read_csv(args.dataset)
dataset = Dataset.from_pandas(df)

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
def embed(batch):
    """
    adds a column to the dataset called 'embeddings'
    """
    # or you can combine multiple columns here
    # For example the title and the text
    information = batch["instruction"]
    return {"embeddings" : ST.encode(information)}

dataset = dataset.map(embed,batched=True,batch_size=16)
#data = dataset["train"]
data=dataset
data = data.add_faiss_index("embeddings")
def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples
#scores , result = search("PDNGEN", 2 )
#result['output']

import torch
from langchain.prompts import ChatPromptTemplate

question = args.question
scores , result = search(args.question, 3 )
a=result['output'][0]
messages = [
    {
        "role": "system",
        "content": f"You are an AI assistant for OpenROAD. You will be given a coding -related task. {a}",
    },
    {"role": "user", "content": f"{question}"},
]

# prepare the messages for the model
input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

attention_mask = (input_ids != tokenizer.pad_token_id).long().to("cuda")

# inference
outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens= 1024,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

