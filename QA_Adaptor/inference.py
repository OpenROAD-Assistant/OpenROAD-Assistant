#BSD 3-Clause License
#
#Copyright (c) 2024, OpenROAD-Assistant
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
parser.add_argument('--Model_Name', type=str, required=True, help='Name of the model to use for inference')
parser.add_argument('--Dataset', type=str, required=True, help='Name of the dataset to use for inference')
parser.add_argument('--Question', type=str, required=True, help='Question to use for inference')
args = parser.parse_args()

max_seq_length = 2048
dtype = None
load_in_4bit = True


model = AutoPeftModelForCausalLM.from_pretrained(
    args.Model_Name, # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = load_in_4bit,
)
tokenizer = AutoTokenizer.from_pretrained(args.Model_Name)


#FastLanguageModel.for_inference(model) # Enable native 2x faster inference


import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
df = pd.read_csv(args.Dataset)
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

question = args.Question
scores , result = search(args.Question, 3 )
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

