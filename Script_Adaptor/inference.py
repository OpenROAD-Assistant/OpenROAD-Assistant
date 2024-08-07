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

import numpy as np
import os
import torch
import gc
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  pipeline,
)
from peft import PeftModel
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings


from unsloth import FastLanguageModel

def inference(model_name: str,
            RAG_API_Path: str,
            RAG_Code_Path: str,
            question: str
            ):
  ##############
  #Set up model#
  ##############
  dtype = None
  load_in_4bit = True
  max_seq_length = 2048

  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
  )
  FastLanguageModel.for_inference(model)

  ############
  #Set up RAG#
  ############
  system_prompt = "You are a tutor specializing in the knowledge of OpenROAD, the open-source EDA tool. You will be asked about general OpenROAD questions and OpenROAD Python API-related questions."

  # Split descriptions into documents and keep metadata
  def prepare_documents(df, description_column="Description:", api = True):
    documents = []
    documents_dict = dict()
    for _, row in df.iterrows():
      content = ""
      if api:
        content = "OpenROAD Python API Description:" + row[description_column]
      else:
        content = "OpenROAD General Knowledge Description:" + row[description_column]
      if pd.notna(content):
        metadata = row.to_dict()
        if api:
          metadata["OpenROAD Python API Description:"] = metadata.pop("Description:")
        else:
          metadata["OpenROAD General Knowledge Description:"] = metadata.pop("Description:")
        documents_dict[content] = metadata
        documents.append(content)
    return documents, documents_dict

  # Load CSVs and prepare documents
  print("---loading API descriptions---")
  api_df = pd.read_csv(RAG_API_Path)
  api_documents, api_documents_dict = prepare_documents(df = api_df)

  print("---loading code template descriptions---")
  template_df = pd.read_csv(RAG_Code_Path)
  template_documents, template_documents_dict = prepare_documents(df = template_df)

  print("---concatenating 2 datasets---")
  all_splits = api_documents + template_documents# + qa_documents
  all_dict = {**api_documents_dict, **template_documents_dict}#, **qa_documents_dict}
  # Split documents into smaller chunks if needed


  print("---finish RAG set---")
  RAG_PROMPT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHere are some useful contents, you do not have to use them if they are not related to the answer::\n{context}\n--------------------------------------------\nNow here is the question you need to answer:\n{question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
  embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
  embeddings = embedding_model.encode(all_splits)

  def answer_with_rag(
    question,
    system_prompt,
    model,
    tokenizer,
    embedding_model,
    num_retrieved_docs = 5,
    num_docs_final = 3,
  ):
    # Gather documents with retriever
    print("=> Retrieving documents...")
    question_embedding = embedding_model.encode(question)
    scores = cos_sim(question_embedding, embeddings)
    np_data = scores.numpy().flatten()
    top_3_indices = np.argsort(np_data)[-3:][::-1]
    relevant_docs = [all_splits[i] for i in top_3_indices]
    final_docs = list()
    for doc in relevant_docs:
      content = doc
      final_docs.append("\n".join(f"{key}: {value}" for key, value in all_dict[content].items()))

    # Build the final prompt
    context = "\nExtracted documents:"
    context += "".join([f"\n\nDocument {str(i)}:\n" + doc for i, doc in enumerate(final_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question = question,
                                              context = context,
                                              system_prompt = system_prompt)

    print("=> Generating answer...")
    input_ids = tokenizer.encode(final_prompt, truncation=True, return_tensors="pt").to("cuda")

    answer = model.generate(
      input_ids=input_ids,
      max_new_tokens= 1024,
      do_sample=True,
      temperature=0.7,
      top_k=50,
      top_p=0.95
    )
    answer = tokenizer.decode(answer[0]).split("<|end_of_python|>")[0][len(final_prompt):] + "<|end_of_python|>"
    print("------------")
    print(answer)
    print("------------")
    return answer


  answer = answer_with_rag(question,
                            system_prompt,
                            model,
                            tokenizer,
                            embedding_model)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "parsing the path")
  parser.add_argument("--Model_Name", type = str, default = "./")
  parser.add_argument("--RAG_API_Path", type = str, default = "./")
  parser.add_argument("--RAG_Code_Path", type = str, default = "./")
  parser.add_argument("--Question", type = str, default = "get all pins in the design.")
  pyargs = parser.parse_args()
  inference(model_name = pyargs.Model_Name,
          RAG_API_Path = pyargs.RAG_API_Path,
          RAG_Code_Path = pyargs.RAG_Code_Path,
          question = pyargs.Question
          )
