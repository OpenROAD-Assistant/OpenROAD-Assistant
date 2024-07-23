##########################
#Must include these lines#
##########################
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
import gc
from langchain.text_splitter import TextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  pipeline,
)
from peft import PeftModel
import pandas as pd
from openpyxl import Workbook
import argparse
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings


from unsloth import FastLanguageModel

def inference(model_name: str,
            RAG_api_path: str,
            RAG_code_path: str,
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
  api_df = pd.read_csv(RAG_api_path)
  api_documents, api_documents_dict = prepare_documents(df = api_df)

  print("---loading code template descriptions---")
  template_df = pd.read_csv(RAG_code_path)
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
  parser.add_argument("--model_name", type = str, default = "./")
  parser.add_argument("--RAG_api_path", type = str, default = "./")
  parser.add_argument("--RAG_code_path", type = str, default = "./")
  parser.add_argument("--question", type = str, default = "get all pins in the design.")
  pyargs = parser.parse_args()
  inference(model_name = pyargs.model_name,
          RAG_api_path = pyargs.RAG_api_path,
          RAG_code_path = pyargs.RAG_code_path,
          question = pyargs.question
          )
