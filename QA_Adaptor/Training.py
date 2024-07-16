# Install Unsloth, Xformers (Flash Attention) before running the below code! 

from unsloth import FastLanguageModel
import torch
import argparse
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers.utils import logging

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

parser = argparse.ArgumentParser(description='Fine-tune a language model with specified datasets.')
parser.add_argument('--datasets', type=str, nargs=2, required=True, help='Paths of the two datasets to use for training')
args = parser.parse_args()


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

#@title Slim Orca data prep
from datasets import load_dataset
EOS_TOKEN = tokenizer.eos_token
dataset = load_dataset("Open-Orca/SlimOrca", split = "train")
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    mapper = {"system" : "SYSTEM:", "human" : "USER:", "gpt" : "ASSISTANT:"}
    end_mapper = {"system" : "\n\n", "human" : "\n", "gpt" : "</s>\n"}
    for convo in convos:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}{end_mapper[turn]}" for x in convo)+ EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
dataset = dataset.map(formatting_prompts_func, batched = True,)



trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title OpenROAD RAFT data prep
data = pd.read_json(args.datasets[0], lines=True)
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    mapper = {"system" : "SYSTEM:", "human" : "USER:", "gpt" : "ASSISTANT:"}
    end_mapper = {"system" : "\n\n", "human" : "\n", "gpt" : "</s>\n"}
    for convo in convos:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}{end_mapper[turn]}" for x in convo)+ EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
data = data.map(formatting_prompts_func, batched = True,)


logging.set_verbosity_info()

trainer2 = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = data,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer2.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

#@title OpenROAD data prep
data1 = pd.read_json(args.datasets[1], lines=True)
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    mapper = {"system" : "SYSTEM:", "human" : "USER:", "gpt" : "ASSISTANT:"}
    end_mapper = {"system" : "\n\n", "human" : "\n", "gpt" : "</s>\n"}
    for convo in convos:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}{end_mapper[turn]}" for x in convo)+ EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
data1 = data1.map(formatting_prompts_func, batched = True,)

logging.set_verbosity_info()

trainer3 = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = data1,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer3.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("OpenROAD-Assistant_QA_Model") # Local saving
tokenizer.save_pretrained("OpenROAD-Assistant_QA_Model")
# model.push_to_hub("your_name/OpenROAD-Assistant_QA_Model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/OpenROAD-Assistant_QA_Model", token = "...") # Online saving
