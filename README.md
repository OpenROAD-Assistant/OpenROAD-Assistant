# OpenROAD Assistant
This repository hosts the large-language models that serve as chatbots to OpenROAD. 
It includes the prompt-script and question-answer adpaters and the associated codes to train the models and perform inference. 



## Model Description

![Model Architecture](Images/Model_Architecture.png)

## Installation
### Requirements
- Python >= 3.8
- torch = 2.2.2+cu121
- xformers < 0.0.26
- CUDA = 7.5
- CUDA Toolkit = 12.1

### Run requirements.txt
```
pip install -r requirements.txt
```

## Data
1) For Q-A adaptor training, the dataset for instruction SFT can be found [here](https://huggingface.co/datasets/Open-Orca/SlimOrca)
2) For the  RAFT fine-tuning, the dummy datasets are uploaded in this repo and can be found [here](https://github.com/OpenROAD-Assistant/OpenROAD-Assistant/tree/main/Data)

*While fine-tuning the model in your system, kindly change the names of the datasets to make them consistent. For this study, we uploaded these datasets on HuggingFace which we recommend for ease of use but is not necessary.

## Running the Model
### Overview
While there are many ways to fine-tune LLMs including vLLMs, Unsloth, etc; this script fine-tunes Llama3 using the Unsloth library, optimized with techniques such as Low Rank Adaption (LoRA) and gradient checkpointing.

### Prerequisites
```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

### Configuration

While the default parameters used to fine-tune OpenROAD-Assistant are set in the Training script, you can also, according to your preference, configure the following parameters in the script:

max_seq_length: Maximum sequence length of the model.
- dtype: Data type for model parameters (auto-detected if set to None).
- load_in_4bit: Enables 4-bit quantization to reduce memory usage.
- Model-specific parameters like lora_alpha, lora_dropout and others as per your optimization needs.

Ensure that parameters like max_seq_length, dtype and load_in_4bit are consistent for both fine-tuning and inference

### Fine-Tuning
```
python Training.py
```
After running the script, the model will start training, and logs will be generated for each step. The output model's trained weights will be stored locally and on HuggingFace, thus, make sure you have a HF account or comment that part of the code


### Running Inference
Before running the script, ensure you adjust the following settings according to your environment:
- Model_name: Specify the pre-trained model.
- Batch size and other parameters in the embedding and retrieval functions for performance tuning.

```
python inference.py
```

OpenROAD-Assistant is available at the following link for direct inference:

[OpenROAD-Assistant](https://huggingface.co/Utsav2001/OR-QA-Adaptor)

Change the Model_name in the inference file to use the above model from HuggingFace


### Expected Output

A sample question- "What is Detailed Placement in OpenROAD?" has been asked in the script. The script processes a predefined query for which the output will be printed to the console.


