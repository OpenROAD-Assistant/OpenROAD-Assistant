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

## Running the Model
### Overview
While there are many ways to fine-tune LLMs including vLLMs, Unsloth, etc; this script fine-tunes Llama3 using the Unsloth library, optimized with techniques such as Low Rank Adaption (LoRA) and gradient checkpointing.

### Prerequisites
```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

### Fine-Tuning
```
python Training.py
```
After running the script, the model will start training, and logs will be generated for each step. Outputs are typically saved in the specified output directory.


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


