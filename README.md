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


## Running the Model
### Overview
While there are many ways to fine-tune LLMs including vLLMs, Unsloth, etc; this script fine-tunes Llama3 using the Unsloth library, optimized with techniques such as Low Rank Adaption (LoRA) and gradient checkpointing.

### Data
1) For Q-A adaptor training, the dataset for instruction SFT can be found [here](https://huggingface.co/datasets/Open-Orca/SlimOrca)
2) For the  RAFT fine-tuning, the dummy datasets are uploaded in this repo and can be found [here](https://github.com/OpenROAD-Assistant/OpenROAD-Assistant/tree/main/Data)

*While fine-tuning the model in your system, kindly change the names of the datasets to make them consistent. For this study, we uploaded these datasets on HuggingFace which we recommend for ease of use but is not necessary.


### Configuration

While the default parameters used to fine-tune OpenROAD-Assistant are set in the Training script, you can also, according to your preference, configure the following parameters in the script:

max_seq_length: Maximum sequence length of the model.
- dtype: Data type for model parameters (auto-detected if set to None).
- load_in_4bit: Enables 4-bit quantization to reduce memory usage.
- Model-specific parameters like lora_alpha, lora_dropout and others as per your optimization needs.

Ensure that parameters like max_seq_length, dtype and load_in_4bit are consistent for both fine-tuning and inference

### Fine-Tuning

#### 1. Clone this repo
```
git clone "https://github.com/OpenROAD-Assistant/OpenROAD-Assistant.git"
```
#### 2. If you are on High Performance Clusters (HPC), start an interactive instance to gain compute resources (Optional)
```
srun --cpus-per-task=2 --mem=16GB --gres=gpu:v100:1 --time=4:00:00 --pty /bin/bash -l
```
#### 3. Create and activate a virtual environment (Recommended)
```
python -m venv ORA
source ORA/bin/activate
```
#### 4. Download all requirements 
```
cd OpenROAD-Assistant/QA_Adaptor/ or cd OpenROAD-Assistant/Script_Adaptor/
pip install -r requirements.txt
```
#### 5. Run the training script
```
python training.py --datasets ../Data/Dummy_Raft-FT-1.jsonl ../Data/Dummy_Raft-FT-2.jsonl
```
After running the script, the model will start training, and logs will be generated for each step. The output model's trained weights will be stored locally and on HuggingFace, thus, make sure you have an HF account or comment that part of the code

##### Note:
 It takes 2 hours and 37 minutes to train the OpenROAD-Assistant’s script adapter using four NVIDIA RTX A5500 GPUs, and it takes 50 minutes to train the OpenROAD-Assistant’s Question-answer adapter using one NVIDIA V100 GPU. OpenROAD-Assistant is trained for 30 epochs until the training loss converges


### Running Inference
Before running the script, ensure you adjust the following settings according to your environment:
- Model_name: Specify the pre-trained model.
- Batch size and other parameters in the embedding and retrieval functions for performance tuning.

```
python inference.py --model_name "OpenROAD-Assistant" --dataset ../Data/RAG_Database.csv --question "What is Detailed Placement in OpenROAD?"
```

OpenROAD-Assistant is available at the following link for direct inference:

[OpenROAD-Assistant](https://huggingface.co/Utsav2001/OR-QA-Adaptor)

Change the Model_name in the inference file to use the above model from HuggingFace


### Expected Output

A sample question- "What is Detailed Placement in OpenROAD?" has been asked in the script. The script processes a predefined query for which the output will be printed to the console.


