# OpenROAD Assistant
This repository hosts the scripts to use the large-language models that serve as chatbots to OpenROAD. The large language models are hosted on HuggingFace. The repository scripts to run training and inference for both the prompt-script (PS or script) and question-answer (QA) adpaters.


## Model description and data description

![Model Architecture](Images/Model_Architecture.png)



### Overview
While there are many ways to fine-tune LLMs, including vLLMs, Unsloth, etc., this script fine-tunes Llama3 using the Unsloth library, optimized with techniques such as Low-Rank Adaption (LoRA) and gradient checkpointing.

### Data
1) For QA adaptor training, the dataset for instruction SFT can be found [here](https://huggingface.co/datasets/Open-Orca/SlimOrca)
2) For the  RAFT fine-tuning, the dummy datasets are uploaded in this repo and can be found [here](https://github.com/OpenROAD-Assistant/OpenROAD-Assistant/tree/main/Data)
3) For the QA adaptor the RAG database is uploaded to this repo and can be found [here](https://github.com/OpenROAD-Assistant/OpenROAD-Assistant/tree/main/Data)
4) For script adaptor the RAG databased is also uploaded to this repo and can be found [here](https://github.com/OpenROAD-Assistant/OpenROAD-Assistant/tree/main/Data)
5) In addition, to the above data, we also use EDA-Corpus for training which can be found [here](https://github.com/OpenROAD-Assistant/EDA-Corpus). 



## Using OpenROAD-Assistant

### Requirements

- Python3.9.16
- Pip24.0

### Steps to run OpenROAD-Assistant


#### 1. Clone this repo
```
git clone "https://github.com/OpenROAD-Assistant/OpenROAD-Assistant.git"
```

#### 2. Create and activate a virtual environment (Recommended)
```
python3 -m venv ORA
source ORA/bin/activate
```
#### 3. Download all requirements 
```
pip3 install -r requirements.txt
```
#### 4. Run the training script

The following command will perform training for the QA adaptor

```
cd QA_Adaptor
python3 training.py --datasets ../Data/Example_Raft_FT_1.jsonl ../Data/Example_Raft_FT_2.jsonl
```
After running the script, the model will start training, and logs will be generated for each step. The output model's trained weights will be stored locally and on HuggingFace, thus, make sure you have an HF account or comment that part of the code

The following command will perform training for the script adaptor

``` 
TBD
```

While the default parameters used to fine-tune OpenROAD-Assistant are set in the training script, you can also, according to your preference, configure the following parameters in the script:

max_seq_length: Maximum sequence length of the model.
- dtype: Data type for model parameters (auto-detected if set to None).
- load_in_4bit: Enables 4-bit quantization to reduce memory usage.
- Model-specific parameters like lora_alpha, lora_dropout and others as per your optimization needs.

Ensure that parameters like max_seq_length, dtype and load_in_4bit are consistent for both fine-tuning and inference


##### Note:
It takes about 2.5 hours to train the OpenROAD-Assistant’s script adapter using four NVIDIA RTX A5500 GPUs, and it takes about 1 hour to train the OpenROAD-Assistant’s Question-Answer adapter using one NVIDIA V100 GPU. OpenROAD-Assistant is trained for 30 epochs until the training loss converges. If you do not want to spend resources training, please use are pretrained models from HuggingFace and directly run inference as shown below. 


#### 5. Run the inference script
When you run inference, the code fetches the model directly from HuggingFace. If you have run your own training and a different model, please point to the name of the model below with the correct path as an argument. If not, the following script will fetch a pretrained model. 

For QA adaptor
```
cd QA_Adaptor
python3 inference.py --Model_Name "OpenROAD-Assistant/QA-Adaptor" --Dataset ../Data/RAG_QA_Database.csv --Question "What is Detailed Placement in OpenROAD?"
```
For Script adaptor
```
cd Script_Adaptor
python3 inference.py --Model_Name "OpenROAD-Assistant/Script_Adaptor" --RAG_API_Path ../Data/RAG_APIs.csv --RAG_Code_Path ../Data/RAG_Code_Piece.csv --Question "How can I get every pin in the design in a list?"
```
##### Note:
OpenROAD-Assistant-QA-Adaptor is available at the following link for direct inference:

[OpenROAD-Assistant/QA-Adaptor](https://huggingface.co/OpenROAD-Assistant/QA-Adaptor)

OpenROAD-Assistant-Script-Adaptor is available at the following link for direct inference:

[OpenROAD-Assistant/Script_Adaptor](https://huggingface.co/OpenROAD-Assistant/Script_Adaptor)

##### Note:
If you run training with a different batch size and model name, before running the script, ensure you adjust the following settings according to your environment:
- Model_name: Specify the pre-trained model.
- Batch size and other parameters in the embedding and retrieval functions for performance tuning.

## Cite this work:
```
U. Sharma, B.-Y. Wu, S. R. D. Kankipati, V. A. Chhabria, and A. Rovinski, “OpenROAD-Assistant: An Open-Source Large Language Model for Physical Design Tasks,” ACM/IEEE International Symposium on Machine Learning for CAD (MLCAD), 2024.
```
