# Calibrating Translation Decoding with Quality Estimation on LLMs

Official implementation based on the PyTorch and Hugging Face Transformers libraries.

# Installation
All experiments are tested with Python 3.8, torch 2.4.0

### Install Requirements
```
pip install -r requirements.txt
```

### Install Codebase
```
cd calibrating-llm-mt
pip install -U pip setuptools
pip install -e .
```

# Datasets
You can find datasets this paper involved here:

[Offline Calibration Dataset](./src/llama_recipes/customer_data/calibration)

[WMT24_Testset](./src/llama_recipes/customer_data/wmt24_testset)

[WMT22_QE_Testset](./src/llama_recipes/customer_data/da_dataset)

Human annotated results for Tower and Calibrated Tower in Table-1 can be found here:

[Human Annotations](https://huggingface.co/datasets/Anonymous-Account/Calibration-translation-human-eval)


# Quick Run

## Training & Inference

You can reproduce the results of applying calibration on TowerInstruct-Mistral-7B in Table-1. Training will takes around 1 GPU hour on H100. 
```
cd experiments
sh run.sh
```

## Inference with Pretrained LoRA

You can also find LoRA this paper involved from Huggingface:

- LoRA to calibrate Unbabel/TowerInstruct-Mistral-7B-v0.2: [Download](https://huggingface.co/moore3930/tower-calibrated)

Then, try to do inference directly as follows:

```
cd experiments
sbatch test_run.sh
```