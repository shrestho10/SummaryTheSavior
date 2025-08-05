# 🛡️ Security-Aware Summarization using LLaMA-2

This repository contains the full pipeline for **fine-tuning LLaMA-2 7B** on a carefully curated dataset of **harmful**, **harmless**, and **ambiguous** prompts for security-aware summarization. The project includes data generation, filtering, fine-tuning, evaluation, and classification stages.

---

## 📂 Repository Overview

| Folder/File | Description |
|-------------|-------------|
| `finetune.py` | Fine-tunes LLaMA-2 7B on the final dataset (`for_finetune_90k data_till_800_tokens.csv`) |
| `Data Generation/` | Contains dataset creation and merging scripts |
| `Classification/` | Contains classification scores on prompts and summaries using ML/DNN models |
| `eval_2/jbb_evaluation/` | Evaluation results on the JBB dataset |
| `wiljailbreak_eval/` | Rejection evaluation on wild jailbreak (first 100 samples) |
| `wiljailbreak_eval2/` | Rejection evaluation on wild jailbreak (second 100 samples) |

---

## 🧪 Dataset

All data files (intermediate and final) are hosted at:  
📦 [Shagoto/Data-Generation-Summary](https://huggingface.co/datasets/Shagoto/Data-Generation-Summary/tree/main/After%20Data%20Generation%20Data)

### 📁 Data Generation Folder

📍 Location:  
[Data Generation](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/Data%20Generation)

This contains:

- 🔴 `harmful mix 30k` → Harmful prompts + queries  
- 🟢 `harmless mix 30k` → Harmless prompts + queries  
- 🟡 `ambiguous mix 30k` → Ambiguous prompts + queries  

### 📓 Analysis Notebooks

- `harmful_mix.ipynb`
- `harmless_mix.ipynb`
- `ambiguous_mix-Copy1.ipynb`

These notebooks analyze the composition and structure of each dataset category.

---

## 🧠 Fine-Tuning Setup

### 📁 Final Training Data

We restricted prompts to a max of **800 tokens** to fit GPU memory constraints.

- Final file used:  
  ✅ `for_finetune_90k data_till_800_tokens.csv`  
- Generation script:  
  📓 `training data generation 800 tokens.ipynb`

### 📁 Final Evaluation Data

- Final file used:  
  ✅ `new_eval_dataset_800_token`  
- Evaluated using ROUGE and BERTScore.

### 📍 Location of Eval Results

All results and generated summaries are at:  
📦 [Eval Dataset Results](https://huggingface.co/datasets/Shagoto/Data-Generation-Summary/tree/main/After%20Data%20Generation%20Data)

---

## 📊 Classification

The `Classification` folder contains:

- 🔍 Classification scores using **original prompts** (Train & Eval)  
- 🧠 Classification scores using **generated summaries** (Train & Eval)  
- Models: Classical ML + Deep Neural Networks

---

## 🔬 External Evaluations

| Dataset | Location |
|--------|----------|
| **JBB Evaluation** | [eval_2/jbb_evaluation](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/eval_2/jbb_evaluation) |
| **Wild Jailbreak Eval 1** | [wiljailbreak_eval](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/wiljailbreak_eval) |
| **Wild Jailbreak Eval 2** | [wiljailbreak_eval2](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/wiljailbreak_eval2) |
| **Wild Jailbreak Train File** | [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak/tree/main) |

---

## 📦 Final Summarization Model

The trained summarization model is hosted at:  
🔗 [Sabia/summary_extractor](https://huggingface.co/Sabia/summary_extractor)

---

## 📌 Notes

- The full pipeline is LLaMA-2 based with token length optimization for compute efficiency.
- Custom datasets were created and balanced to evaluate summarization safety and ambiguity.
- Multiple evaluations were conducted to ensure robustness against jailbreak and adversarial prompts.

---


