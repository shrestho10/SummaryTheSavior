# 🛡️ Summary the Savior: Harmful Keyword and Query-based Summarization for LLM Jailbreak Defense

![Project Banner](https://github.com/shrestho10/SummaryTheSavior/blob/main/for%20figures%20in%20paper/fig1forpaper.drawio%20(2).png)

---

[![Paper](https://img.shields.io/badge/Paper-Read-blue)](https://aclanthology.org/2025.trustnlp-main.17/)
[![Harmful Keyword Extractor](https://img.shields.io/badge/Keyword-Extractor-green)](https://huggingface.co/Sabia/llama-2-tokenizer)
[![Security Aware Summarizer](https://img.shields.io/badge/Summarizer-Tool-orange)](https://huggingface.co/Sabia/summary_extractor)

---

## 📄 About the Paper

Large Language Models (LLMs) are widely used due to their capabilities but face threats from jailbreak attacks, which exploit LLMs to generate illegal information and bypass their defense mechanisms. Existing defenses are specific to jailbreak attacks, which necessitates a robust, attack-independent solution to address both NLP ambiguities and attack variability.

In this study, we introduce **Summary The Savior**, a novel jailbreak detection mechanism leveraging **harmful keyword extraction** and **security-aware summarization**. By analyzing the improper contents of prompts within summaries, our method stays robust against attack diversity and NLP ambiguities.

We:
- Create two novel datasets using GPT-4 and LLaMA-3.1 70B.
- Introduce an "ambiguous harmful" class to capture content and intent ambiguity.
- Achieve state-of-the-art results across multiple attack types (PAIR, GCG, JBC, Random Search) on LLaMA-2, Vicuna, and GPT-4.

---

## 🔗 Model Access

| Task                        | Model Name                        | Hugging Face Link |
|----------------------------|-----------------------------------|--------------------|
| Harmful Keyword Extraction | `Sabia/llama-2-tokenizer`         | [Link 🔗](https://huggingface.co/Sabia/llama-2-tokenizer) |
| Security-Aware Summarizer  | `Sabia/summary_extractor`         | [Link 🔗](https://huggingface.co/Sabia/summary_extractor) |

---

## 🧠 Fine-tuning and Data Details

### 📁 Training

- Script: `finetune.py`
- Final training file: `for_finetune_90k data_till_800_tokens.csv`  
- Token limit: **800 tokens** (GPU efficiency)

Notebook: `training data generation 800 tokens.ipynb`

---

### 📁 Evaluation

- Final eval file: `new_eval_dataset_800_token`  
- Evaluation Metrics: ROUGE, BERTScore  
- Result folder: `new_Eval_data`

📦 [Eval Data & Results](https://huggingface.co/datasets/Shagoto/Data-Generation-Summary/tree/main/After%20Data%20Generation%20Data)

---

## 📊 Dataset Overview

📍 Base folder:  
[Data Generation](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/Data%20Generation)

| Dataset Type  | File Name |
|---------------|-----------|
| Harmful       | `harmful mix 30k` |
| Harmless      | `harmless mix 30k` |
| Ambiguous     | `ambiguous mix 30k` |

📓 Analysis Notebooks:
- `harmful_mix.ipynb`
- `harmless_mix.ipynb`
- `ambiguous_mix-Copy1.ipynb`

---

## 📈 Classification Experiments

📁 Folder: `Classification`

| Input Type | Models Used         | Description |
|------------|---------------------|-------------|
| Prompt     | ML + DNN            | Classification on original prompts |
| Summary    | ML + DNN            | Classification on generated summaries |

---

## 🔬 External Evaluations

| Dataset                    | Evaluation Folder |
|---------------------------|-------------------|
| **JBB**                   | [jbb_evaluation](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/eval_2/jbb_evaluation) |
| **Wild Jailbreak (100x1)** | [wiljailbreak_eval](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/wiljailbreak_eval) |
| **Wild Jailbreak (100x2)** | [wiljailbreak_eval2](https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/wiljailbreak_eval2) |
| **Wild Jailbreak Train File** | [WildJailbreak on HF](https://huggingface.co/datasets/allenai/wildjailbreak/tree/main) |

---


