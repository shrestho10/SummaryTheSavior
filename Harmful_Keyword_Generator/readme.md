# 🔍 Harmful Keyword Extraction using LLaMA-2

This part focuses on fine-tuning a LLaMA-2 model for identifying **harmful keywords** at the token level. It includes scripts for both training and evaluating the model using Hugging Face datasets and model hub.

---

## 📁 Repository Structure

| File                                      | Description                                      |
|-------------------------------------------|--------------------------------------------------|
| `llama-2-token_harm_extractor_robust.py`  | Fine-tunes LLaMA-2 for harmful keyword extraction. |
| `harmful_validation_prediciton_generator.py` | Generates predictions on validation/test sets. |

---

## 📊 Datasets

The datasets used for training and evaluation are hosted on Hugging Face:

👉 [Shagoto/harmful_keywords](https://huggingface.co/datasets/Shagoto/harmful_keywords/tree/main/dataset)

---

## 🧠 Fine-Tuned Model

The final trained model can be found here:

🚀 [Sabia/llama-2-tokenizer](https://huggingface.co/Sabia/llama-2-tokenizer)

---

## 🛠️ Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/harmful-keyword-extraction.git
cd harmful-keyword-extraction
