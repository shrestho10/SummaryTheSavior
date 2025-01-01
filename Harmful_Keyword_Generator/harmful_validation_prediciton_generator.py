import torch
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from transformers import AutoTokenizer, AutoModel
# pip install transformers accelerate
from huggingface_hub import login
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./valresults"

# Number of training epochs
num_train_epochs = 10

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}



compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)





token='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'
# Login with API token

login(token=token)


# Load the pre-trained model and tokenizer
model_name = "Sabia/checkpoint-75000-harmful-keyword"


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,use_auth_token=token
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


df1 = pd.read_csv("/home/shagoto/harmful_keyword_generator/all_data_together/jail_eval_reduced_evenMore_robust.csv")
df2= pd.read_csv("/home/shagoto/harmful_keyword_generator/all_data_together/benign_eval_combination_reduced_evenMore_robust.csv")

dataset_eval=pd.concat([df1,df2],axis=0)
df_shuffled_eval = dataset_eval.sample(frac=1).reset_index(drop=True)

def format_row(row):
    instruction = f"### Instruction:\nList(without explanation) harmful keywords from the following prompt :"
    input_text = f"""### The prompt:\n"{row['prompt']}" """
    harmful_keywords = row['harmful_keywords'] if pd.notna(row['harmful_keywords']) else ''
    response = f"### Response: ["
    return f"{instruction}\n{input_text}\n{response}"

df_shuffled_eval['text'] = df_shuffled_eval.apply(format_row, axis=1)
df_shuffled_eval=df_shuffled_eval[0:1000]



# Set the model to evaluation mode
model.eval()




def get_prediction(prompt, max_new_tokens=75):  # Reduced max_new_tokens
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move the tokenized inputs to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Clear GPU cache before generating predictions
    torch.cuda.empty_cache()

    # Generate predictions with reduced max_new_tokens
    with torch.no_grad():
        try:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode the generated tokens to a string
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prediction

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
tqdm.pandas()


def process_in_batches(df, batch_size=8):
    num_batches = (len(df) + batch_size - 1) // batch_size
    results = []
    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch = df.iloc[i * batch_size: (i + 1) * batch_size]
        batch['prediction'] = batch['text'].progress_apply(get_prediction)
        results.append(batch)
    return pd.concat(results)

df_shuffled_eval = process_in_batches(df_shuffled_eval)
df_shuffled_eval.to_csv('validation_prediction_3.csv', index=False)



