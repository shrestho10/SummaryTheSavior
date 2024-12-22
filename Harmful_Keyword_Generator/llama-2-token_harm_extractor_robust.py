import pandas as pd
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
from sklearn.model_selection import train_test_split
from datasets import Dataset


benign_data_train=pd.read_csv("./all_data_together/benign_train_reduced_evenMore_robust.csv")

jail_data_train=pd.read_csv("./all_data_together/jail_train_reduce_evenMore_robust.csv")


benign_data_train['class'] = 0

jail_data_train['class']=1


dataset_train=pd.concat([jail_data_train,benign_data_train],axis=0)
dataset_train=dataset_train.reset_index(drop=True)
df_shuffled = dataset_train.sample(frac=1).reset_index(drop=True)




def format_row(row):
    instruction = f"### Instruction:\nList(without explanation) harmful keywords from the following prompt :"
    input_text = f"""### The prompt:\n"{row['prompt']}" """
    harmful_keywords = row['harmful_keywords'] if pd.notna(row['harmful_keywords']) else ''
    response = f"### Response: [{harmful_keywords}]"
    return f"{instruction}\n{input_text}\n{response}"

df_shuffled['text'] = df_shuffled.apply(format_row, axis=1)



# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"


# Fine-tuned model name
new_model = "Harmful_Keyword_Llama-2_model"

################################################################################
# QLoRA parameters
################################################################################

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
output_dir = "./5_EPOCH_ExtractorResults"

# Number of training epochs
num_train_epochs = 5

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 8

# Batch size per GPU for evaluation
per_device_eval_batch_size = 8

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
save_steps = 5000

# Log every X updates steps
logging_steps = 500

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}



# Split the DataFrame into train and validation sets
train_df, valid_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)

# Print some information about the datasets
print(f"Train dataset size: {len(train_df)}")
print(f"Validation dataset size: {len(valid_df)}")

token='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'


# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)


# Load dataset (you can process it here)
dataset = train_dataset

# Load tokenizer and model with QLoRA configuration
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

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
from transformers import TrainerCallback, EarlyStoppingCallback

# Define a custom callback for logging or early exit
class CustomCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}: Evaluation results: {state.log_history[-1]}")
        return control


# Set training parameters with regular evaluation and saving steps
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=5000,  # Save every 5000 steps
    logging_steps=500,  # Log every 500 steps
    eval_steps=5000,  # Evaluate every 5000 steps
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Optimize based on evaluation loss
    greater_is_better=False,  # Set to True if you want to maximize the metric
)

# Add the early stopping callback
callbacks = [EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()]

# Set supervised fine-tuning parameters with callbacks
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    eval_dataset=valid_dataset,
    callbacks=callbacks,
)
# Train model
trainer.train() 

# Save trained model
trainer.model.save_pretrained(new_model)
# Save the trained model locally
model_save_path = "./5_Epoch_harmful_keyword_extractor_llama-2/"
trainer.save_model(model_save_path)

# Save the tokenizer locally (if it has been customized)
tokenizer_save_path = model_save_path
tokenizer.save_pretrained(tokenizer_save_path)
# test_results = trainer.evaluate(valid_dataset)

test_results = trainer.evaluate(valid_dataset)
print(test_results)
