import pandas as pd
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from transformers import  set_seed

set_seed(42)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


token=''
from transformers import AutoTokenizer, AutoModel

# Login with API token
from huggingface_hub import login
login(token=token)

filtered_df=pd.read_csv(".for_finetune_90k data_till_800_tokens.csv")
# Load datasets
dataset = filtered_df

# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# Fine-tuned model name
new_model = "Security-Aware-Summarizer"

################################################################################
# QLoRA parameters
################################################################################

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

output_dir = "./ExtractorResults"
num_train_epochs = 5
fp16 = False
bf16 = False
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 1000
logging_steps = 500

################################################################################
# SFT parameters
################################################################################

max_seq_length = 800
packing = False
device_map = {"": 0}
# Split the DataFrame into train and validation sets
train_df, valid_df = train_test_split(dataset, test_size=0.1, random_state=42)

# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)



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
        
        


# Load base model with device_map set to auto for multi-GPU usage
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map, 
    use_auth_token=''
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token='')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

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
    save_steps=1000,  
    logging_steps=500,  
    eval_steps=1000,  
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
    resume_from_checkpoint=True,  # Automatically resume from the last checkpoint

)

# Add the early stopping callback
callbacks = [EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()]

# Set supervised fine-tuning parameters with callbacks
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    eval_dataset=valid_dataset,
    callbacks=callbacks,
)

# Load from a specific checkpoint if available
checkpoint_dir = "./ExtractorResults/10000"
if os.path.exists(checkpoint_dir):
    print("Checkpoint Found!!!!!!!!!!!!!!!!")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# Save the trained model locally
model_save_path = "./security_aware_summarizer_llama-2-7B/"
trainer.save_model(model_save_path)

# Save the tokenizer locally (if it has been customized)
tokenizer.save_pretrained(model_save_path)

# Evaluate the model on the validation set
test_results = trainer.evaluate(valid_dataset)
print(test_results)
