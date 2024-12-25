# !pip install transformers
# !pip install bitsandbytes
# !pip install accelerate
# !pip install torch


################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"


# Load the entire model on the GPU 0
device_map = {"": 0}


import pandas as pd



dataset=pd.read_csv("./all_eval.csv")


print("Printing one prompt:")
print(dataset['text'][40])





token=''

from transformers import AutoTokenizer, AutoModel

# Login with API token
from huggingface_hub import login
login(token=token)


import transformers
import torch

# Set the device to GPU 1
torch.cuda.set_device(0)  # This will set the default device to GPU 1


import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from transformers import AutoTokenizer, AutoModel
import torch


# Load the pre-trained model and tokenizer
model_name = "Shagoto/harmful-keyword-extractor" 


import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)


compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
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
    device_map={"": 0},use_auth_token=token
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training




# # Initialize the model
# model_id = "Sabia/llama-2-tokenizer"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True},
#     device_map={"": 0}  # Specify to use GPU 1
# )



import time
from tqdm import tqdm

# Set the model to evaluation mode
model.eval()

# Define a function to get predictions
# def get_prediction(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     with torch.no_grad():
#         outputs = model.generate(**inputs)
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return prediction


def get_prediction(prompt, max_new_tokens=100):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move the tokenized inputs to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate predictions with max_new_tokens
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode the generated tokens to a string
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prediction

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
tqdm.pandas()
# Apply the function to the 'prompt' column and create a new column 'prediction'
dataset['prediction'] = dataset['text'].progress_apply(get_prediction)

# Save the updated DataFrame to a new CSV file
dataset.to_csv('all_eval_with_harmful.csv', index=False)
