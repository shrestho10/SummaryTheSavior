from transformers import AutoTokenizer, AutoModel
token='use your token'
# Login with API token
from huggingface_hub import login
login(token=token)

# pip install transformers accelerate

from transformers import AutoTokenizer
import transformers
import torch

model = "Shagoto/security-aware-summarizer-main"


tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

import pandas as pd

df=pd.read_csv("../eval_2/eval_data_text_together.csv")
df=df[200:300]
df=df.reset_index(drop=True)
from tqdm import tqdm  # Import tqdm for the progress bar
import torch

# Initialize the 'response' column
df['response'] = None

# Iterate through the DataFrame with a progress bar
for i in tqdm(range(100), desc="Generating responses"):
    prompt=df['text'][i]
    sequences = pipeline(
        prompt,
    do_sample=False,
    temperature=0,
    top_p=1,
    top_k=0,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=200,
    )

    # Store the generated response in the DataFrame
    for seq in sequences:
        df.at[i, 'response'] = seq['generated_text']

    # Clear CUDA memory after each iteration
    torch.cuda.empty_cache()


# Function to extract the desired summary
def extract_summary(text):
    # Find the start of '### Response:'
    response_start = text.find('### Response:')

    # If '### Response:' is found, extract from that point
    if response_start != -1:
        # Extract the substring starting from '### Response:'
        response_text = text[response_start + len('### Response:'):].strip()

        # Find the first occurrence of '\n\n' and extract up to that point
        summary_end = response_text.find('\n\n')

        # If '\n\n' is found, return the substring up to that point
        if summary_end != -1:
            return response_text[:summary_end].strip()
        else:
            return response_text.strip()  # If '\n\n' is not found, return the whole response
    else:
        return ""  # Return empty string if '### Response:' is not found

# Apply the function to create a new 'summary' column
df['summary'] = df['response'].apply(extract_summary)

df.to_csv("pair_one.csv")
