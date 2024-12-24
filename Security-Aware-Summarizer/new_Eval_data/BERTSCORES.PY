import pandas as pd
from bert_score import score
from tqdm import tqdm
import torch

# Sample DataFrame with columns 'reference' and 'hypothesis'
eval_dataset = pd.read_csv("new_eval_dataset.csv")
df = eval_dataset.fillna('')
# Initialize tqdm for progress bar
tqdm.pandas(desc="Calculating BERTScore")

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Compute BERTScore for each row
def calculate_bertscore(reference, hypothesis):
    # Compute BERTScore with CUDA if available
    P, R, F1 = score([hypothesis], [reference], lang='en', verbose=False, device=device)

    # Return the average scores
    return {'precision': P.mean().item(), 'recall': R.mean().item(), 'f1': F1.mean().item()}

# Apply BERTScore calculation to each row with progress bar
df['bertscore'] = df.progress_apply(lambda row: calculate_bertscore(row['summary'], row['predicted_summary']), axis=1)

bertscore_df = pd.json_normalize(df['bertscore'])

# Join scores back to original DataFrame
df = pd.concat([df, bertscore_df], axis=1)

# Calculate average BERTScore across all rows
average_bertscore = df[['precision', 'recall', 'f1']].mean()

print("Average BERTScore:")
print(average_bertscore)
