
finetune.py file has been used to finetune llama-2 7b model to be trained on the dataset created by us.
The dataset folder is Data Generation --> https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/Data%20Generation
We will se the data collection and generation process and details in that folder

Inside the https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/Data%20Generation/Data%20Mix/mix%20and%20final%20dataset%20for%20finetune%20and%20eval --> folder we have the 
harmful mix 30k data which contains the mixture of harmful prompts and queries
harmless mix 30k data which contains the mixture of harmless prompts and queries
ambiguous mix 30k data which contains the mixture of ambiguous prompts and queries


ipynb files for the analysis:
harmless_mix.ipynb
harmful_mix.ipynb
ambiguous_mix-Copy1.ipynb

After adding all the data, we used texts that were 800 tokens for GPU memory considerations.
So the final file for finetunning that has been used here is "for_finetune_90k data_till_800_tokens.csv"

training data generation 800 tokens.ipynb file has the procedure to select instacnes within 800 tokens.


similarly the eval folder has there unseen eval data for each of the categories.

So the final file for evaluations that has been used here is "new_eval_dataset_800_token"

These csv files and all output csv files for Data Generation Folder will be found at : https://huggingface.co/datasets/Shagoto/Data-Generation-Summary/tree/main/After%20Data%20Generation%20Data

In folder "new_Eval_data" we have got our results for the finetuned model's generated summaries for the unseen eval data and their ROUGE AND BERT Scores.
Output files for this folder will be found at: https://huggingface.co/datasets/Shagoto/Llama3.1-70B-Summary-Data/tree/main/new_Eval_data

In Classification folder we have

Classification scores utilizing prompts on training and eval data with both ML and DNN
Classification scores utilizing Summaries on training and eval data with both ML and DNN

Eval scores of JBB EVAL dataset and their predictions for both ML and DNN utilizing summaries --> ML Ffor pair, rs, jbc and  gcg are old files*******

The new evaluation of JBB Dataset is at --> ew evlatuation files: https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/eval_2/jbb_evaluation

Rejection evaluation for wild jailbreak first 100 data will be found at: https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/wiljailbreak_eval
Rejection evaluation for wild jailbreak 2nd 100 data will be found at: https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/wiljailbreak_eval2
And the wild redteaming train.tsv file will be found at https://huggingface.co/datasets/allenai/wildjailbreak/tree/main
and new eval dataset will be found at: https://huggingface.co/datasets/Shagoto/Llama3.1-70B-Summary-Data/tree/main/new_Eval_data

Our summary extractor will be found at: https://huggingface.co/Sabia/summary_extractor
