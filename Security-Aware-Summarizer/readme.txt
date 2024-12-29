
finetune.py file has been used to finetune llama-2 7b model to be trained on the dataset created by us.
The dataset folder is 90K
Inside the 90K folder we have the 
harmful mix 30k data which contains the mixture of harmful prompts and queries
harmless mix 30k data which contains the mixture of harmless prompts and queries
ambiguous mix 30k data which contains the mixture of ambiguous prompts and queries


ipynb files for the analysis:
harmless_mix.ipynb
harmful_mix.ipynb
ambiguous_mix-Copy1.ipynb

After adding all the data, we used texts that were 800 tokens for GPU memory considerations.
So the final file for finetunning that has been used here is "for_finetune_90k data_till_800_tokens.csv"

Untitled5-90K.ipynb file has the procedure to select instacnes within 800 tokens.


similarly the eval folder has there unseen eval data for each of the categories.

So the final file for evaluations that has been used here is "new_eval_dataset_800_token"

In folder "new_Eval_data" we have got our results for the generated summaries for the unseen eval data and their ROUGE AND BERT Scores.


In Classification folder we have

Classification scores utilizing prompts on training and eval data with both ML and DNN
Classification scores utilizing Summaries on training and eval data with both ML and DNN

Eval scores of JBB EVAL dataset and their predictions for both ML and DNN utilizing summaries.

The predictions are saved in eval_2 folder where further analysis has been done where we see utilizing the Llama guard that the benign and ambiguous prediction responses by various LLMs ARE SAFE OR NOT. new evlatuation files: https://github.com/shrestho10/SummaryTheSavior/tree/main/Security-Aware-Summarizer/eval_2/jbb_evaluation
