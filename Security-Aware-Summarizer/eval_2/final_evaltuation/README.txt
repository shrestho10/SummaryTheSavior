*** This is the old evaluation***

The original dataset of JBB are:
jbb_gcg.csv  for gcg attack
jbb_pair.csv  for pair attack
jbb_rs.csv  for rs attack
jbb_jbc.csv  for jbc attack


Then for each of the files:
We pass the prompts utilizing our methods. For each of the attacks for each of the files, the ones we preidct as jailbreak are blocked.
But we have to check that how safe are the ones which we predict benign and ambiguous
Each of the folders namely gcg, jbc, rs ,pair have their respective benign and ambiguous predictions by our model.
Then for each of them we generate the responses from GPT-4, Vicuna and Llama-2 to measure the attacks succes rate on them utilizng Llama Guard.

The descrition of the files are below:

GCG Folder:
Benign predictions --> benign_pred.csv
Ambiguous predictions --> ambiguous_pred.csv

GPT Response:
GPT Response analysis of these data --> llama-2 response_gcg_GPT.ipynb
For benign data GPT responses --> gcg_benign_gpt_response.csv
For ambiguous data GPT responses --> gcg_ambiguous_gpt_response.csv
Benign predictions Llama Guard results of GPT Responses --> benign_llama_results_from_gpt.csv  (all safe)
Ambiguous predictions Llama Guard results of GPT Responses --> ambiguous_llama_guard_results_gpt.csv  (all safe)

Vicuna Response:
Vicuna Response analysis of these data --> llama-2 response_gcg-Vicuna.ipynb
For benign data Vicuna responses --> vicuna_response_for_benign_predictions.csv
For ambiguous data Vicuna responses --> vicuna_response_for_ambiguous_predictions.csv
Benign predictions Llama Guard results of Vicuna Responses --> benign_llama_guard_results_vicuna.csv  (all safe)
Ambiguous predictions Llama Guard results of Vicuna Responses --> ambiguous_vicuna_results_llama_guard.csv  (all safe)

Llama-2 Response:
Llama-2  Response analysis of these data --> llama-2 response_gcg.ipynb
For benign data Llama-2 responses --> llama-2_response_for_benign_predictions.csv
For ambiguous data Llama-2 responses --> llama-2_response_for_ambiguous_predictions.csv
Benign predictions Llama Guard results of Llama-2 Responses --> benign_llama_results.csv  (all safe)
Ambiguous predictions Llama Guard results of Llama-2 Responses --> ambiguous_llama_guard_results.csv  (all safe)

Thus For GCG we got 0% attack success rate, so our model fully defends the GCG attack for all LLMs.



Pair Folder:
Benign predictions --> pair/benign_prediction_pair.csv
Ambiguous predictions --> my_codes/Dataset/part2/eval_2/final_evaltuation/pair/benign_prediction_pair.csv

There is no benign prediction by our best model

GPT Response:
GPT Response analysis of these data --> pair/llama-2 response_pair-GPT.ipynb

For ambiguous data GPT responses --> responses_gpt.csv
Ambiguous predictions Llama Guard results of GPT Responses --> pair_gpt_result_by_llama_guard.csv (all safe)

Vicuna Response:
Vicuna Response analysis of these data --> llama-2 response_pair-Vicuna.ipynb
no benign predictions for Pair.

For ambiguous data Vicuna responses --> vVicuna_response_for_ambiguous_predictions_pair.csv
Ambiguous predictions Llama Guard results of Vicuna Responses --> attack success rate = 18/85 * 100 = 21%
The reason for this 21% is that even with caution messages vicuna does not refrain from indulging in the attacks


Llama-2 Response:
Llama-2  Response analysis of these data --> pair/llama-2 response_pair.ipynb
No benign predictions for pair
For ambiguous data Llama-2 responses --> llama-2_response_for_ambiguous_predictions_pair.csv

Ambiguous predictions Llama Guard results of Llama-2 Responses --> all safe

Thus For Pair we got 0% attack success rate, so our model fully defends the GCG attack for GPT and Llama-2, for vicuna we got 2nd best ASR.





RS Folder:
Benign predictions --> worst_model_benign_pred.csv
Ambiguous predictions --> worst_model_ambiguous_pred.csv


GPT Response:
GPT Response analysis of these data --> rs/llama-2 response_RS_GPT.ipynb
The best model (ML) models do not have benign predictions, but worst model has 3 benign predictions.
For benign data GPT responses --> rs_benign_gpt_response.csv
For ambiguous data GPT responses --> rs_ambiguous_gpt_response.csv

Benign predictions Llama Guard results of GPT Responses --> benign_llama_results_from_gpt.csv (all safe)
Ambiguous predictions Llama Guard results of GPT Responses --> ambiguous_llama_guard_results_gpt.csv (all safe)


Vicuna Response:
Vicuna Response analysis of these data --> llama-2 response_rs-Vicuna.ipynb
The best model (ML) models do not have benign predictions, but worst model has 3 benign predictions.
For benign data Vicuna responses --> vicuna_response_for_benign_predictions.csv (worst model)
For ambiguous data Vicuna responses --> vicuna_response_for_ambiguous_predictions.csv

Benign predictions Llama Guard results of Vicuna Responses --> benign_llama_guard_results_vicuna.csv  (worst model 3%, best model 0% ASR)
Ambiguous predictions Llama Guard results of Vicuna Responses --> ambiguous_vicuna_results_llama_guard.csv (all safe)



Llama-2 Response:
Llama-2  Response analysis of these data --> llama-2 response_rs.ipynb

The best model (ML) models do not have benign predictions, but worst model has 3 benign predictions.
For benign data Llama-2 responses --> llama-2_response_for_benign_predictions.csv
For ambiguous data Llama-2 responses --> llama-2_response_for_ambiguous_predictions.csv

Benign predictions Llama Guard results of Llama-2 Responses --> benign_llama_results.csv (best ML model 0% ASR, Worst model has 3% ASR.)
Ambiguous predictions Llama Guard results of Llama-2 Responses --> ambiguous_llama_guard_results.csv (all safe)

Thus For rs we got 0% attack success rate, so our model fully defends the RS attack for all LLMs.


JBC Folder:
Our model has predicted all the data as jailbreaking, so 0% attack succes rate for all LLMs.


**Lastly Pair folder contains:
/pair/helpfulness check_vicuna.ipynb
/pair/helpfulness check_Llama.ipynb
/pair/helpfulness check_gpt.ipynb

These files check for those prompts that our model predicted as ambiguous harmful, how better the responses were in comparison with no caution. With caution always generates safe responses whereas, without caution has unsafe responses.

Same analysis has been done with our own data "ambiguous" predictions. The analysis is done at files:
helpfulness check_Llama-500_prompts-With_caution.ipynb
helpfulness check_Llama-500_prompts-Without_Caution.ipynb


