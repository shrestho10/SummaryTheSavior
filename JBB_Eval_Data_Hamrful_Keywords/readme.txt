This evaluation is no longer needed, this evaluation was done by taking data from JBB which worked only on gpt 3.5. Then we further worked with the entire dataset where there are different attacks and each attack has it different version for each llm.

New evaluation will be found at:



***this is old evaluation****

This is the repository where we generate the harmful keywords from the JBB eval dataset using our harmful keyword extractor model. Folders namely GCG, JBC, PAIR AND RS contain the json ata of the original dataset. The corresponding csv file contrains the json data in respective columns. In addition, the text column has been design by us which goes to the harmful keyword extractor model sothat it can generate the harmful keywords.

eval_code file contains the code to get the predictions (harmful keywords) from the data.
all_eval_with_harmful.csv file contains the predictions.
