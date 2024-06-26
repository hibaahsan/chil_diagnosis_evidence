**Data**


The MIMIC-III evaluation dataset is under review on Physionet. We will update the link once approved!

**Evidence Generation**

1. To generate evidence using Flan-T5 or Mistral-Instruct

		cd scripts
		sh zero_shot_pipeline.sh [FILE_NAME] [FOLDER] [MODEL_NAME] [SUFFIX]
	
	`FILE_NAME`: input .csv file with columns: `ROW_ID` (Unique instance identifier (string)), `DIAGNOSIS` (suspected diagnosis for the instance), `NOTE_ID` (EHR note identifier), `TEXT` (EHR note), `NOTE_TYPE` (For example, radiology, discharge summary)
	`FOLDER`: Folder that contains the above input file. This is also the output folder <br>
	`MODEL_NAME`: Huggingface model name <br>
	`SUFFIX`: This is a string that is appended at the end of every output file <br>
	
	The output file [FOLDER]/two_step_output_[FILE_NAME][SUFFIX].csv contains evidence.

3. To generate evidence using Clinical-BERT

	2.1 Generate CBERT embeddings for risk factors from GPT 3.5. We have proivided an example file in the data folder. Run the below sbatch after updating the output and cache paths.

   		cd scripts
		sbatch run_embed_risk_factors.sbatch
		sh retrieval_pipeline.sh [FILE_NAME] [FOLDER] [CACHE_PATH] [EMBEDDINGS_PATH]

	2.2 Run the pipeline
   
   		sh retrieval_pipeline.sh [FILE_NAME] [FOLDER] [CACHE_PATH] [EMBEDDINGS_PATH]

	`FILE_NAME`: input .csv file with columns: `ROW_ID` (Unique instance identifier (string)), `DIAGNOSIS` (suspected diagnosis for the instance), `NOTE_ID` (EHR note identifier), `TEXT` (EHR note), `NOTE_TYPE` (For example, radiology, discharge summary)
	`FOLDER`: Folder that contains the above input file. This is also the output folder <br>
	`CACHE_PATH`: Huggingface cache path <br>
	`EMBEDDINGS_PATH`: Path of embeddings generated from run_embed_risk_factors.sbatch <br>


**Automatic Evaluation**

1. Extract risks and signs from the evidence. Evaluate each extracted risk/sign. (Update evidence filename, suffix, folder arguments)

		cd scripts
		sbatch run_extract_risks_signs.sbatch

2. Find hallucinations. The risks/signs extracted above may not be present in the note, so we need to detect these. (Update evidence filename, suffix, folder arguments)

		cd scripts
		sbatch run_find_hallucinations.sh




