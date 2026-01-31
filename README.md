* P1(15%): 9.20 / 10.00,  Document (45%): 8.5; Presentation (35%): 9.5; Answers (20%): 10
* P2(30%): 7.15 / 10.00,  a) 5,6, b) 8,25, c) 9
* P3(45%): 7.15 / 10.00,  a) 5,6, b) 8,25, c) 9


This project is splited in 3 areas, db extraction, modeling and api. now we obtain extra data in the file /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/resources_p3/df_icu.csv, i want that you create a new file and improve the model with the new states like sofa, ibm ... try to extract as much information as possible and you have to impove the results that /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/mortality_prediction_models.py

PROMPT 1: 
explain me without changing anything what this doc does /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/A3_GroupM (1).ipynb



PROMTP 2: 
okey with that dataframe what woudl be the best way to create a predictive model of mortalitiy using the maximum variables possibles?



## prompt 3 
Yo need to create a prediction mortality model of the 24h after enter in the icu. for that task: 
1. read the 5 first lines of the dataset to understand its columns and format /Users/arriazui/Desktop/master/ELECTRONIC_HEALTH_RECORDS/P3/db_processing/A3_v2_all_sofa_extra.ipynb 

2. that dataset would be the input for the model 
3. select the best prediction model for that structure of the dataset 
4. in a new file, using python generate the prediction model 

Some suggestions: 
Enhanced Clinical Model__ (Add First 24h Physiology)
focus in the strucutre of the dataset and generate the model 
__Add these SOFA-related variables:__

- __Respiratory:__ pf_ratio, resp_score
- __Cardiovascular:__ map_min, cv_score, any_vaso
- __Liver:__ bilirubin, liver_score
- __CNS:__ gcs, cns_score
- __Coagulation:__ platelets, sofa_coag
- __Renal:__ creatinine_mgdl, urine_ml_24h, renal_score
- __Total severity:__ total_sofa