
# Exploring Cancer Mortality Factors

## Notes: 
The purpose of this project is to replicate Dr. Martin Skarzynski's capstone project using Python. Dr. Skarzynski's capstone project was completed using R. It is titled "Exploratory Analysis of Factors Associated with Cancer Mortality in the National Health and Nutrition Examination Survey Dataset. This capstone project was in partial fulfillment of Dr. Skarzynski's MPH degree from Johns Hopkins University School of Public Health. 

One major modification from Dr. Skarzynski's capstone is that in this project, the top 25 correlated predictor variables were chosen before inputing the data into a Cox Proportional Hazard Model. 

The capstone project, datasets used, and R codes can be found in the following link: https://github.com/marskar/nhanes. 

## Background:
The scale of cancer burden in the United States is huge with millions of new cancer cases a year. Susceptibility to cancer has been organized into modifiable and non-modifiable factors. Modifiable factors include body mass index (BMI) and cigarette use while non-modifiable factors include age, sex, and race/ethnicity. Modifiable cancers are responsible for a huge fraction of all cancer cases and cancer prevention interventions that target modifiable risk factors have been shown to be effective in reducing cancer cases and deaths. By using data science to map out cancer risk prediction models this could help pinpoint those at greatest risk for cancer. To do this a cancer risk prediction model much include both modifiable and non-modifiable risk factors. We started by analyzing data from the Third National Health and Nutrition Examination Survey (NHANES III) and the National Death Index (NDI) Public-Use Linked Mortality Files. The goal of our analysis was to examine the NHANES III data and identify variables useful for prediction of mortality risk in python. 

## Methods:
The Third National Health and Nutrition Examination Survey (NHANES III) consisted of interview, medical examination, and laboratory data sets. This data was collected from 1988 to 1994 in the United States. NHANES III was linked with mortality data from the NDI death records by the National Center for Health Statistics. 

The exclusion factors included participants who: 
1. were under 18 years of age, 
2. had a self-reported cancer at baseline, and 
3. had missing values for variables related for the study (SDDPSU6, SDSTRA6, WTPFQX6)

In this study, the model used to relate the time that passed (before cancer death occurred) to risk factors was the Cox proportional hazard model. Specifically, the follow-up time variable (XXX) and cancer death were used as the survival outcome in the regression analysis. 

Using Python's **pandas**, we: 
1. Read the csv datasets into dataframes (i.e., adult, lab, exam, mort).
2. Removed participants with missing values for cause of death (UCOD_LEADING) or missing follow-up time from interview (PERMTH_INT).
3. Merged all datasets using the participant identifier variables (SEQN).
4. Removed participants with missing values for primary sampling units (SDPPSU6), stratificatioin (SDPSTRA6), and sampling weight (WTPFQX6).
5. Removed baseline cancer cases using the HAC1N and HAC1O variables.
6. Created a cancer mortality variable based on whether "Malignant neoplasms" (C00-C97) was recorded as the cause of death
7. Remove variables (columns) that were non-numeric or only had one unique value 
8. Select variables (columns) with less than 10% missing values (NaN

Using Python's **Scikit-Learn**, we:
1. Removed highly correlated variables (r≥0.9).
2. Chose the 25 most correlated predictor variables (for cancer_mort outcome).  

Using Python's **Lifelines**, we applied the Cox Proportional Hazards Model to the 25 predictor variables mentioned above. 
> In addition to the 25 selected predictor variables, the model also included:> 
> 1. A "survival object" created from the "cancer mortality" event and follow-up time variables.
> 2. A "design object" created from the "primary sampling unit" (SDDPSU6), "Stratification" (SDSTRA6), and "Weight" (WTPFQX6) variables. 

Using Python's **Plotly**, the Cox Proportional Hazard model was represented in the form of a volcano plot. 

## Results:
## Discussion:
With the use of variables fitted to a cancer risk prediction model, this can be an effective strategy to explore large datasets such as NHANES III.

## Conclusion: 
## References: 
