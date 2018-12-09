
# coding: utf-8

# In[1]:


import pandas as pd
from functools import reduce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm


# In[2]:


# Step = Python's pandas #1
# Read csv datasets into dataframes (adult, lab, exam, mort)
mort = pd.read_csv(r'C:\Users\hans11\Desktop\python_project\mort.csv')
lab = pd.read_csv(r'C:\Users\hans11\Desktop\python_project\lab.csv')
adult = pd.read_csv(r'C:\Users\hans11\Desktop\python_project\adult.csv', low_memory=False)
exam = pd.read_csv(r'C:\Users\hans11\Desktop\python_project\exam.csv', low_memory=False)


# In[3]:


""" Step = Python's pandas #2; clean NHANES mortality (mort) dataframe
    Remove columns that start with 'MORTSRCE' """

for col in mort.columns:
    if 'MORTSRCE' in col:
        del mort[col]
mort.info()
# 33,994 entries, 9 columns


# In[4]:


""" Step = Python's pandas #2; clean NHANES mortality (mort) dataframe
    Remove those with missing UCOD_LEADING or PERMTH_INT
        UCOD_LEADING = underlying cause of death recode from UCOD_113 (Leading causes)
        PERMTH_INT = person months of follow-up from interview date """ 
clean_mort = mort[(mort.UCOD_LEADING.notnull() & mort.PERMTH_INT.notnull())]
clean_mort.info()
# clean_mort = 19,592 entries, 9 columns


# In[5]:


# delete PERMTH_EXM column (PERMTH_INT will be used in final cox model) 
clean_mort2 = clean_mort.drop('PERMTH_EXM', 1)
clean_mort2.info()

# clean_mort = 19,592 entries; 8 columns


# In[6]:


""" Step = Python's pandas #2 cont.; clean NHANES mortality (mort) dataframe
    Remove those with missing (blank) MORTSRCE_NDI
        MORTSRCE_NDI = mortality status was ascertained through a probabilistic match to a
        National Death Index (NDI) record.
            1 = Yes (i.e., mortality source info. is available)
            Blank = Assumed alive, ineligible for mortality follow-up, or under age 18

clean_mort2 = clean_mort.dropna(subset=['MORTSRCE_NDI'])
clean_mort2.info()
# clean_mort2 = 6,596 entries

"""


# In[7]:


# change SEQN column in lab, adult, exam files to numeric (int)
lab['SEQN'] = lab['SEQN'].astype(int)
adult['SEQN'] = adult['SEQN'].astype(int)
exam['SEQN'] = exam['SEQN'].astype(int)


# In[8]:


""" Step = Python's pandas #3: merge all datasets using participant identifier variable (SEQN)
    into merged dataframe (df_merged) """ 
dfs = [clean_mort2, lab, adult, exam]
df_merged = reduce(lambda left,right: pd.merge(left,right,on='SEQN'), dfs)
df_merged.info()
# df_final = 17,738 entries, 3,967 columns


# In[9]:


# Exclusion #1: check age (HSAGEIR) range for dataset
    # HSAGEIR = age at interview
df_merged.HSAGEIR.describe()
# Age range = 18-90 


# In[10]:


""" Step = Python's pandas #4: remove those with missing values for SDPPSU6, SDPSTRA6, and WTPFQX6 
df_merged.dropna(subset=['SDPPSU6'])
df_merged.dropna(subset=['SDPSTRA6'])
df_merged.dropna(subset=['WTPFQX6'])
df_merged.info()
""" 


# In[11]:


""" Step = Python's pandas #5: 
    Exclusion #2: had a self-reported cancer at baseline
        HAC1N = 'doctor ever told you had: skin cancer'
        HAC1O = 'doctor ever told you had: other cancer'
            1 = Yes
            2 = No
            9 = Don't know 
    Only keep those who answered 2 for HAC1N or HAC1O"""
df_merged = df_merged[(df_merged.HAC1N == 2) & (df_merged.HAC1O == 2)]
df_merged.info()
#df_merged = 16,404 entries; 3,967 columns


# In[12]:


# List all unique entries for UCOD_LEADING column
df_merged.UCOD_LEADING.unique()


# In[13]:


""" Step = Python's pandas #6
        create cancer mortality variable based on whether 'malignant neoplasms (COO-C97)'
        was recorded as the cause of death (UCOD_LEADING) """
def cancer_mort(df_merged):
    if df_merged['UCOD_LEADING'] == 'Malignant neoplasms (C00-C97)':
        return 1
    else:
        return 0
df_merged['cancer_mort'] = df_merged.apply(cancer_mort, axis=1)

df_merged.info()

# 16,404 entries; 3,968 columns


# In[14]:


# Frequency table (crosstabs) using pandas pd.crosstab() function
# For cancer_mort outcome

cancer_mort_freq = pd.crosstab(index=df_merged['cancer_mort'],
                              columns = 'count')
cancer_mort_freq

# 964 cancer deaths; 15,440 other deaths


# In[15]:


""" Step = Python's pandas #7: remove variables that are non-numeric or only have one unique value"""

# deleted variables (columns) that are non-numeric: 
df_merged2 = df_merged._get_numeric_data()
df_merged2.info()
# df_merged2 = 16,404 entries, 3,089 columns; all floats or ints


# In[16]:


# Cont. Python's pandas #7: delete variables (columns) with only one unique value:
for col in df_merged2.columns:
    if len(df_merged2[col].unique()) == 1:
        df_merged2.drop(col, inplace=True, axis=1)
        
df_merged2.info()

# df_merged2 = 16,404 entries, 2,763 columns; all floats and ints


# In[17]:


""" Step = 
Python's pandas #8: select columns with more than 90% missing values (NaN) and delete"""

limitPer = len(df_merged2) * 0.90
df_merged3 = df_merged2.dropna(thresh=limitPer, axis=1)

df_merged3.info()

# df_merged3 = 16,404 entries, 1,106 columns; all floats and ints


# In[18]:


""" # sum follow-up time column (PERMTH_INT) based on cancer_mort column
df_merged3.groupby('cancer_mort')['PERMTH_INT'].sum() """


# In[20]:


# delete columns that start with 'WTPQRP'
    # 'WTPQRP1 - WTPQRP52' = Fay's BRR Replicate weights for MEC- examined sample
for col in df_merged3.columns:
    if 'WTPQRP' in col:
        del df_merged3[col]

df_merged3.info()

# 16,404 entries, 950 columns; floats and ints


# In[ ]:


# generating correlation heat-map
corr = df_merged3.corr()
sns.heatmap(corr) 


# In[93]:


# compare correlation between variables and remove those that have a correlation higher than 0.9

columns = np.full((corr.shape[0]), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = df_merged3.columns[columns]
df_merged4 = df_merged3[selected_columns]

df_merged4.info()
# df_merged4 = 4,769 entries, 413 columns; all floats and ints
# df_merged4 = 964 cancer cases 


# In[26]:


""" # delete columns that start with 'WTPQRP'
    # 'WTPQRP1 - WTPQRP52' = Fay's BRR Replicate weights for MEC- examined sample
for col in df_merged4.columns:
    if 'WTPQRP' in col:
        del df_merged4[col]

df_merged4.info()

# 16,404 entries, 950 columns; floats and ints """


# In[118]:


# Delete entries with missing value (NaN) in any of the columns 
# Make all variables numeric type
df_merged5 = df_merged4.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_merged5.apply(pd.to_numeric)
df_merged5.isna().sum().sum()

# df_merged5 = 3,773 entries, 361 columns; all floats and ints
# df_merged5 = no missing value 

# 791 cancer cases 


# In[22]:


cancer_mort_freq = pd.crosstab(index=df_merged5['cancer_mort'],
                              columns = 'count')
cancer_mort_freq


# In[23]:


df_merged5.to_csv('df_merged5.csv', index=False)


# In[26]:


# converting all values in df_merged5 to integer 
# due to error when running LogisticRegression
    # ValueError: Unknown label type: 'continuous'
    
df_merged6 = df_merged5.astype(int)
df_merged6.info()


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

array = df_merged6.values
predictors = array[:, 0:10]
outcome = array[:, 10]

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(predictors, outcome)

print("Num Features: %d"% fit.n_features_) 
print("Selected Features: %s"% fit.support_)
print("Feature Ranking: %s"% fit.ranking_)


# In[ ]:


""" Step = Python's Scikit-Learn #2: choose 25 most correlated predictor variables (for cancer_mort outcome)
        - Perform feature selection = process where you automatically select those features in data that 
        contribute mostto the prediction variable (cancer_mort). 
        - Logistic regression used b/c predictor variables are continuous and outcome (cancer_mort) is binary.
        - Used SelectKBest feature to select 25 most correlated predictor variables. 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

array = df_merged6.values
predictor = array[:, 0:412]
outcome = array[:,412]

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(predictor, outcome)
print('Num Feautres: %d') % fit.n_features_
print('Selected Features: %s') % fit.support_
print('Feature Ranking: %s') % fit.ranking_ 

ERROR = is not able to complete the run""" 

