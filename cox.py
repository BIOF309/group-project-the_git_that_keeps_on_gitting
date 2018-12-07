import pandas as pd
# from sklearn.model_selection import train_test_split
# import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('df_merged5.csv')

# array = df.values
# predictors = array[:, 2:-1]
# outcome = array[:, -1]
df.head()
# futime = df.iloc[:, 1]
outcome = df.iloc[:, -1]
predictors = df.iloc[:, 1:-1]

mod = smf.phreg("PERMTH_INT ~ DMPFSEQ_x + BDPEXFLR + BDPSCAN",
                predictors, status=outcome, ties="efron")

result = mod.fit()
print(result.summary())

# X_train, X_test, y_train, y_test = train_test_split(predictors, outcome)
