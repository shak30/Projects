import pandas as pd
import numpy as np
import seaborn as sns

df= pd.read_csv("C:\\Users\\shsingh4139\\Downloads\\income.csv")

data=df.copy()

data.info()

print(data.isnull().sum())

# for numerical variables
summary=data.describe()
print(summary)

# for categorical variables 
summary_cat=data.describe(include='O')
print(summary_cat)

#freq of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#cheking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

data=pd.read_csv("C:\\Users\\shsingh4139\\Downloads\\income.csv",na_values=[" ?"])

#============================
#Data pre-processing
#============================

data.isnull().sum()

missing=data[data.isnull().any(axis=1)]

data2=data.dropna(axis=0)

#checking correlation between independent variables
correlation= data2.corr()

#checking the relation between categorical variables 
data2.columns

#checking gender proportion
gender=pd.crosstab(index=data2["gender"],columns='count',normalize=True)

#gender vs salarystatus
gen_vs_salstat=pd.crosstab(index=data2['gender'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index')
