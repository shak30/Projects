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

#======================================
#Cheking correlation between variables
#======================================

#checking correlation between independent variables
correlation= data2.corr()

#checking the relation between categorical variables 
data2.columns

#checking gender proportion
gender=pd.crosstab(index=data2["gender"],
                   columns='count',
                   normalize=True)

#gender vs salarystatus
gen_vs_salstat=pd.crosstab(index=data2['gender'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index')

#Freq distribution of salarystatus 
salstat=sns.countplot(data2['SalStat'])

#for age distribution
sns.distplot(data2['age'],bins=10,kde=False)
## People with age 20-45 are high in frequency

#Age vs salarystatus
sns.boxplot('SalStat','age',data=data2)

data2.groupby('SalStat')['age'].median()

#jobtype vs salarystatus
jobtype_vs_salstat=pd.crosstab(index=data2['JobType'],
                               columns=data2['SalStat'],
                               margins=True,
                               normalize='index')

#education vs salarystatus
edtype_vs_alstat=pd.crosstab(index=data2['EdType'],
                               columns=data2['SalStat'],
                               margins=True,
                               normalize='index')

#occupation vs salarystatus
occupation_vs_salstat=pd.crosstab(index=data2['occupation'],
                               columns=data2['SalStat'],
                               margins=True,
                               normalize='index')

#hoursperweek vs salarystatus
hoursperweek=sns.boxplot('hoursperweek','SalStat',data=data2)

#capital gain and loss
sns.distplot(data2['capitalgain'],bins=10,kde=False)

sns.distplot(data2['capitalloss'],bins=10,kde=False)


#=======================================================
       #LOGISTIC REGRESSION
#=======================================================

# reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

#Storing the column names 
column_list=list(new_data.columns)
print(column_list)

#Seperating the input names from data
features=list(set(column_list)-set(['SalStat']))
print(features)

#storing the column values in Y
y=new_data['SalStat'].values
print(y)

#storing the input values from features in X
x=new_data[features].values
print(x)

#splitting the data into train and test

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#make an instance of the model
logistic=LogisticRegression()

# fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_


# prediction from test data
prediction = logistic.predict(test_x)

#confusion matrix
confusion_matrix= confusion_matrix(test_y, prediction)
print(confusion_matrix)

#calculating the accuracy
accuracy_score= accuracy_score(test_y, prediction)
print(accuracy_score)

#Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())


#=============================================================================
  #LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
#=============================================================================
  
# # reindexing the salary status names to 0,1
# # data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
# print(data2['SalStat'])


# cols=['gender','nativecountry','race','JobType']
# data2= data2.drop(cols,axis=1)
      
# new_data1=pd.get_dummies(data2, drop_first=True)

# #Storing the column names 
# column_list=list(new_data1.columns)
# print(column_list)

# #Seperating the input names from data
# features=list(set(column_list)-set(['SalStat']))
# print(features)

# #storing the column values in Y
# y1=new_data1['SalStat'].values
# print(y)

# #storing the input values from features in X
# x1=new_data1[features].values
# print(x)

# #splitting the data into train and test

# train_x1,test_x1,train_y1,test_y1=train_test_split(x1,y1,test_size=0.3,random_state=0)

# #make an instance of the model
# logistic1=LogisticRegression()

# # fitting the values for x and y
# logistic1.fit(train_x1,train_y1)
# logistic1.coef_
# logistic1.intercept_


# # prediction from test data
# prediction = logistic1.predict(test_x1)

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# #confusion matrix
# confusion_matrix1= confusion_matrix(test_y1, prediction)
# print(confusion_matrix1)

# #calculating the accuracy
# accuracy_score1= accuracy_score(test_y1, prediction)
# print(accuracy_score1)

# #Printing the misclassified values from prediction
# print('Misclassified samples: %d' % (test_y != prediction).sum())



