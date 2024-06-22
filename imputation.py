# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
# linear regression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
# metrics


# load data from seaborn
titanic = sns.load_dataset('titanic')


# analys the col = 'age'
sns.displot(data=titanic,
            x='age',
            kde=True,
            bins=20,
            hue='survived',
            multiple='stack',
            col='sex')

                    
# columns
titanic.columns

# sex counts
female_1, male_1 = titanic['sex'].value_counts()
# 1 column based other column
female_2, male_2 = titanic[titanic['survived']==1]['sex'].value_counts()

print(male_1, male_2)
print(female_1, female_2)

# ratio of survivers
print(f'female surviver ratio {female_2/314} \n male surviver ratio {male_2/577}')


##########################################################
###### mean values

''' calculate the mean values '''
titanic['age'].mean()
titanic[titanic['sex']=='male']['age'].mean()
titanic[titanic['sex']=='female']['age'].mean()

''' display the mean values '''
sns.catplot(x='pclass', y='age',data=titanic,
            kind='box', height=3,
            aspect=2, hue='sex')



##########################################################
###### imputation function

''' Imputation '''
def imputation(pclass, sex):
    
    # get mean value of males in each pclass(1,2,3)
    if sex == 'male':
        if pclass == 1:
            #filtering
            return titanic[ (titanic['pclass']==1) & (titanic['sex']=='male') ]['age'].mean()

        elif pclass == 2:
            #filtering
            return titanic[ (titanic['pclass']==2) & (titanic['sex']=='male') ]['age'].mean()

        elif pclass == 3:
            #filtering
            return titanic[ (titanic['pclass']==3) & (titanic['sex']=='male') ]['age'].mean()


    # get mean value of females in each pclass(1,2,3)
    else:
        if pclass == 1:
            #filtering
            return titanic[ (titanic['pclass']==1) & (titanic['sex']!='male') ]['age'].mean()

        elif pclass == 2:
            #filtering
            return titanic[ (titanic['pclass']==2) & (titanic['sex']!='male') ]['age'].mean()

        elif pclass == 3:
            #filtering
            return titanic[ (titanic['pclass']==3) & (titanic['sex']!='male') ]['age'].mean()


# apply imputation for age with null values only
# titanic['age'] = titanic.apply(lambda x: imputation(x['pclass'], x['sex']) if np.isnan(x['age']) else x['age'], axis=1)
titanic['age'].isnull().sum() 


###########################################################
# Or we can use:

from sklearn.impute import SimpleImputer


''' by sklearn'''
print(titanic['age'].isnull().sum())
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age = titanic.loc[:,'age':'age'].values
age
imputer.fit(age)
age = imputer.transform(age)
print(age)
print(age.shape)

titanic['age'] = age
titanic['age']



##################################################################################
###### missing values in Embark_town

''' show missing values '''
titanic.isnull().sum()

''' show missing values in Embark_town '''
titanic['embark_town'].isnull().sum()

''' count values in Embark_town '''
titanic['embark_town'].value_counts()

''' fill with null values '''
titanic['embark_town'].fillna('Southampton', inplace=True)
titanic['embark_town'].isnull().sum()

''' drop deck & embark columns'''
titanic.drop(['embarked', 'deck'], inplace=True, axis=1)
titanic['embark_town'].isnull().sum()
