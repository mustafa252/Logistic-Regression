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

# check null values
titanic.isnull().sum()

# ckeck the ratio of null-values
print(titanic.isnull().sum()['deck'] / titanic.shape[0])
print(titanic.isnull().sum()['age'] / titanic.shape[0])


# show hearmap of null values
plt.rcParams['figure.figsize'] = [8,7]
plt.rcParams['figure.dpi'] = 300
sns.heatmap(titanic.isnull())
sns.heatmap(titanic.isnull(), cmap='viridis', cbar=False)


# show histogram of the column 'age'
ax = titanic['age'].hist(bins=30, color='teal', grid=False,
                            alpha=0.8, density=True)

# create density
titanic['age'].plot(kind='density')
ax.set_xlabel('Age')
                    
