<H3>ENTER YOUR NAME : MAMTHA I </H3>
<H3>ENTER YOUR REGISTER NO : 212222230076 </H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs

Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
### STEP 1: 
Importing the libraries<BR>

###  STEP 2:
Importing the dataset<BR>

###  STEP 3:
Taking care of missing data<BR>

###  STEP 4:
Encoding categorical data<BR>

###  STEP 5:
Normalizing the data<BR>

### STEP 6:
Splitting the data into test and train<BR>

##  PROGRAM:

### IMPORT ALL LIBRARIES : 
```
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

```
### READ THE DATASET:
```
df=pd.read_csv("/content/Churn_Modelling (1).csv")
```
###  PRINTING HEAD & TAIL :
```
df.head()
df.tail()
df.columns
```
### CHECKING THE MISSING DATA :
```
df.isnull().sum()
```
### CHECKING FOR DUPLICATES :
```
df.duplicated()
```

### Y VALUE :
```
y = df.iloc[:, -1].values
print(y)
```
### CHECKING FOR OUTLIERS :
```
df.describe()
```
### DROPPING STRING VALUES :
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)

data.head()

```
### NORMALIZE THE DATA SET :
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
### SPLIT THE DATASET :
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
### TRAINING AND TESTING THE DATASET MODEL :

```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))

```






## OUTPUT:
***DATA SET:***




![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/103a33b0-e360-4814-8ef5-9bf8faf83683)


***FINDING ALL MISSING VALUE:***



![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/f7ba5cc1-f38c-4f6c-b7b6-d69050d3f297)


***CHECKING THE DUPLICATE VALUES:***



![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/29e2a9b0-3308-4bc9-ae10-412ee8e2e9ea)


***DECTECT OUTLIERS:***


![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/a65b5e85-be00-4477-a6b8-01852ef96915)


***NORMALIZE THE DATASETS:***


![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/3c59b074-278b-420f-9567-adb8db47bdba)


***SPLITING THE DATASETS INTO INPUT AND OUPUT:***


![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/a1f5e191-ad7f-4328-927c-cec538e33ea9)


***SPLITING THE DATASETS FOR TRAINING AND TESTING :***


![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/164e5051-721c-43b2-8f0f-84dfd3b72bcb)



![image](https://github.com/Mamthaiyappaprabu/Ex-1-NN/assets/119393563/f8861104-af34-4b1d-917a-d312ee6660dc)














## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


