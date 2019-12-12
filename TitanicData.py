#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #수학적 기능(평균 등 연산가능)
import pandas as pd #데이터 처리와 가공 가능(파일 부를 때 사용, train과 test데이터 연결 시 사용)
import matplotlib.pyplot as plt #데이터 시각화 (표)
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2.5) #폰트 사이즈 2.5로 함

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set() 


# In[3]:


df_train = pd.read_csv('D:\/train.csv') #train파일을 불러와 df_train객체에 저장
df_test = pd.read_csv('D:\/test.csv') #test파일을 불러와 df_test객체에 저장


# In[4]:


df_train.head(20) #train내용의 20줄을 읽어들임


# In[5]:


df_train.describe() #df_train데이터의 통계치 반환 (mean은 평균값)


# In[6]:


df_test.describe() #df_test데이터의 통계치 반환(mean은 평균값)


# In[9]:


for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg) #df_train에 각 column에 널데이터가 어느 정도있는지 확인하기 위한 코드 (age, cabin, embarked에 널데이터가 있음) 


# In[10]:


for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg) #df_test에 각 column에 널데이터가 어느 정도있는지 확인하기 위한 코드 (age, fare, cabin에 널데이터가 있음) 


# In[11]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show() #61.6%가 사망하였고, 38.4%가 생존했음을 보여줌 0은 사망이고, 1은 생존을 뜻함


# In[12]:


def pie_chart(feature): #파이차트를 그리기 위한 함수
    
    feature_ratio = df_train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = df_train[df_train['Survived'] == 1][feature].value_counts()
    dead = df_train[df_train['Survived'] == 0][feature].value_counts()

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
        plt.show()


# In[13]:


pie_chart('Pclass') #Pclass 비율과 각각 Pclass에 따른 생존률 파이차트


# In[14]:


pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r') 
#All 은 각 pclass마다 몇 명의 사람이 있는가를 나타내고, 1은 그 클래스 중에서 생존자, 0은 사망자를 의미함.


# In[15]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
#pclass 가 높을수록 생존 확률이 높음 (pclass는 1계급,2계급,3계급이 있고 1이 제일 좋음) -pclass1의 생존확률은 약 0.6 pclass2는 약 0.4 
#pclass3은 약 0.2의 생존 확률을 가짐.


# In[16]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()
#성별과 생존률 관계 (여성이 남성보다 생존률이 높음)


# In[17]:


pie_chart('Sex') #성별 비율과 각 성별에 따른 생존률 파이차트


# In[18]:


df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#성별에 따른 생존률 확률


# In[19]:


pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
#All은 총 사람 수이고, 1은 생존자, 0은 사망자


# In[20]:


fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
#생존과 나이의 관계 (Survived 가 1일땐, 생존, 0일땐, 사망)


# In[21]:


f, ax = plt.subplots(1, 1, figsize=(7, 7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
#탑승한 항구에 따른 생존률 (항구는 C,Q,S가 총 3가지가 있음)


# In[22]:


pie_chart('Embarked') #Embarked의 비율과 각각 Embarked에 따른 생존률


# In[23]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 
#•SibSp(형제,자매)와 Parch(부모님)를 FamilySize로 합침  


# In[24]:


f,ax=plt.subplots(1, 3, figsize=(40,10))
sns.countplot('FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
#가족크기와 생존률의 관계 (4명일 때가 생존률이 제일 높음)


# In[25]:



fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
#운임요금 히스토그램 그래프


# In[26]:


df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() 
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
#운임 요금의 데이터를 log 값 취함


# In[27]:


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
#운임 요금의 데이터를 log 값 취한 그래프


# In[28]:


df_test["Fare"].fillna(df_test.groupby('Pclass')['Fare'].transform('median'), inplace=True)
#test파일에 운임요금 널값들을 채워줌
#운임요금은 클래스와 관련이 크므로 클래스의 티켓값 중간값을 넣어줌


# In[29]:


df_train['FareBand'] = pd.qcut(df_train['Fare'],4,labels=[0,1,2,3])
df_test['FareBand'] = pd.qcut(df_test['Fare'],4,labels=[0,1,2,3])

df_train.head()
#운임요금을 0,1,2,3 총 4단계로 나눔


# In[30]:


df_train.Cabin.value_counts()
#Cabin값 출력


# In[31]:


for dataset in df_train:
    df_train['Cabin'] = df_train['Cabin'].str[:1]
for dataset in df_test:
    df_test['Cabin'] = df_test['Cabin'].str[:1]
#Cabin의 알파벳을 추출함    


# In[32]:


df_train.Cabin.value_counts()
#Cabin은 C가 제일 많고, B,D,E순으로 많다.


# In[33]:


Pclass1 = df_train[df_train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = df_train[df_train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = df_train[df_train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
#계급(Pclass)에 따른 Cabin값들을 분류해서 저장
#1등석(1계급)에 Cabin C,B,D,E,A가 몰려있음.


# In[34]:


df_train['Cabin'] = df_train['Cabin'].map({'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F':2, 'G':2.4, 'T': 2.8})
df_test['Cabin'] = df_test['Cabin'].map({'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F':2, 'G':2.4, 'T': 2.8})
#Cabin의 알파벳들을 수치화시킴


# In[35]:


df_train['Cabin'].fillna(df_train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
df_test['Cabin'].fillna(df_test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
#Cabin값중에서 널데이터들은 Pclass에 따른 Cabin값의 중간값을 넣어줌


# In[36]:


import re

def find_M(datas):   
    #wow = re.search("M[a-z]+\.", datas)
    wow = re.search("[A-Z][a-z]+\.", datas)
    
    if wow == None:
        return None
    
    return wow.group(0)
#name데이터의 이니셜을 뽑아냄


# In[37]:


df_train["Name_convert"]=df_train["Name"].map(find_M)
df_test["Name_convert"]=df_test["Name"].map(find_M)


# In[38]:


df_train["Name_convert"].unique()
#이름의 이니셜 종류를 나열함


# In[39]:


df_train["Name_convert"].value_counts()
#각 이니셜별로 사람 수를 나타냄


# In[40]:


df_train[df_train["Name_convert"]=='Dr.']
#이니셜 Dr 7명중, 6명이 남자고, 1명이 여자라 Dr는 Mr와 동일하게 처리하고, 여자 한명은 Mr.s로 변경


# In[41]:


df_train[df_train["Name_convert"]=='Rev.']
#이니셜 Rev 6명 모두가 남자라, Mr와 동일하게 처리


# In[42]:


df_train['Initial'] =  df_train.Name_convert.map({"Mr.":0, "Miss.":1, "Mrs.":2, "Master.":3,                                                      "Dr.":0, "Rev.":0, "Major.":0, "Col.":0, "Sir.":0,                                                       "Lady.":1, "Ms.":1, "Mlle.":1 } )
df_test['Initial'] =  df_train.Name_convert.map({"Mr.":0, "Miss.":1, "Mrs.":2, "Master.":3,                                                      "Dr.":0, "Rev.":0, "Major.":0, "Col.":0, "Sir.":0,                                                       "Lady.":1, "Ms.":1, "Mlle.":1 } )

#이니셜을 수치화시킴
#상위 4개값 Mr. Miss. Mrs. Master. 은 0~3 으로 매칭시키고, Dr. Rev. Major. , Col. Sir. 은 Mr.과 같은 값으로 처리한다.
#Lady., Ms. Mlle. 는 Miss. 와 같은 값으로 처리한다. 
#나머지 이니셜은 4 로 매칭시킨다.


# In[43]:


df_train[df_train['Initial'].isnull()]
#매칭되지 않은 데이터들이 있는 지 확인


# In[44]:


df_train['Initial'].fillna(4, inplace=True)
df_test['Initial'].fillna(4, inplace=True)
#매칭되지 않은 데이터는 4로 매칭시킴


# In[45]:


df_train.loc[ (df_train['Initial']==0)  &  (df_train['Sex']==1), ['Initial']] = 2
#name_code 가 (0) Mr.인데, 성별을 여성으로 변환되었던 데이터는 (2)Mrs. 로 수정해줌


# In[46]:


sns.factorplot(x='Initial' , y='Survived', data=df_train)
plt.show()
#이니셜에 따른 생존률 그래프
#0은 Mr,"Dr.,Rev.,Major., Col., Sir. -제일 생존률이 낮음
#1은 Miss. Lady., Ms., Mlle. 
#2는 Mrs. -제일 생존률이 높음
#3은 Master.
#4는 나머지 이니셜


# In[47]:


print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
#탑승 항구는 Null데이터가 2개있음


# In[48]:


df_train['Embarked'].value_counts() #항구에 따라 탑승객이 수, S가 가장 많은 승객이 탑승했음


# In[49]:


df_train['Embarked'].fillna('S', inplace=True)
#S에 가장 많은 승객이 탑승했으므로 Null데이터를 S로 채워줌


# In[50]:


df_train.groupby('Initial').mean()
#이니셜별로 나이평균


# In[51]:


df_train.loc[(df_train.Age.isnull())&(df_train.Initial==0),'Age'] =33 
df_train.loc[(df_train.Age.isnull())&(df_train.Initial==1),'Age'] = 22
df_train.loc[(df_train.Age.isnull())&(df_train.Initial==2),'Age'] = 36
df_train.loc[(df_train.Age.isnull())&(df_train.Initial==3),'Age'] = 5
df_train.loc[(df_train.Age.isnull())&(df_train.Initial==4),'Age'] = 41

df_test.loc[(df_test.Age.isnull())&(df_test.Initial==0),'Age'] = 33
df_test.loc[(df_test.Age.isnull())&(df_test.Initial==1),'Age'] = 22
df_test.loc[(df_test.Age.isnull())&(df_test.Initial==2),'Age'] = 36
df_test.loc[(df_test.Age.isnull())&(df_test.Initial==3),'Age'] = 5
df_test.loc[(df_test.Age.isnull())&(df_test.Initial==4),'Age'] = 41
#이니셜에 따른 평균 나이 널값에 추가함


# In[52]:


df_train['Age_cat'] = 0
df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0
df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1
df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2
df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3
df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4
df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5
df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6
df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7

df_test['Age_cat'] = 0
df_test.loc[df_test['Age'] < 10, 'Age_cat'] = 0
df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1
df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2
df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3
df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4
df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5
df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6
df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7

#나이를 10살간격으로 나눠 총 8개의 그룹으로 나눔


# In[53]:


def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7    
    
df_train['Age_cat_2'] = df_train['Age'].apply(category_age)
# 간단한 함수를 만들어 apply 메소드에 넣어주는 방법


# In[54]:


df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)
#중복되는 Age_cat과 원래 Age 를 제거함


# In[55]:


df_train['Embarked'].unique()
#탑승 항구가 S,C,Q임을 알  수 있음


# In[56]:


df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
#탑승 항구 데이터를 수치화시켜줌


# In[57]:


df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
#성별 데이터를 수치화시켜줌


# In[58]:


heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Embarked', 'FamilySize', 'Initial', 'Age_cat','FareBand','Cabin']] 

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

del heatmap_data
#상관관계를 heatmap으로 표현하여 구함


# In[59]:


df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
 #one hot encoding함
#하나의 값만 True(1)로 표현하고, 나머지 값들은 False(0)로 표현하는 것
#분류하는 문제일 때, 정확성을 높여주고, 다양한 모델에 적용이 가능함 
#initial 데이터- 이름(Mr같은) 데이터를 one hot encoding함


# In[60]:


df_train = pd.get_dummies(df_train, columns=['FareBand'], prefix='FareBand')
df_test = pd.get_dummies(df_test, columns=['FareBand'], prefix='FareBand')
#FareBand데이터- 운임요금 데이터를 one hot encoding함


# In[61]:


#df_train = pd.get_dummies(df_train, columns=['Cabin'], prefix='Cabin')
#df_test = pd.get_dummies(df_test, columns=['Cabin'], prefix='Cabin')
#Cabin 데이터를 one hot encoding함


# In[62]:


df_train = pd.get_dummies(df_train, columns=['Pclass'], prefix='Pclass')
df_test = pd.get_dummies(df_test, columns=['Pclass'], prefix='Pclass')
#Pclass 데이터를 one hot encoding함


# In[63]:


df_train = pd.get_dummies(df_train, columns=['Sex'], prefix='Sex')
df_test = pd.get_dummies(df_test, columns=['Sex'], prefix='Sex')
#Sex 데이터를 one hot encoding함


# In[64]:


df_train = pd.get_dummies(df_train, columns=['Age_cat'], prefix='Age_cat')
df_test = pd.get_dummies(df_test, columns=['Age_cat'], prefix='Age_cat')
#Age_cat 데이터를 one hot encoding함


# In[65]:


#df_train = pd.get_dummies(df_train, columns=['FamilySize'], prefix='FamilySize')
#df_test = pd.get_dummies(df_test, columns=['FamilySize'], prefix='FamilySize')
#FamilySize 데이터를 one hot encoding함


# In[66]:


df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket','Name_convert','Fare','Embarked'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket','Name_convert','Fare','Embarked'], axis=1, inplace=True)
#필요없는 요소들은 지움


# In[67]:


#머신러닝
from sklearn import metrics # 모델의 평가 위함
from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수
from sklearn.ensemble import RandomForestClassifier  #RandomForestClassifier를 구현하기 위함
from sklearn.neighbors import KNeighborsClassifier #KNN을 구현하기 위함
from sklearn.tree import DecisionTreeClassifier #결정트리를 구현하기 위함
from sklearn.naive_bayes import GaussianNB #나이브 베이즈를 구현하기 위함
from sklearn.svm import SVC #SVM을 구현하기 위함


# In[68]:


X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values


# In[69]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)


# In[70]:


model_RF = RandomForestClassifier(n_estimators=15) #랜덤포레스트 머신러닝 - 결정트리기반 모델 15개의 이웃을 기준으로 정함
model_RF.fit(X_tr,y_tr)
RF_prediction = model_RF.predict(X_vld)


# In[71]:


print('{:.2f}% 정확도'.format( 100 * metrics.accuracy_score(RF_prediction, y_vld)))


# In[72]:


model_NB=GaussianNB() #나이브베이즈 머신러닝
model_NB.fit(X_tr,y_tr)
NB_prediction = model_NB.predict(X_vld)


# In[73]:


print('{:.2f}% 정확도'.format( 100 * metrics.accuracy_score(NB_prediction, y_vld)))


# In[74]:


model_SV=SVC() #SVM 머신러닝
model_SV.fit(X_tr,y_tr)
SV_prediction = model_SV.predict(X_vld)


# In[75]:


print('{:.2f}% 정확도'.format( 100 * metrics.accuracy_score(SV_prediction, y_vld)))


# In[76]:


model_DT = DecisionTreeClassifier() #결정트리 머신러닝
model_DT.fit(X_tr,y_tr)
DT_prediction = model_DT.predict(X_vld)


# In[77]:


print('{:.2f}% 정확도'.format( 100 * metrics.accuracy_score(DT_prediction, y_vld)))


# In[78]:


model_KN=KNeighborsClassifier(n_neighbors = 15) #KNN 머신러닝 15개의 이웃을 기준으로 정함
model_KN.fit(X_tr,y_tr)
KN_prediction = model_KN.predict(X_vld)


# In[79]:


print('{:.2f}% 정확도'.format( 100 * metrics.accuracy_score(KN_prediction, y_vld)))


# In[80]:


submission = pd.read_csv('D:\/gender_submission.csv')


# In[81]:


prediction = model_KN.predict(X_test)
submission['Survived'] = prediction


# In[82]:


submission.to_csv('D:\/finish3.csv', index=False)


# In[ ]:




