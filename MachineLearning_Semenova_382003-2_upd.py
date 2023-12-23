#!/usr/bin/env python
# coding: utf-8

# # Постановка задачи

# Автомобильная компания планирует выйти на новый рынок с имеющимся арсеналом товаров. После проведенного исследования стало понятно, что поведение нового рынка идентично поведению старого. При работе со старым рынком отдел продаж сегментировал всех покупателей на 4 группы А, В, С, D. Эта стратегия хорошо сработала и на новом рынке. Так как было выявлено 2627 новых клиентов, то требуется спрогнозировать подходящую группу для этих клиентов.
# 
# База данных: https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation?datasetId=841888&language=Python

# In[1]:


import numpy as np              
import pandas as pd            
import matplotlib.pyplot as plt 
import seaborn as sns           
import sklearn


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# # Чтение данных

# Клиенты компании

# In[3]:


data = pd.read_csv('D:\\Train.csv')
data


# Потенциальные клиенты

# In[4]:


data_test = pd.read_csv('D:\\Test.csv')
data_test


# Всего в задаче 11 признаков:
# 1. ID - уникальный номер клиента
# 2. Gender - пол
# 3. Ever_Married - семейный статус
# 4. Age - возраст
# 5. Graduated - образование
# 6. Profession - профессия
# 7. Work_Experience - стаж работы
# 8. Spending_Score - оценка расходов
# 9. Family_Size - количество членов семьи
# 10. Var_1 - категория клиента
# 11. Segmentation - группа (сегмент) клиента

# In[5]:


data.info()


# In[6]:


data["Gender"] = data["Gender"].astype('category')
data["Ever_Married"] = data["Ever_Married"].astype('category')
data["Graduated"] = data["Graduated"].astype('category')
data["Profession"] = data["Profession"].astype('category')
data["Spending_Score"] = data["Spending_Score"].astype('category')
data["Var_1"] = data["Var_1"].astype('category')
data["Segmentation"] = data["Segmentation"].astype('category')


# In[7]:


data["Gender"].dtype,data["Ever_Married"].dtype,data["Graduated"].dtype,data["Profession"].dtype,data["Spending_Score"].dtype,data["Var_1"].dtype,data["Segmentation"].dtype 


# # Визуализация данных

# In[8]:


data.head(10)


# In[9]:


data.describe()


# Можно отметить существенную разницу в максимальном и минимальном возрасте клиентов в 71 год и отсутствие у множества клиентов опыта работы

# In[10]:


data.describe(include=['category'])


# Большая часть клиентов компании - мужчины с образованием и низким уровнем дохода, состоящие в браке и принадлежащие группе D

# In[11]:


data_corr = data.corr(numeric_only = True)
data_corr


# In[12]:


sns.heatmap(data_corr, annot=True, fmt='.2f', cmap='coolwarm', center = 0)


# Существует несильно выраженная отрицательная корреляция возраста клиента c количеством членов его семьи 

# In[13]:


sns.set_style('whitegrid')
numbers = pd.Series(data.columns)
data[numbers].hist(figsize=(7,7))
plt.show()


# У большинства клиентов компании мало опыта работы и меньше 4 членов семьи

# In[14]:


fig, ax = plt.subplots(figsize=(8, 4))
plt.scatter(x = data['Age'], y = data['Family_Size'], marker = ".", s = 30)
plt.xlabel("Age")
plt.ylabel("Family size")

plt.show()


# In[15]:


data.Gender.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Gender")

plt.show()


# In[16]:


sns.catplot(y='Gender', hue='Segmentation', data=data, kind='count')
plt.xlabel("")
plt.ylabel("")

plt.show()


# Мужчин-клиентов компании несколько больше, чем женщин. Распределения клиентов компании по группам для мужчин и женщин практически совпадают, за исключением только группы D, где больше всего мужчин

# In[17]:


data.Work_Experience.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Work experience")
plt.ylabel("")

plt.show()


# In[18]:


sns.catplot(y='Work_Experience', hue='Segmentation', data=data, kind='count')
plt.xlabel("")
plt.ylabel('Work experience')

plt.show()


# Множество клиентов автомобильной компании имеют малый опыт работы

# In[19]:


data.Family_Size.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Family size")

plt.show()


# In[20]:


sns.catplot(y='Family_Size', hue='Segmentation', data=data, kind='count')
plt.ylabel("Family size")
plt.xlabel("")

plt.show()


# Во многих семьях клиентов компании не больше 3 человек, количество членов семьи потенциального клиента оказывает существенное влияние на определение его в группу

# In[21]:


data.Ever_Married.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Ever married")

plt.show()


# In[22]:


sns.catplot(y='Ever_Married', hue='Segmentation', data=data, kind='count')
plt.ylabel("Ever married")
plt.xlabel("")

plt.show()


# Среди клиентов компании больше состоящих в браке людей. В группе D больше не состоящих в браке людей, в остальных группах наоборот

# In[23]:


fig, ax = plt.subplots(figsize=(10, 4))
data.Profession.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Profession")
plt.ylabel("")

plt.show()


# In[24]:


sns.catplot(y='Profession', hue='Segmentation', data=data, kind='count')
plt.ylabel("Profession")
plt.xlabel("")

plt.show()


# Среди клиентов компании большинство занимаются искусством или работают в сфере здравоохранения. Почти все мед. работники принадлежат группе D, а художники - группе C. В группе A много художников, докторов, инженеров и работников сферы развлечений.

# In[25]:


data.Graduated.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Graduated")
plt.ylabel("")

plt.show()


# In[26]:


sns.catplot(y='Graduated', hue='Segmentation', data=data, kind='count')
plt.ylabel("Graduated")
plt.xlabel("")

plt.show()


# Отношение числа имеющих образование к числу неимеющих образования клиентов равно примерно 5 : 3. В группе D преобладают клиенты без образования, а в остальных группах - с образованием. Наиболее явно эта зависимость выражена для группы C.

# In[27]:


data.Spending_Score.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Spending score")
plt.ylabel("")

plt.show()


# In[28]:


sns.catplot(y='Spending_Score', hue='Segmentation', data=data, kind='count')
plt.ylabel("Spending score")
plt.xlabel("")

plt.show()


# Примерно у 60% клиентов компании низкий уровень дохода, они занимают большую часть групп A и D

# In[29]:


data.Var_1.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("Var 1")
plt.ylabel("")

plt.show()


# In[30]:


sns.catplot(y='Var_1', hue='Segmentation', data=data, kind='count')
plt.ylabel("Var 1")
plt.xlabel("")

plt.show()


# Распределение клиентов внутри каждой группы примерно одинаковые, чаще преобладает группа D (за исключением Cat_6, где больше клиентов из группы C). Среди клиентов компании практически нет принадлежащих категориям Cat_1, Cat_5 и Cat_7

# In[31]:


data.Segmentation.value_counts().plot(kind='bar', rot = 0)
plt.xlabel("")
plt.ylabel("")

plt.show()


# В группе D несколько больше клиентов компании, нежели в других. В группах A, B и C клиентов практически поровну

# Диаграмма рассеивания относительно группы клиентов

# In[32]:


data  = data.drop(columns='ID')


# In[33]:


sns.pairplot(data.iloc[np.random.choice(np.arange(data.shape[0]), size=700, replace=False)], hue='Segmentation', diag_kind='hist')


# In[34]:


fig, axes = plt.subplots(2, 4, figsize=(20, 8))
sns.histplot(data=data, x='Gender', ax=axes[0,0]);
sns.histplot(data=data, x='Ever_Married', ax=axes[0,1]);
sns.histplot(data=data, x='Age', ax=axes[0,2]);
sns.histplot(data=data, x='Graduated', ax=axes[0,3]);
sns.histplot(data=data, x='Work_Experience', ax=axes[1,0]);
sns.histplot(data=data, x='Spending_Score', ax=axes[1,1]);
sns.histplot(data=data, x='Family_Size', ax=axes[1,2]);
sns.histplot(data=data, x='Var_1', ax=axes[1,3]);


# Дублирующиеся данные

# In[35]:


data = data.drop_duplicates()


# Пропущенные значения

# In[36]:


data_main = data.copy(deep = True)
data_main.head(10)


# In[37]:


A = data.isnull()
A.head()
print('Missing values:', A.sum(), sep='\n')


# In[38]:


data_train = data_main.copy(deep = True)


# In[39]:


A = data_train.isnull()
A.head()
print('Missing values:', A.sum(), sep='\n')


# In[40]:


data = data_main.copy(deep = True)
data.isnull().sum()


# In[41]:


data['Var_1'].mode()


# Заполним пропуски наиболее частыми значениями

# In[42]:


data['Ever_Married'].fillna('No', inplace=True)


# In[43]:


data['Graduated'].fillna('No', inplace=True)


# In[44]:


data.dropna(subset=["Profession"], inplace=True)


# In[45]:


data['Work_Experience'].fillna(0, inplace=True)


# In[46]:


data['Family_Size'].fillna(data_train['Family_Size'].mode, inplace=True)


# In[47]:


data['Var_1'].fillna("Cat_6", inplace=True)


# In[48]:


A = data.isnull()
A.head()
print('Missing values by features:', A.sum(), sep='\n')


# Категориальные признаки

# In[49]:


from sklearn.preprocessing import  LabelEncoder
le = LabelEncoder()

data_train['Gender'] = le.fit_transform(data_train['Gender'])
Gender_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Ever_Married'] = le.fit_transform(data_train['Ever_Married'])
Ever_Married_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Graduated'] = le.fit_transform(data_train['Graduated'])
Graduated_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Spending_Score'] = le.fit_transform(data_train['Spending_Score'])
Spending_Score_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Var_1'] = le.fit_transform(data_train['Var_1'])
Var_1_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Profession'] = le.fit_transform(data_train['Profession'])
Profession_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Family_Size'] = le.fit_transform(data_train['Family_Size'])
Family_Size_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Work_Experience'] = le.fit_transform(data_train['Work_Experience'])
Work_Experience_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Age'] = le.fit_transform(data_train['Age'])
Age_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Segmentation'] = le.fit_transform(data_train['Segmentation'])
Segmentation_mapping = {l: i for i, l in enumerate(le.classes_)}


# In[50]:


print(Gender_mapping, Ever_Married_mapping, Graduated_mapping, Spending_Score_mapping, Profession_mapping, Family_Size_mapping, Work_Experience_mapping, Var_1_mapping, Age_mapping, Segmentation_mapping)


# # Класификатор ближайших соседей

# In[51]:


from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[52]:


dg = data_train.copy(deep=True)
data_train = data_train.drop('Segmentation', axis=1)


# In[53]:


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_train, dg['Segmentation'], test_size=0.4, random_state=42)
xtrain_samples = x_train.shape[0]
xtest_samples = x_test.shape[0]
print(xtrain_samples, xtest_samples)


# In[54]:


neighbours = [1,2,3,4,5,7,10,12,15,18,20]
errs_train = []
errs_test = []

for i in neighbours:    
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    y_train_predict = model.predict(x_train)
    y_test_predict  = model.predict(x_test)
    errs_train.append(np.mean(y_train != y_train_predict))
    errs_test.append(np.mean(y_test != y_test_predict))
    
mat = pd.DataFrame([errs_train, errs_test], columns = neighbours,index=["errs_train", "errs_test"])
mat   


# In[55]:


model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test)

print(np.mean(y_train != y_train_predict), np.mean(y_test != y_test_predict))
print(f'Accuracy Score train : {accuracy_score(y_train, y_train_predict)*100}')
print(f'Accuracy Score test : {accuracy_score(y_test, y_test_predict)*100}')
print("Confusion matrix")
cm_knn = confusion_matrix(y_test, y_test_predict)
print(cm_knn)


# In[56]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test)

print(np.mean(y_train != y_train_predict), np.mean(y_test != y_test_predict))
print(f'Accuracy Score train : {accuracy_score(y_train, y_train_predict)*100}')
print(f'Accuracy Score test : {accuracy_score(y_test, y_test_predict)*100}')
print("Confusion matrix")
cm_knn = confusion_matrix(y_test, y_test_predict)
print(cm_knn)


# In[57]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test)

print(np.mean(y_train != y_train_predict), np.mean(y_test != y_test_predict))
print(f'Accuracy Score train : {accuracy_score(y_train, y_train_predict)*100}')
print(f'Accuracy Score test : {accuracy_score(y_test, y_test_predict)*100}')
print("Confusion matrix")
cm_knn = confusion_matrix(y_test, y_test_predict)
print(cm_knn)


# При предсказывании группы потенциального клиента методом ближайших соседей при k = 1 получили точность классификации около 97% на обучающей выборке и 40% на тестовой. При увеличении числа соседей наблюдается уменьшение точности на обучающей выборке и несущественный рост точности на тестовой.

# # Логистическая регрессия

# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[59]:


model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test)

print(np.mean(y_train != y_train_predict), np.mean(y_test != y_test_predict))
print(f'Accuracy Score train : {accuracy_score(y_train, y_train_predict)*100}')
print(f'Accuracy Score test : {accuracy_score(y_test, y_test_predict)*100}')
print("Confusion matrix")
cm = confusion_matrix(y_test, y_test_predict)
print(cm)


# # CNN

# In[60]:


from sklearn.neural_network import  MLPClassifier 
from sklearn.metrics import accuracy_score


# In[61]:


alpha_arr =np.logspace(-3, 5, 21)
test_err = []
train_err = []
train_acc = []
test_acc = []

for alpha in alpha_arr:
    mlp_model = MLPClassifier(hidden_layer_sizes = (8,2), 
                              solver = 'lbfgs', activation = 'logistic', alpha = alpha, max_iter=1000, random_state = 42)
    mlp_model.fit(x_train, y_train)

    y_train_pred = mlp_model.predict(x_train)
    y_test_pred = mlp_model.predict(x_test)
    
    train_err.append(np.mean(y_train != y_train_pred))
    test_err.append(np.mean(y_test != y_test_pred))
    train_acc.append(accuracy_score(y_train, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))
plt.semilogx(alpha_arr, train_err, '-o', label = 'train')
plt.semilogx(alpha_arr, test_err, '-o', label = 'test')
plt.xlim([np.min(alpha_arr), np.max(alpha_arr)])
plt.title('Error (alpha)')
plt.xlabel('alpha')
plt.ylabel('error')
plt.legend()


# In[62]:


min_train_err = np.min(train_err)
min_test_err = np.min(test_err)
print(min_train_err, min_test_err)


# # RandomForest

# In[63]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics


# In[64]:


model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test)

print(np.mean(y_train != y_train_predict), np.mean(y_test != y_test_predict))
print(f'Accuracy Score train : {accuracy_score(y_train, y_train_predict)*100}')
print(f'Accuracy Score test : {accuracy_score(y_test, y_test_predict)*100}')
print("Confusion matrix")
cm = confusion_matrix(y_test, y_test_predict)
print(cm)


# Вывод:
# На тестовой выборке логистическая регрессия, RandomForest и CNN показывают примерно равные значения accuracy. На обучающей выборке лучше показал себя RandomForest

# # Оптимальные значения гиперпараметров для RandomForest

# In[65]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Количество деревьев решений в лесу

# In[66]:


rf = RandomForestClassifier()
n_estimators = [i for i in range(1, 200, 5)]
accuracy = []
accuracy_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_est in n_estimators:
    model = RandomForestClassifier(n_estimators=n_est)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        model.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = model.predict(data_train.iloc[test_index])
        preds_train = model.predict(data_train.iloc[train_index])
        acc.append(accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracy.append(np.mean(acc))
    accuracy_train.append(np.mean(acc_train))


# In[67]:


plt.plot(n_estimators, accuracy, '-o', label='test')
plt.plot(n_estimators, accuracy_train, '-o', label='train')
plt.legend()
plt.show()


# Максимальное количество признаков, которое модели разрешается опробовать при каждом разбиении

# In[68]:


rf = RandomForestClassifier()
max_features = ["auto", "sqrt", "log2"]   
accuracy = []
accuracy_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in max_features:
    model = RandomForestClassifier(max_features=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        model.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = model.predict(data_train.iloc[test_index])
        preds_train = model.predict(data_train.iloc[train_index])
        acc.append(accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracy.append(np.mean(acc))
    accuracy_train.append(np.mean(acc_train))    


# In[69]:


plt.plot(max_features, accuracy, '-o', label='test')
plt.plot(max_features, accuracy_train, '-o', label='train')
plt.legend()
plt.show()


# Максимальная глубина каждого дерева в модели

# In[70]:


rf = RandomForestClassifier()
max_depth = list([i for i in range(1, 20)])
accuracy = []
accuracy_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in max_depth:
    model = RandomForestClassifier(max_depth=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        model.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = model.predict(data_train.iloc[test_index])
        preds_train = model.predict(data_train.iloc[train_index])
        acc.append(accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracy.append(np.mean(acc))
    accuracy_train.append(np.mean(acc_train)) 


# In[71]:


plt.plot(max_depth, accuracy, '-o', label='test')
plt.plot(max_depth, accuracy_train, '-o', label='train')
plt.legend()
plt.show()


# Минимальное количество образцов, необходимое для разбиения внутреннего узла каждого дерева

# In[72]:


rf = RandomForestClassifier()
min_samples_split = [2, 5, 10]
accuracy = []
accuracy_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in min_samples_split:
    clf = RandomForestClassifier(min_samples_split=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        model.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = model.predict(data_train.iloc[test_index])
        preds_train = model.predict(data_train.iloc[train_index])
        acc.append(accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracy.append(np.mean(acc))
    accuracy_train.append(np.mean(acc_train))


# In[73]:


plt.plot(min_samples_split, accuracy, '-o', label='test')
plt.plot(min_samples_split, accuracy_train, '-o', label='train')
plt.legend()
plt.show()


# Минимальное количество образцов, необходимое для нахождения в листовом узле каждого дерева

# In[74]:


rf = RandomForestClassifier()
min_samples_leaf = [i for i in range(1, 63, 5)]
accuracy = []
accuracy_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in min_samples_leaf:
    model = RandomForestClassifier(min_samples_leaf=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        model.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = model.predict(data_train.iloc[test_index])
        preds_train = model.predict(data_train.iloc[train_index])
        acc.append(accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracy.append(np.mean(acc))
    accuracy_train.append(np.mean(acc_train))


# In[75]:


plt.plot(min_samples_leaf, accuracy, '-o', label='test')
plt.plot(min_samples_leaf, accuracy_train, '-o', label='train')
plt.legend()
plt.show()


# In[76]:


pipeline = Pipeline([('algo',RandomForestClassifier(n_jobs=-1,random_state=42))])


# In[77]:


param_rf = {
    'algo__n_estimators':[200],
    'algo__max_depth':[6],
    'algo__max_features':[0.5],
    'algo__min_samples_leaf':[63],
    'algo__class_weight':[{0:0.34,
                           1:0.48,
                           2:0.39,
                           3:0.2}]
}


# In[78]:


cv_rs = GridSearchCV(pipeline,param_rf,cv=10,n_jobs=-1,verbose=1)
cv_rs.fit(x_train, y_train)
print(cv_rs.best_params_)


# In[79]:


gridcv_preds = cv_rs.predict(x_test)
gridcv_preds_train = cv_rs.predict(x_train)


# In[80]:


print(np.mean(y_train != y_train_predict), np.mean(y_test != y_test_predict))
print(f'Accuracy Score train : {accuracy_score(y_train, gridcv_preds_train)*100}')
print(f'Accuracy Score test : {accuracy_score(y_test, gridcv_preds)*100}')
print("Confusion matrix")
cm = confusion_matrix(y_test, gridcv_preds)
print(cm)


# Подобранные с помощью GridSearchCV оптимальные гиперпараметры близки к тем, что были видны на графиках. Качество модели улучшилось, но не значительно

# # Оптимальные значения гиперпараметров для CNN

# In[81]:


Alpha = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100]
N = [i for i in range(5, 50, 5)]

accuracy = pd.DataFrame(0., index = Alpha, columns = N)
accuracy_train = pd.DataFrame(0., index = Alpha, columns = N)

error_train = pd.DataFrame(0., index = Alpha, columns = N)
error = pd.DataFrame(0., index = Alpha, columns = N)

for n in N :
    for alpha in Alpha:
        model = MLPClassifier(hidden_layer_sizes = (n,2),solver = 'lbfgs', activation = 'logistic',random_state = 42,alpha = alpha)
        model.fit(x_train, y_train)
        y_train_predict = model.predict(x_train)
        y_test_predict  = model.predict(x_test)
        error_train[n][alpha] = np.mean(y_train != y_train_predict)
        error[n][alpha] = np.mean(y_test != y_test_predict)
        accuracy_train[n][alpha] = accuracy_score(y_train, y_train_predict)
        accuracy[n][alpha] = accuracy_score(y_test, y_test_predict)


# In[82]:


M1 = accuracy.values.max()
M2 = accuracy_train.values.max()
M3 = error.values.min()
M4 = error_train.values.min()

M = ((accuracy == M1) | (accuracy_train == M2) | (error == M3) | (error_train == M4))


# In[83]:


plt.figure(figsize=(7, 7))
sns.heatmap(accuracy, annot = round(accuracy[M], 4).fillna(''),
            fmt='', linewidths=2, linecolor='black',
            vmin = accuracy.quantile(0.9).min())
plt.title('accuracy', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# Наибольшего значения на тестовой выборке accuracy достигает при n = 20, alpha = 1.0 и n = 30, alpha = 0.5

# In[87]:


model = MLPClassifier(hidden_layer_sizes = (20,2),
                              solver = 'lbfgs', 
                              activation = 'logistic',
                              random_state = 42,
                              alpha = 1.0)

model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test) 
print(f'Train error     =  {np.mean(y_train != y_train_predict)}')
print(f'Test error      =  {np.mean(y_test != y_test_predict)}')
print(f'Train accuracy  =  {accuracy_score(y_train, y_train_predict)}')
print(f'Test accuracy   =  {accuracy_score(y_test, y_test_predict)}')
print("Confusion matrix")
cm = confusion_matrix(y_test, y_test_predict)
print(cm)


# In[88]:


model = MLPClassifier(hidden_layer_sizes = (30,2),
                              solver = 'lbfgs', 
                              activation = 'logistic',
                              random_state = 42,
                              alpha = 0.5)

model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test) 
print(f'Train error     =  {np.mean(y_train != y_train_predict)}')
print(f'Test error      =  {np.mean(y_test != y_test_predict)}')
print(f'Train accuracy  =  {accuracy_score(y_train, y_train_predict)}')
print(f'Test accuracy   =  {accuracy_score(y_test, y_test_predict)}')
print("Confusion matrix")
cm = confusion_matrix(y_test, y_test_predict)
print(cm)


# In[84]:


plt.figure(figsize=(7, 7))
sns.heatmap(accuracy_train, annot = round(accuracy_train[M], 4).fillna(''),
            fmt='', linewidths=2, linecolor='black',
            vmin = accuracy.quantile(0.9).min())
plt.title('train accuracy', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# Наибольшего значения на обучающей выборке accuracy достигает при n = 35, alpha = 1.0
