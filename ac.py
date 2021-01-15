import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
Creditcard=pd.read_csv('creditcard.csv')
Creditcard
Creditcard.columns
Creditcard.dtypes
Creditcard.head(5)
Creditcard.info()
Creditcard.isnull().sum()
Creditcard.describe()
Creditcard.Class.value_counts()
Creditcard['Class'].nunique()

print('Fraud Percentage: {}'.format(round((Creditcard['Class'].value_counts()[1] / len(Creditcard)) * 100, 2)))
print('Non Fraud Percentage: {}'.format(round((Creditcard['Class'].value_counts()[0] / len(Creditcard)) * 100, 2)))
Creditcard.drop("Time", axis=1, inplace=True)
Creditcard.shape

# visulization

sns.scatterplot(x='Amount', y='Class', data=Creditcard)
sns.boxplot(x="Class", y="Amount", data=Creditcard)
plt.ylim(0, 5000)
plt.show()
plt.close();

plt.close()
Creditcard
sns.scatterplot(x='V1', y='Class', data=Creditcard)
sns.scatterplot(x='V2', y='Amount', data=Creditcard)
sns.scatterplot(x='V3', y='Amount', data=Creditcard)
sns.scatterplot(x='V4', y='Amount', data=Creditcard)
sns.scatterplot(x='V5', y='Amount', data=Creditcard)
sns.scatterplot(x='V6', y='Amount', data=Creditcard)
sns.scatterplot(x='V7', y='Amount', data=Creditcard)
sns.scatterplot(x='V8', y='Amount', data=Creditcard)
sns.scatterplot(x='V9', y='Amount', data=Creditcard)
sns.scatterplot(x='V10', y='Amount', data=Creditcard)
sns.scatterplot(x='V11', y='Amount', data=Creditcard)
sns.scatterplot(x='V12', y='Amount', data=Creditcard)
sns.scatterplot(x='V13', y='Amount', data=Creditcard)
sns.scatterplot(x='V14', y='Amount', data=Creditcard)
sns.scatterplot(x='V15', y='Amount', data=Creditcard)
sns.scatterplot(x='V16', y='Amount', data=Creditcard)
sns.scatterplot(x='V17', y='Amount', data=Creditcard)
sns.scatterplot(x='V18', y='Amount', data=Creditcard)
sns.scatterplot(x='V19', y='Amount', data=Creditcard)
sns.scatterplot(x='V20', y='Amount', data=Creditcard)
sns.scatterplot(x='V21', y='Amount', data=Creditcard)
sns.scatterplot(x='V22', y='Amount', data=Creditcard)
sns.scatterplot(x='V23', y='Amount', data=Creditcard)
sns.scatterplot(x='V24', y='Amount', data=Creditcard)
sns.scatterplot(x='V25', y='Amount', data=Creditcard)
sns.scatterplot(x='V26', y='Amount', data=Creditcard)
sns.scatterplot(x='V27', y='Amount', data=Creditcard)
sns.scatterplot(x='V28', y='Amount', data=Creditcard)
sns.scatterplot(x='V27', y='Amount', data=Creditcard)
sns.scatterplot(x='V1', y='V2', data=Creditcard)
sns.scatterplot(x='V1', y='V3', data=Creditcard)
sns.scatterplot(x='V1', y='V4', data=Creditcard)
sns.scatterplot(x='V1', y='V5', data=Creditcard)
sns.scatterplot(x='V1', y='V6', data=Creditcard)
sns.scatterplot(x='V1', y='V7', data=Creditcard)
sns.scatterplot(x='V1', y='V8', data=Creditcard)
sns.scatterplot(x='V1', y='V9', data=Creditcard)
sns.scatterplot(x='V1', y='V10', data=Creditcard)
sns.scatterplot(x='V1', y='V11', data=Creditcard)
sns.scatterplot(x='V1', y='V12', data=Creditcard)
sns.scatterplot(x='V1', y='V13', data=Creditcard)
sns.scatterplot(x='V1', y='V14', data=Creditcard)
sns.scatterplot(x='V1', y='V15', data=Creditcard)
sns.scatterplot(x='V1', y='V16', data=Creditcard)
sns.scatterplot(x='V1', y='V17', data=Creditcard)
sns.scatterplot(x='V1', y='V18', data=Creditcard)
sns.scatterplot(x='V1', y='V19', data=Creditcard)
sns.scatterplot(x='V1', y='V20', data=Creditcard)
sns.scatterplot(x='V1', y='V21', data=Creditcard)
sns.scatterplot(x='V1', y='V22', data=Creditcard)
sns.scatterplot(x='V1', y='V23', data=Creditcard)
sns.scatterplot(x='V1', y='V24', data=Creditcard)
sns.scatterplot(x='V1', y='V25', data=Creditcard)
sns.scatterplot(x='V1', y='V26', data=Creditcard)
sns.scatterplot(x='V1', y='V27', data=Creditcard)
sns.scatterplot(x='V1', y='V28', data=Creditcard)

# model Building
# LogisticRegression
X = Creditcard[['Amount']]
y = Creditcard['Class']
sc = StandardScaler()
sc
X
X = sc.fit_transform(X)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
log_reg.coef_
log_reg.intercept_
y_pred = log_reg.predict(X_test)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

# smote
from imblearn.over_sampling import SMOTE

smote_oversample = SMOTE()
X, y = smote_oversample.fit_resample(X, y)
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))

# LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

svc_reg = LinearSVC(random_state=0)
svc_reg.fit(X_train, y_train)
y_pred = svc_reg.predict(X_test)
print(classification_report(y_test, y_pred))

# RandomForest
rf_reg = RandomForestClassifier(random_state=0)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
print(classification_report(y_test, y_pred))

# scaling data using standard scaller
X = Creditcard.drop(['Class'], axis=1)
y = Creditcard['Class']
sc = StandardScaler()
smote_oversample = SMOTE()
X, y = smote_oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))

X = sc.fit_transform(X)
svc_reg = LinearSVC(random_state=0)
svc_reg.fit(X_train, y_train)
y_pred = svc_reg.predict(X_test)
print(classification_report(y_test, y_pred))

rf_reg = RandomForestClassifier(random_state=0)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.preprocessing import StandardScaler
Creditcard['Amount(Normalized)'] = StandardScaler().fit_transform(Creditcard['Amount'].values.reshape(-1,1)

Creditcard.iloc[:,[29,31]].head()

Creditcard = Creditcard.drop(columns = ['Amount', 'Time'], axis=1) # This columns are not necessary anymore.

X = Creditcard.drop('Class', axis=1)
y = Creditcard['Class']

#train-test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# We are transforming data to numpy array to implementing with keras
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential([
    Dense(units=20, input_dim = X_train.shape[1], activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(units=20,activation='relu'),
    Dense(units=24,activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=30, epochs=5)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=30, epochs=5)

score = model.evaluate(X_test, y_test)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score[1]*100,score[0]))

#SMOTE

from imblearn.over_sampling import SMOTE
X_smote, y_smote = SMOTE().fit_sample(X, y)
X_smote = pd.DataFrame(X_smote)
y_smote = pd.DataFrame(y_smote)
y_smote.iloc[:,0].value_counts()

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 30, epochs = 5)

#Accurancy
score = model.evaluate(X_test, y_test)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score[1]*100,score[0]))


#Confusion matrix1
y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)
cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm, annot=True, fmt='.0f')
plt.show()
#Confusion matrix2
y_pred2 = model.predict(X)
y_test2 = pd.DataFrame(y)
cm2 = confusion_matrix(y_test2, y_pred2.round())
sns.heatmap(cm2, annot=True, fmt='.0f', cmap='coolwarm')
plt.show()

scoreNew = model.evaluate(X, y)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(scoreNew[1]*100,scoreNew[0]))

print(classification_report(y_test2, y_pred2.round()))

