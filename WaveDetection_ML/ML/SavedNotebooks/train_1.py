from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Create dataframe 
def convert_to_list(s):
    return eval(s)
df = pd.read_csv('WaveDetection_ML/train_wave1.csv', converters={'upperRightShoulder': convert_to_list,'upperLeftShoulder': convert_to_list })
# view the dataframe
print(df['upperLeftShoulder'])
print(df['upperLeftShoulder'][1][0])

df = df.drop(df.columns[[0]],axis = 1)

# df.head()

# X is your independent variable list, y is your target variable (0 or 1)
X = df[['upperRightShoulder','upperLeftShoulder']]
y = df['TargetWave']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = np.asanyarray(X_train)
X_train = np.array(X_train)

# Train a logistic regression model
model = LogisticRegression()
# print(X_train.type())
print(X_train.shape)
print(X_train.dtype)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
