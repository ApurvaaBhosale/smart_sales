import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data=pd.read_csv("E:\letter-recognition.data")

X,y=data.iloc[:,1:].values,data.iloc[:,0].values

y=LabelEncoder().fit_transform(y)
y=OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler().fit(X_train)
X_train,X_test=scaler.transform(X_train),scaler.transform(X_test)

model=Sequential([
    Dense(128,activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64,activation='relu'),
    Dense(26,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,batch_size=32,validation_split=0.2)

accuracy=model.evaluate(X_test,y_test)[1]
predictions=np.argmax(model.predict(X_test),axis=1)

print(f"Test Accuracy: {accuracy:}")
