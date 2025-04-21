import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df=pd.read_csv(r'D:\Pracs\dl\train_data.csv')  

df['sentiment']=df['sentiment'].map({0:'negative',1:'neutral',2:'positive'})

texts=df['sentence'].astype(str).values
labels=df['sentiment'].values

tokenizer=Tokenizer(num_words=10000,oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
padded=pad_sequences(sequences,maxlen=100,padding='post')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labels_encoded=le.fit_transform(labels)
labels_categorical=tf.keras.utils.to_categorical(labels_encoded,num_classes=3)

X_train,X_test,y_train,y_test=train_test_split(padded,labels_categorical,test_size=0.2,random_state=42)

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000,output_dim=128,input_length=100),
    tf.keras.layers.LSTM(64,return_sequences=False,dropout=0.3,recurrent_dropout=0.2),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history=model.fit(X_train,y_train,validation_split=0.1,epochs=5,batch_size=32)

loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred=model.predict(X_test)
y_pred_classes=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test,axis=1)

print("\nClassification Report:")
print(classification_report(y_true,y_pred_classes,target_names=le.classes_))

plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Val Accuracy')
plt.show()
