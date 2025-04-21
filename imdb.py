import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

vocab_size=10000
max_length=200
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=vocab_size)

x_train=pad_sequences(x_train,maxlen=max_length,padding='post')
x_test=pad_sequences(x_test,maxlen=max_length,padding='post')

model=Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=5,batch_size=32,validation_split=0.2,verbose=1)

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.show()
