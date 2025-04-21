import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

mnist=keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

x_train=x_train.reshape((x_train.shape[0],28*28))
x_test=x_test.reshape((x_test.shape[0],28*28))

y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

model=keras.Sequential([
    layers.Dense(128,activation='relu',input_shape=(28*28,)),  
    layers.Dense(64,activation='relu'),                           
    layers.Dense(10,activation='softmax')                          
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)

test_loss,test_acc=model.evaluate(x_test,y_test)
print(f'Test accuracy: {test_acc:.4f}')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()