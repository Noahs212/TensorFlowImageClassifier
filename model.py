import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1) data setup
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
#convert to one-hot
yTrainEncoded = to_categorical(yTrain)
yTestEncoded = to_categorical(yTest)
#unroll images into vectors
xTrainReshaped = np.reshape(xTrain, (60000, 784))
xTestReshaped = np.reshape(xTest, (10000, 784))
#normilzation
xMean = np.mean(xTrainReshaped)
xStd = np.std(xTrainReshaped)
epsilon = 1e-10
xTrainNorm = (xTrainReshaped - xMean) / (xStd + epsilon)
xTestNorm = (xTestReshaped - xMean) / (xStd + epsilon)


#2) model setup
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(
    optimizer = 'sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

#3) model training
model.fit(xTrainNorm, yTrainEncoded, epochs=3)


#4) evlauation 
loss, accuracy = model.evaluate(xTestNorm, yTestEncoded)
print('Test set accuracy:', accuracy * 100, '%')


#5) results
preds = model.predict(xTestNorm)
plt.figure(figsize=(12,12))
start_index = 0
for i in range(25):
    plt.subplot(5,5, i +1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index + i])
    gt = yTest[start_index+i]
    
    col = 'g'
    if pred != gt:
        col = 'r'
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color=col)
    plt.imshow(xTest[start_index+i], cmap='binary')
plt.show()