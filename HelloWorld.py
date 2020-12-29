import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Reshape, LSTM, Conv2D, MaxPooling2D, Dense, Flatten, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import load_model


# CIFAR-10 데이터 로드
from tensorflow.keras.datasets.cifar10 import load_data
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = train_images.reshape((50000,32,32,3))
test_images = test_images.reshape((10000,32,32,3))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 데이터 레이블 one hot 코드 변경
one_hot_train_labels = to_categorical(train_labels, 10)
one_hot_test_labels = to_categorical(test_labels, 10)


# #훈련에 사용할 옵티마이저(optimizer)와 손실 함수를 선택:
model = models.Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), activation='relu', padding="same"))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), activation='relu', padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), activation='relu', padding="same"))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
#
# #return 하도록 사용해 보기
model.add(Reshape(target_shape=(4*4, 128)))
model.add(LSTM(30, input_shape=(4*4, 128), return_sequences=True)) #unit 64로 변경
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# history = model.fit(train_images, one_hot_train_labels, epochs=10, batch_size=32)
history = model.fit(train_images, one_hot_train_labels, epochs=5, batch_size=128, validation_split=0.2)

# plt.figure(figsize=(12,4))#그래프의 가로세로 비율
# plt.subplot(1,1,1)#1행1열의 첫 번째 위치
# plt.plot(history.history['loss'],'b--',label='loss')#loss는 파란색 점선
# plt.plot(history.history['accuracy'],'g-',label='Accuracy')#accuracy는 녹색실선
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
# print('최적화 완료!')
#
model.save("/content/cnn_lstm_model.h5")
#
print("\n=============test results==========")
labels=model.predict(test_images)
print("\n Accuracy: %.4f" % (model.evaluate(test_images,one_hot_test_labels,verbose=2)[1]))#[0]: loss [1]:accuracy
# #
# #list index
# fig=plt.figure()
# for i in range(15):
#   subplot=fig.add_subplot(3,5,i+1)
#   subplot.set_xticks([])
#   subplot.set_yticks([])
#   subplot.set_title('%d' % np.argmax(labels[i]))
#   subplot.imshow(test_images[i].reshape((32,32,3)),cmap=plt.cm.gray_r)
# plt.show()
# #
print("===================================")