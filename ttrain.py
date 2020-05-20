import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import joblib


model = Sequential()
#实验中经过特征筛选后剩余的特征维度是24维
model.add(Dense(96, input_dim = 12, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = pd.read_csv('./feature/train.csv')
train_data = data



train_x = train_data.drop(['label'],axis=1)
print(np.array(train_x))
print(train_x.loc[0])

# train_x = train_x.drop(["BA_ppm"],axis=1)
# train_x = train_x.drop(["BA_skew"],axis=1)
# train_x = train_x.drop(["BA_var"],axis=1)
# train_x =train_x.drop(['RPM'],axis=1)
train_y = train_data['label']
tmp_x = np.array(train_x)
train_x = tmp_x



# data_test = pd.read_csv("./test/TEST02_feature.csv")
# test_x = np.array(data_test)

L = [[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0]]
train_y = list(train_y)
for i,x in enumerate(train_y):
    train_y[i] = L[x]

tmp_y = np.array(train_y)
train_y = tmp_y



model.fit(train_x,train_y,epochs=150,batch_size=100)
model.save("./model/nns.model")

# print(model.evaluate(test_x, test_y,batch_size=20))
# print(accuracy)

# print(train_x)
# print(train_y)