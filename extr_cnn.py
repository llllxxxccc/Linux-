 #import the modules we need
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras import metrics
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.09#显存设置
set_session(tf.Session(config=config))

#define the Sequential model
class CNNNet:

    @staticmethod
    def createNet(input_shapes,nb_class):

        feature_layers = [
        BatchNormalization(input_shape=input_shapes),#规范层 
        Conv2D(64,3,3,border_mode="same"),#输入和输出格式相同，valid会缩小
        Activation("relu"),#激活防止过拟合
		
        BatchNormalization(),
        Conv2D(64,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
		
        BatchNormalization(),
        Conv2D(128,3 ,3,border_mode="same"),
        Activation("relu"),
		
        BatchNormalization(),
        Dropout(0.5),#随机丢掉百分之五十的神经元
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
		
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
		
        BatchNormalization()
        ]

        classification_layer=[
        Flatten(),
        Dense(512),
        Activation("relu"),
        Dropout(0.5),
        Dense(nb_class),
        Activation("softmax")
        ]

        model = Sequential(feature_layers+classification_layer)
        return model

#parameters
NB_EPOCH = 40
BATCH_SIZE = 128#每一次迭代的样本数
VERBOSE = 1#更新日志
VALIDATION_SPLIT = 0.2#验证集的一部分
IMG_ROWS=32
IMG_COLS = 32
NB_CLASSES = 10
INPUT_SHAPE =(IMG_ROWS,IMG_COLS,3)

#load cifar-10 dataset
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train.reshape(X_train.shape[0],IMG_ROWS,IMG_COLS,3)
X_test = X_test.reshape(X_test.shape[0],IMG_ROWS,IMG_COLS,3)

print(X_train.shape[0],"train samples")
print(Y_test.shape[0],"test samples")

#convert class vectors to binary class matrices
Y_train = to_categorical(Y_train,NB_CLASSES)
Y_test = to_categorical(Y_test,NB_CLASSES)

# init the optimizer and model
model = CNNNet.createNet(input_shapes=(32,32,3),nb_class=NB_CLASSES)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(X_train,Y_train,
                batch_size = BATCH_SIZE,
                nb_epoch = NB_EPOCH,
                verbose=VERBOSE,
                validation_split=VALIDATION_SPLIT,
                callbacks=[early_stopping]
                )

score = model.evaluate(X_test,Y_test,verbose = VERBOSE)
print("")
print("====================================")
print("====================================")
print(score[0])
print(score[1])
print("====================================")
print("====================================")

#save model
model.save('my_model.h5')

#show the data in history
print(history.history.keys())

#summarize history for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")

#summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.savefig("Performance:"+str(score[1])+".jpg")

