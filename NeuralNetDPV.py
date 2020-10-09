import os
import keras
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from joblib import load
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model 
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dropout,BatchNormalization
os.chdir(r'C:\Users\jeanp\Desktop\Universiteit\M\Research - JPs Folder\Python\Chapter 4 - Application')

#Obtained from stackoverflow
class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, filename='output/training_plot.jpg'):
        self.filename = filename

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename)
            plt.close()
            
PlotLosses = TrainingPlot('LossPlotNNDPV.jpg')
def DPVimg(DPV,img):
    Num1,Num2 = (np.quantile(DPV.flatten(),[0.5,1]))
    DPVimg = np.zeros(img.reshape(-1,1).shape)
    Idx = (DPV.reshape(-1,1)<=Num2)*(DPV.reshape(-1,1)>=Num1)
    DPVimg[Idx] = img.reshape(-1,1)[Idx]
    DPVimg = DPVimg.reshape(img.shape).astype(int)
    return DPVimg

#Loading data an pre-processing
XD1 = load('Dog.gz')
XC1 = load('Cat.gz')
XD2 = load('DogDPV.gz')
XC2 = load('CatDPV.gz')

#Image pre-processing
from joblib import Parallel, delayed
resDog = Parallel(n_jobs=-1)(delayed(DPVimg)(XD2[i,:,:],XD1[i,:,:]) for i in tqdm(range(XD1.shape[0])))
resDog = pd.Series(resDog).apply(lambda i: i.reshape(1,200,200))
XD2 = np.concatenate(resDog,axis=0)
resCat = Parallel(n_jobs=-1)(delayed(DPVimg)(XC2[i,:,:],XC1[i,:,:]) for i in tqdm(range(XC1.shape[0])))
resCat = pd.Series(resCat).apply(lambda i: i.reshape(1,200,200))
XC2 = np.concatenate(resCat,axis=0)

yD1 = np.ones((XD1.shape[0],1))
yD2 = np.ones((XD2.shape[0],1))
yC1 = np.zeros((XC1.shape[0],1))
yC2 = np.zeros((XC2.shape[0],1))

X_train11,X_test,y_train11,y_test = train_test_split(XD1,yD1,test_size=300,random_state=42)
X_val1,X_test1,y_val1,y_test1 = train_test_split(X_test,y_test,test_size=200,random_state=72)
X_train12,_,y_train12,_ = train_test_split(XD2,yD2,test_size=300,random_state=42)
X_train1 = np.concatenate((X_train11,X_train12))
y_train1 = np.concatenate((y_train11,y_train12))

X_train21,X_test,y_train21,y_test = train_test_split(XC1,yC1,test_size=300,random_state=41)
X_val2,X_test2,y_val2,y_test2 = train_test_split(X_test,y_test,test_size=200,random_state=71)
X_train22,_,y_train22,_ = train_test_split(XC2,yC2,test_size=300,random_state=41)
X_train2 = np.concatenate((X_train21,X_train22))
y_train2 = np.concatenate((y_train21,y_train22))

X_train = np.concatenate((X_train1,X_train2))
y_train = np.concatenate((y_train1,y_train2))
X_test = np.concatenate((X_test1,X_test2))
y_test = np.concatenate((y_test1,y_test2))
X_val = np.concatenate((X_val1,X_val2))
y_val = np.concatenate((y_val1,y_val2))

#NeuralNet 
img_rows,img_cols=200,200
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1) 
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1) 
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1) 
InputShape = (img_rows, img_cols, 1) 

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 
X_val = X_val.astype('float32') 
X_train /= 255
X_test /= 255
X_val /= 255

y_train = keras.utils.to_categorical(y_train) 
y_test = keras.utils.to_categorical(y_test) 
y_val = keras.utils.to_categorical(y_val) 

#Build and train NN
inpx = Input(shape=InputShape) 
layer1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(inpx) 
layer2 = BatchNormalization()(layer1) 
layer3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(layer2) 
layer4 = Conv2D(32, (3, 3), activation='relu')(layer3) 
layer5 = Conv2D(32, (3, 3), activation='relu')(layer4) 
layer6 = BatchNormalization()(layer5) 
layer7 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(layer6)
layer8 = BatchNormalization()(layer7) 
layer9 = Flatten()(layer8) 
layer10 = Dense(1024, activation='relu')(layer9) 
layer11 = Dropout(0.5)(layer10)
output = Dense(2, activation='softmax')(layer11) 

model = Model([inpx], output) 
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), 
              loss=keras.losses.binary_crossentropy, 
              metrics=['accuracy']) 
start = time()
model.fit(X_train, y_train, epochs=40,validation_data=(X_val,y_val),callbacks=[PlotLosses]) 
finish = time()

Returns = np.concatenate((PlotLosses.acc,PlotLosses.losses,PlotLosses.val_acc,PlotLosses.val_losses))
from joblib import dump
dump(Returns,'AccLossDPV.gz')

#Evaluating model
model.evaluate(X_test,y_test)
print('Time: {}'.format((finish-start)))
model.save('NN_DPV.h5')

#model = load_model('NN_DPV.h5')
yhat = model.predict(X_test).argmax(axis=1).reshape(-1,1)
ytest = y_test.argmax(axis=1).reshape(-1,1)
acc = (yhat==ytest).mean()
print('Accuracy(DPV): '+str(acc))



