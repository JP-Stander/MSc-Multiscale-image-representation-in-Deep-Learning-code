import os
import keras
import numpy as np
from time import time
from joblib import load
from keras.models import Model 
from matplotlib import pyplot as plt
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dropout,BatchNormalization

#Add working directory where data is saved
os.chdir(r'C:\...')
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
            
PlotLosses = TrainingPlot('LossPlotNNOG.jpg')
#Load data
XD = load('Dog.gz')
XC = load('Cat.gz')
yD = np.ones((XD.shape[0],1))
yC = np.zeros((XC.shape[0],1))

#Image preprocessing
X_train1,X_test,y_train1,y_test = train_test_split(XD,yD,test_size=300,random_state=42)
X_val1,X_test1,y_val1,y_test1 = train_test_split(X_test,y_test,test_size=200,random_state=72)

X_train2,X_test,y_train2,y_test = train_test_split(XC,yC,test_size=300,random_state=41)
X_val2,X_test2,y_val2,y_test2 = train_test_split(X_test,y_test,test_size=200,random_state=71)

X_train = np.concatenate((X_train1,X_train2))
y_train = np.concatenate((y_train1,y_train2))
X_test = np.concatenate((X_test1,X_test2))
y_test = np.concatenate((y_test1,y_test2))
X_val = np.concatenate((X_val1,X_val2))
y_val = np.concatenate((y_val1,y_val2))

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
output = Dense(2, activation='sigmoid')(layer11) 

model = Model([inpx], output) 
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), 
              loss=keras.losses.binary_crossentropy, 
              metrics=['accuracy']) 
start = time()
model.fit(X_train, y_train, epochs=40,validation_data=(X_val,y_val),callbacks=[PlotLosses]) 
finish = time()

Returns = np.concatenate((PlotLosses.acc,PlotLosses.losses,PlotLosses.val_acc,PlotLosses.val_losses))
from joblib import dump
dump(Returns,'AccLossOG.gz')

#Evaluating model
model.evaluate(X_test,y_test)
print('Time: {}'.format((finish-start)))
model.save('NN_OG.h5')

yhat = model.predict(X_test).argmax(axis=1).reshape(-1,1)
ytest = y_test.argmax(axis=1).reshape(-1,1)
acc = (yhat==ytest).mean()
print('Accuracy(OG): '+str(acc))

