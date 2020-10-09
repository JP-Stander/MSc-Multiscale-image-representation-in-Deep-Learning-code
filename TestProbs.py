import os
import numpy as np
from joblib import load
from keras.models import load_model
from sklearn.model_selection import train_test_split

#Add working directory where models and data is saved
os.chdir(r'C:\...')
#Load models
XD = load('Dog.gz')
XC = load('Cat.gz')
yD = np.ones((XD.shape[0],1))
yC = np.zeros((XC.shape[0],1))

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

OG = load_model('NN_OG.h5')
DPV = load_model('NN_DPV.h5')

yhOG = OG.predict(X_test)[:,1]
yhDPV = DPV.predict(X_test)[:,1]
yvOG = OG.predict(X_val)[:,1]
yvDPV = DPV.predict(X_val)[:,1]

accOG_opt = 0
accDPV_opt = 0
for i in np.arange(0,1,0.01):
    yhatOG = (yvOG>=i).reshape(-1,1)
    yhatDPV = (yvDPV>=i).reshape(-1,1)
    accOG = (yhatOG==y_val).mean()
    accDPV = (yhatDPV==y_val).mean()
    if accOG>accOG_opt:
        OG_opt = i
        accOG_opt = accOG
    if accDPV>accDPV_opt:
        DPV_opt = i
        accDPV_opt = accDPV
    
accOG = (y_test.reshape(-1,1)==(yhOG>=0.94).reshape(-1,1)).mean()
accDPV = (y_test.reshape(-1,1)==(yhDPV>=0.91).reshape(-1,1)).mean()

