#Import packages and define functions

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from keras.preprocessing.image import load_img

#Add working directory where data is saved
os.chdir(r'C:\...')
import rmpavage.rmpa as rmpa

def LenPulses(i,RP):
  RP
  Temp = [i]
  AbsTot = 0
  while len(Temp)>0:
    AbsTot += np.square(RP.PG.nodes(data=True)[Temp[0]]['value'])
    Temp = list(nx.neighbors(RP.PG,Temp[0]))
  return np.sqrt(AbsTot)

def DPV(img):
  img = np.array(img)
  RP = rmpa.RoadmakersPavage(img,output=False)
  RP.create_feat_table()
  RP.dpt()
  DPV = np.arange(1,img.shape[0]*img.shape[1]+1,1)
  DPV = pd.Series(DPV).apply(lambda i: LenPulses(i,RP))
#  DPV = pd.Series(DPV).apply(lambda i: AbsPulses(i,RP))
  DPV = np.array(DPV).reshape(img.shape)
  return DPV
  
def DPVimg(DPV,img):
    Num1,Num2 = (np.quantile(DPV.flatten(),[0.5,1]))
    DPVimg = np.zeros(img.reshape(-1,1).shape)
    Idx = (DPV.reshape(-1,1)<=Num2)*(DPV.reshape(-1,1)>=Num1)
    DPVimg[Idx] = img.reshape(-1,1)[Idx]
    DPVimg = DPVimg.reshape(img.shape).astype(int)
    return DPVimg

#%%Load Files
Files = glob.glob(r'train\*.jpg')
DogFiles = [a for a in Files if 'dog' in a]
CatFiles = [a for a in Files if 'cat' in a]

nsamp = 1500
np.random.seed(5)
DogFile = np.random.choice(DogFiles,nsamp,False)
np.random.seed(10)
CatFile = np.random.choice(CatFiles,nsamp,False)

Dog = np.empty((nsamp,200,200))
Cat = np.empty((nsamp,200,200))
for i in range(nsamp):
  Dog[i,:,:] = np.array(load_img(DogFile[i], target_size=(200, 200)).convert('L'))
  Cat[i,:,:] = np.array(load_img(CatFile[i], target_size=(200, 200)).convert('L'))
Dog = Dog.astype(int)
Cat = Cat.astype(int)


#%% Apply DPT in parallel
from joblib import Parallel, delayed, dump
resDog = Parallel(n_jobs=-1)(delayed(DPV)(Dog[i,:,:]) for i in tqdm(range(nsamp)))
resDog = pd.Series(resDog).apply(lambda i: i.reshape(1,200,200))
DogDPV = np.concatenate(resDog,axis=0)
dump(Dog,'Dog.gz')
dump(DogDPV,'DogDPV.gz')
print('Dogs Done!')

resCat = Parallel(n_jobs=-1)(delayed(DPV)(Cat[i,:,:]) for i in tqdm(range(nsamp)))
resCat = pd.Series(resCat).apply(lambda i: i.reshape(1,200,200))
CatDPV = np.concatenate(resCat,axis=0)
dump(Cat,'Cat.gz')
dump(CatDPV,'CatDPV.gz')
print('Cats Done!')










