# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:03:34 2020

@author: Satgu
"""
import keras 
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
from pickle import dump
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from keras.layers import  Dropout, Activation
from keras.layers import Convolution2D
from matplotlib import pyplot as plt
import joblib
import pandas as pd
import cv2
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
import os
import random
import seaborn as sns
import matplotlib.image as mpimg
from keras.optimizers import Adam       
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QTimer,QTime,QAbstractTableModel
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QTableView, QFileDialog,QMessageBox,QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from finalodevi import Ui_Form
from sklearn.model_selection import KFold 
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.datasets import make_classification
from xgboost import XGBClassifier # pip install xboost ön işlemini gerektirdi.
from sklearn.metrics import accuracy_score
from skimage import color,io
from skimage import transform
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PIL import Image, ImageOps
from skimage.transform import rescale, resize, downscale_local_mean
import skimage
import skimage.feature
from skimage.feature import (match_descriptors, corner_harris,corner_peaks, ORB, plot_matches)
from PyQt5.QtWidgets import QLabel, QVBoxLayout
import sys

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from PIL.ImageQt import ImageQt 
from skimage import data
import argparse
import shutil 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 

class MainWindow(QWidget,Ui_Form):
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.siftrgb=0
        self.siftcie=0
        self.sifthsv=0
        self.orbrgb=0
        self.orbcie=0
        self.orbhsv=0
        self.toplam=0
        self.rgbtoplam=0
        self.cietoplam=0
        self.hsvtoplam=0
        self.goster.clicked.connect(self.secilenresimgoster)
        self.hsvuygula.clicked.connect(self.hsvyecevir)
        self.cieuygula.clicked.connect(self.cieyecevir)
        self.rgbuygula.clicked.connect(self.rgbycevir)
        self.gosterhsv.clicked.connect(self.gosterhsvislemi)
        self.gostercie.clicked.connect(self.gostercieislemi)
        self.gosterrgb.clicked.connect(self.gosterrgbislemi)
        self.sift.clicked.connect(self.siftuygula)
        self.orb.clicked.connect(self.orbuygula)
        self.gerceklestir.clicked.connect(self.ayiralim)
        self.gostersift.clicked.connect(self.siftigosterelim)
        self.gosterorb.clicked.connect(self.orbyigosterelim)
        self.modelgiris.clicked.connect(self.goruntuayarla)
        self.gosterkfold.clicked.connect(self.kfoldlistele)
        self.modelegitim.clicked.connect(self.modelegitimiyap)
        self.modeltest.clicked.connect(self.modeltestet)
        self.tableView_5.clicked.connect(self.test)
        self.resimsecvgg.clicked.connect(self.vggyapalim)
        self.resimsecresnet.clicked.connect(self.resnetyapalim)
        self.resimsecinception.clicked.connect(self.inceptionyapalim)    
        self.verisetisec.clicked.connect(self.verisetinial)
    def test(self, item):
        self.row=item.row
    def secilenresimgoster(self):
        self.yeni='./yeniboyut/'
        item = self.listWidget.currentItem()
        
        if item is not None:
           resimismi=item.text()
           image = Image.open(self.path+resimismi)
           new_image = image.resize((190, 190))
           new_image.save(self.yeni+resimismi)
           self.secilenorijinal.setStyleSheet("background-image : url('"+self.yeni+resimismi+"')") 
    def verisetinial(self):
        self.path = QFileDialog().getExistingDirectory(None, 'Klasör seçiniz.')
        self.path=self.path+ '/'
        patients = os.listdir(self.path) #  listdir kullnarak   goruntuler listelendi
        for i in os.listdir(self.path):
            self.listWidget.insertItem(self.toplam, i)
            self.toplam = self.toplam+1
   
    
    def ciedoldur(self):
        pathcie='./ciefiles/'
        for i in os.listdir(pathcie):
              self.listWidgetcie.insertItem(self.cietoplam, i)
              self.cietoplam = self.cietoplam+1
        
    def hsvdoldur(self):
          pathhsv='./hsvfiles/'
          for i in os.listdir(pathhsv):
              self.listWidgethsv.insertItem(self.hsvtoplam, i)
              self.hsvtoplam = self.hsvtoplam+1
    def rgbdoldur(self):
        pathrgb='./rgbfiles/'
        for i in os.listdir(pathrgb):
              self.listWidgetrgb.insertItem(self.rgbtoplam, i)
              self.rgbtoplam = self.rgbtoplam+1




    def gosterhsvislemi(self):
         hsvkonum='./hsvfiles/'
         konum2='./yeniboyut/'
         item = self.listWidgethsv.currentItem()
         if item is not None:
             resimismi=item.text()
             image = Image.open(hsvkonum+resimismi)
             new_image = image.resize((110, 120))
             new_image.save(konum2+resimismi)
             self.hsvgoster.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
    def gostercieislemi(self):
        ciekonum='./ciefiles/'
        konum2='./yeniboyut/'
        item = self.listWidgetcie.currentItem()
        if item is not None:
             resimismi=item.text()
             image = Image.open(ciekonum+resimismi)
             new_image = image.resize((110, 120))
             new_image.save(konum2+resimismi)
             self.ciegoster.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
    def gosterrgbislemi(self):
        rgbkonum='./rgbfiles/'
        konum2='./yeniboyut/'
        item = self.listWidgetrgb.currentItem()
        if item is not None:
             resimismi=item.text()
             image = Image.open(rgbkonum+resimismi)
             new_image = image.resize((110, 120))
             new_image.save(konum2+resimismi)
             self.rgbgoster.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
    def siftigosterelim(self):
        itemrgb = self.listWidget_2.currentItem()
        itemcie = self.listWidget_3.currentItem()
        itemhsv = self.listWidget_4.currentItem()


        if(itemrgb!=None):
            self.label_2.setText("")
           
            rgbkonum='./siftrgb/'
            konum2='./yeniboyut/'
            
            if itemrgb is not None:
                 resimismi=itemrgb.text()
                 image = Image.open(rgbkonum+resimismi)
                 new_image = image.resize((220, 220))
                 new_image.save(konum2+resimismi)
                 self.label_2.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
                 self.listWidget_2.clearSelection()
                 self.listWidget_3.clearSelection()
                 self.listWidget_4.clearSelection()
        if(itemcie!=None ):
            self.label_2.setText("")

            ciekonum='./siftcie/'
            konum2='./yeniboyut/'
            
            if itemcie is not None:
                 resimismi=itemcie.text()
                 image = Image.open(ciekonum+resimismi)
                 new_image = image.resize((220, 220))
                 new_image.save(konum2+resimismi)
                 self.label_2.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
                 self.listWidget_2.clearSelection()
                 self.listWidget_3.clearSelection()
                 self.listWidget_4.clearSelection()
        if( itemhsv!=None):
            

            hsvkonum='./sifthsv/'
            konum2='./yeniboyut/'
            
            if itemhsv is not None:
                 resimismi=itemhsv.text()
                 image = Image.open(hsvkonum+resimismi)
                 new_image = image.resize((220, 220))
                 new_image.save(konum2+resimismi)
                 self.label_2.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
                 self.listWidget_2.clearSelection()
                 self.listWidget_3.clearSelection()
                 self.listWidget_4.clearSelection()
        else:
            
            self.listWidget_2.clearSelection()
            self.listWidget_3.clearSelection()
            self.listWidget_4.clearSelection()
    def orbyigosterelim(self):
        itemrgb = self.listWidget_5.currentItem()
        itemcie = self.listWidget_6.currentItem()
        itemhsv = self.listWidget_7.currentItem()


        if(itemrgb!=None):
            self.label_2.setText("")
            rgbkonum='./orbrgb/'
            konum2='./yeniboyut/'
            
            if itemrgb is not None:
                 resimismi=itemrgb.text()
                 image = Image.open(rgbkonum+resimismi)
                 new_image = image.resize((220, 220))
                 new_image.save(konum2+resimismi)
                 self.label_6.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
                 self.listWidget_5.clearSelection()
                 self.listWidget_6.clearSelection()
                 self.listWidget_7.clearSelection()
        if(itemcie!=None ):
            self.label_2.setText("")
            ciekonum='./orbcie/'
            konum2='./yeniboyut/'
            
            if itemcie is not None:
                 resimismi=itemcie.text()
                 image = Image.open(ciekonum+resimismi)
                 new_image = image.resize((220, 220))
                 new_image.save(konum2+resimismi)
                 self.label_6.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
                 self.listWidget_5.clearSelection()
                 self.listWidget_6.clearSelection()
                 self.listWidget_7.clearSelection()
        if( itemhsv!=None):
            self.label_2.setText("")
            hsvkonum='./orbhsv/'
            konum2='./yeniboyut/'
            
            if itemhsv is not None:
                 resimismi=itemhsv.text()
                 image = Image.open(hsvkonum+resimismi)
                 new_image = image.resize((220, 220))
                 new_image.save(konum2+resimismi)
                 self.label_6.setStyleSheet("background-image : url('"+konum2+resimismi+"')") 
                 self.listWidget_5.clearSelection()
                 self.listWidget_6.clearSelection()
                 self.listWidget_7.clearSelection()
        else:
            
            self.listWidget_5.clearSelection()
            self.listWidget_6.clearSelection()
            self.listWidget_7.clearSelection()
                 
    def hsvyecevir(self):
        self.bilgilendirme.setText("")
        new = './hsvfiles/' 
        if not os.path.exists(new):
            os.makedirs(new)
        hsvliste=[]
        for i in os.listdir(self.path):
            imag = mpimg.imread(self.path+i)
            hsv = cv2.cvtColor(imag, cv2.COLOR_RGB2HSV)
            hsvliste.append(hsv)
            cv2.imwrite(new+i,hsv)
        self.bilgilendirme.setText('HSV dönüştürme işlemi gerçekleştirildi.')
        self.hsvdoldur()
    def cieyecevir(self):
        self.bilgilendirme.setText("")
        new = './ciefiles/' 
        
        if not os.path.exists(new):
            os.makedirs(new)
        cieliste=[]
        for i in os.listdir(self.path):
            imag = mpimg.imread(self.path+i)
            cie = cv2.cvtColor(imag, cv2.COLOR_RGB2Lab)
            cieliste.append(cie)
            cv2.imwrite(new+i,cie)
        self.bilgilendirme.setText('CIE dönüştürme işlemi gerçekleştirildi.')    
        self.ciedoldur()
    def rgbycevir(self):
        self.bilgilendirme.setText("")
        konum=self.path
        kaynak='./rgbfiles/'
        for i in os.listdir(konum):
          
            im = Image.open(konum+i)
            if( im.mode !='RGB'):
                        image2=mpimg.imread(konum+i)
                        rgb= cv2.cvtColor(image2, cv2.COLOR_RGBA2RGB)
                        cv2.imwrite(kaynak+i, rgb)
            elif im.mode == 'GRAY':
                  image2=mpimg.imread(konum+i)
                  rgb= cv2.cvtColor(image2, cv2.GRAY2RGB)
                  cv2.imwrite(kaynak+i, rgb)
            elif im.mode == 'RGB':
                  image2=mpimg.imread(konum+i)
                  
                  cv2.imwrite(kaynak+i, image2)
        self.bilgilendirme.setText('RGB dönüştürme işlemi gerçekleştirildi.')
        self.rgbdoldur()
      
       
      
    def siftuygulacie(self):
        self.path='./ciefiles/'
        new = './siftcie/' 
        self.listesiftcie=[]
        self.fotoadisiftcie=[]
        if not os.path.exists(new):
            os.makedirs(new)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
        for i in os.listdir(self.path):
            self.fotoadisiftcie.append(i)
            imag = mpimg.imread(self.path+i)
            keypoints_sift, descriptors = sift.detectAndCompute(imag, None)
            self.pts=[p.pt for p in keypoints_sift]
            self.listesiftcie.append(self.pts)
            img = cv2.drawKeypoints(imag, keypoints_sift, None)
            cv2.imwrite(new+i,img)
        
        self.siftciedoldur()
        self.siftuygulahsv()
    def siftciedoldur(self):
        pathrgb='./siftcie/'
        for i in os.listdir(pathrgb):
              self.listWidget_3.insertItem(self.siftcie, i)
              self.siftcie = self.siftcie+1
        
    def siftuygula(self):
        self.path='./rgbfiles/'
        
        self.listesiftrgb=[]
        self.fotoadisiftrgb=[]
        new = './siftrgb/' 
        if not os.path.exists(new):
            os.makedirs(new)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
        for i in os.listdir(self.path):
            self.fotoadisiftrgb.append(i)
            imag = mpimg.imread(self.path+i)
            keypoints_sift, descriptors = sift.detectAndCompute(imag, None)
            self.pts=[p.pt for p in keypoints_sift]
            self.listesiftrgb.append(self.pts)
            
            img = cv2.drawKeypoints(imag, keypoints_sift, None)
            cv2.imwrite(new+i,img)
        
        self.siftrgbdoldur()
        self.siftuygulacie()
    def siftrgbdoldur(self):
        pathrgb='./siftrgb/'
        for i in os.listdir(pathrgb):
              self.listWidget_2.insertItem(self.siftrgb, i)
              self.siftrgb = self.siftrgb+1

    def siftuygulahsv(self):
        self.path='./hsvfiles/'
        new = './sifthsv/'
        self.listesifthsv=[]
        self.fotoadisifthsv=[]
        if not os.path.exists(new):
            os.makedirs(new)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
        for i in os.listdir(self.path):
            self.fotoadisifthsv.append(i)
            imag = mpimg.imread(self.path+i)
            keypoints_sift, descriptors = sift.detectAndCompute(imag, None)
            self.pts=[p.pt for p in keypoints_sift]
            self.listesifthsv.append(self.pts)
            img = cv2.drawKeypoints(imag, keypoints_sift, None)
            cv2.imwrite(new+i,img)
        
        self.sifthsvdoldur()
    def sifthsvdoldur(self):
        pathhsv='./sifthsv/'
        for i in os.listdir(pathhsv):
              self.listWidget_4.insertItem(self.sifthsv, i)
              self.sifthsv = self.sifthsv+1
        QMessageBox.question(self, 'Bilgilendirme', "Sift algoritması başarıyla gerçekleşti.",QMessageBox.Yes)   
    def orbuygula(self):
        self.path='./rgbfiles/'
        new='./orbrgb/'
        self.listesiorbrgb=[]
        self.fotoadiorbrgb=[]
        if not os.path.exists(new):
            os.makedirs(new)
        for i in os.listdir(self.path):
            self.fotoadiorbrgb.append(i)
            imag = mpimg.imread(self.path+i)
            orb= cv2.ORB_create(nfeatures=20)
            keypoint_orb, descriptors = orb.detectAndCompute(imag, None)
            img=cv2.drawKeypoints(imag, keypoint_orb, None)
            self.ptsorb=[p.pt for p in keypoint_orb]
            self.listesiorbrgb.append(self.ptsorb)
            cv2.imwrite(new+i,img)
        self.orbrgbdoldur()
        self.orbuygulahsv()
    def orbrgbdoldur(self):
        pathrgb='./orbrgb/'
        for i in os.listdir(pathrgb):
              self.listWidget_5.insertItem(self.orbrgb, i)
              self.orbrgb = self.orbrgb+1
    def orbuygulahsv(self):
        self.path='./hsvfiles/'
        new='./orbhsv/'
        self.listesiorbhsv=[]
        self.fotoadiorbhsv=[]
        if not os.path.exists(new):
            os.makedirs(new)
        for i in os.listdir(self.path):
            self.fotoadiorbhsv.append(i)
            imag = mpimg.imread(self.path+i)
            orb= cv2.ORB_create(nfeatures=20)
            keypoint_orb, descriptors = orb.detectAndCompute(imag, None)
            img=cv2.drawKeypoints(imag, keypoint_orb, None)
            self.ptsorb=[p.pt for p in keypoint_orb]
            self.listesiorbhsv.append(self.ptsorb)
            cv2.imwrite(new+i,img)
        self.orbhsvdoldur()
        self.orbuygulacie()
    def orbhsvdoldur(self):
        pathsv='./orbhsv/'
        for i in os.listdir(pathsv):
              self.listWidget_7.insertItem(self.orbhsv, i)
              self.orbhsv = self.orbhsv+1
       
    def orbuygulacie(self):
        self.path='./ciefiles/'
        new='./orbcie/'
        self.listesiorbcie=[]
        self.fotoadiorbcie=[]
        if not os.path.exists(new):
            os.makedirs(new)
        for i in os.listdir(self.path):
            self.fotoadiorbcie.append(i)
            imag = mpimg.imread(self.path+i)
            orb= cv2.ORB_create(nfeatures=20)
            keypoint_orb, descriptors = orb.detectAndCompute(imag, None)
            img=cv2.drawKeypoints(imag, keypoint_orb, None)
            self.ptsorb=[p.pt for p in keypoint_orb]
            self.listesiorbcie.append(self.ptsorb)
            cv2.imwrite(new+i,img)
       
        self.orbciedoldur()
    def orbciedoldur(self):
        pathrgb='./orbcie/'
        for i in os.listdir(pathrgb):
              self.listWidget_6.insertItem(self.orbcie, i)
              self.orbcie = self.orbcie+1
        QMessageBox.question(self, 'Bilgilendirme', "ORB algoritması başarıyla gerçekleşti.",QMessageBox.Yes)

    def goruntuayarla(self):
        self.secilenlerinismi=[]
        nbilgisi=self.lineEdit.text()
        nbilgisi=int(nbilgisi)
        self.descsliste=[]
        self.targetliste=[]
        sayac=0
        secilen=""
        if self.radioButton_13.isChecked()==True:
            secilen='sift'
            if self.radioButton_4.isChecked()==True:
                secilen= secilen+'rgb'
            if self.radioButton_5.isChecked()==True:
                secilen= secilen+ 'hsv'
            if self.radioButton_6.isChecked()==True:
                secilen=secilen=secilen+'cie'
        if self.radioButton_14.isChecked()==True:
            secilen='orb'
            if self.radioButton_4.isChecked()==True:
                secilen= secilen+'rgb'
            if self.radioButton_5.isChecked()==True:
                secilen= secilen+ 'hsv'
            if self.radioButton_6.isChecked()==True:
                secilen=secilen=secilen+'cie'
        print(secilen)
        secilen='./'+secilen+'/'
        if secilen=='./siftrgb/':
                self.fotoadi=self.fotoadisiftrgb
                self.liste=self.listesiftrgb
        if secilen=='./siftcie/':
            self.fotoadi=self.fotoadisiftcie
            self.liste=self.listesiftcie
        if secilen=='./sifthsv/':
            self.fotoadi=self.fotoadisifthsv
            self.liste=self.listesiftcie
        if secilen=='./orbrgb/':
                self.fotoadi=self.fotoadiorbrgb
                self.liste=self.listesiorbrgb
        if secilen=='./orbcie/':
            self.fotoadi=self.fotoadiorbcie
            self.liste=self.listesiorbcie
        if secilen=='./orbsv/':
            self.fotoadi=self.fotoadiorbhsv
            self.liste=self.listesiorbhsv
              
        new='./modelgirisgoruntuleri/'
        newno='./modelgirisgoruntuleri/no/'
        newyes='./modelgirisgoruntuleri/yes/'
        if not os.path.exists(new):
            os.makedirs(new)
        if not os.path.exists(newno):
            os.makedirs(newno)
        if not os.path.exists(newyes):
            os.makedirs(newyes)
        
        for i in range(0,20):
           
            
            resim=self.fotoadi[i]
            resimoku= mpimg.imread(secilen+resim)
            degisken= (resimoku.shape)
                        
            x=degisken[0]
            y=degisken[1]
            keypointnoktalari=self.liste[i]
            keypointsayisi=len(keypointnoktalari)
            for j in range(0,keypointsayisi):
                
                point=keypointnoktalari[j]
                pointx=point[0]
                pointx= int(pointx)
                pointy=point[1]
                pointy=int(pointy)
              
                if(pointx+nbilgisi>x or pointy+nbilgisi>y):
                    continue
                else:
                    
                    if resim[:2]=='no':
                        yeniresim=str(sayac)+resim
                        crop_img=resimoku[pointx:pointx+nbilgisi, pointy:+pointy+nbilgisi]
                        cv2.imwrite(newno+yeniresim, crop_img)
                        img= mpimg.imread(newno+yeniresim)
                        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = skimage.feature.daisy(img2, step=108, radius=12, rings=2, histograms=6, orientations=8, visualize=True)
                        descs= descs.reshape(descs.shape[0], descs.shape[1]*descs.shape[2])
                        descs=resize(descs, (28,28))
                        descs=descs.flatten()
                        self.descsliste.append(descs)
                        self.targetliste.append(0)
                        self.secilenlerinismi.append(yeniresim)
                        
                    if(resim[:3])=='yes':
                        yeniresim=str(sayac)+resim
                        crop_img=resimoku[pointx:pointx+nbilgisi, pointy:+pointy+nbilgisi]
                        cv2.imwrite(newyes+yeniresim, crop_img)
                        img= mpimg.imread(newyes+yeniresim)
                        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        descs, descs_img = skimage.feature.daisy(img2, step=108, radius=12, rings=2, histograms=6, orientations=8, visualize=True)
                        descs= descs.reshape(descs.shape[0], descs.shape[1]*descs.shape[2])
                        descs=resize(descs, (28,28))
                        descs=descs.flatten()
                        self.descsliste.append(descs)
                        self.targetliste.append(1)
                        self.secilenlerinismi.append(yeniresim)
               
    def ayiralim(self):
        import matplotlib.pyplot as plt
        from sklearn import datasets, svm, metrics
        from sklearn.model_selection import train_test_split
        self.comboBox_2.clear()
        self.kfoldlistesi=[]
        x=self.descsliste
        y=self.targetliste
        secim=self.comboBox.currentText()
        if secim=='Holdout':
            from sklearn.model_selection import train_test_split
            self.tableView.clearSpans()
            self.tableView_2.clearSpans()
            self.tableView_3.clearSpans()
            self.tableView_4.clearSpans()
            self.label_10.setText("")
            testsize=self.lineEdit_2.text()
            testsize=int(testsize)
            testsize=float(testsize/100)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=testsize, random_state=100)
            degisken=pd.DataFrame(self.x_train)
            islenmis=pandasModel(degisken)
            self.tableView.setModel(islenmis)
            degisken=pd.DataFrame(self.y_train)
            islenmis=pandasModel(degisken)
            self.tableView_4.setModel(islenmis)
            degisken=pd.DataFrame(self.x_test)
            islenmis=pandasModel(degisken)
            self.tableView_3.setModel(islenmis)
            degisken=pd.DataFrame(self.y_test)
            islenmis=pandasModel(degisken)
            self.tableView_2.setModel(islenmis)
            #verilerin olceklenmesi iicn 
            sc=StandardScaler(with_mean=False)
            x_train = sc.fit_transform(self.x_train)
            x_test = sc.fit_transform(self.x_test)
            satr,stn=x_train.shape
            #LogisticRegression
            logr=LogisticRegression(random_state=0)
            logr.fit(self.x_train,self.y_train)#egitim
            y_pred=logr.predict(self.x_test)#tahmin
            acc=accuracy_score(y_pred, self.y_test)
            class_names=""
            titles_options = [("Confusion matrix", None)]
            for title in titles_options:
                disp = plot_confusion_matrix(logr, self.x_test, self.y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     )
                disp.ax_.set_title(title)
    
               
            plt.savefig('./holdooutcm',dpi=300)
            plt.show();
            pred_prob1=logr.predict_proba(self.x_test)
            fpr1, tpr1, thres1= roc_curve(self.y_test, pred_prob1[:,1], pos_label=1)
            random_probs = [0 for i in range(len(self.y_test))]
            p_fpr, p_tpr, _ = roc_curve(self.y_test, random_probs, pos_label=1)
            auc_score1 = roc_auc_score(self.y_test, pred_prob1[:,1])
            plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='ROC')
            
            plt.title('ROC curve');
          
            plt.xlabel('False Positive Rate');
            
            plt.ylabel('True Positive rate');
            
            plt.legend(loc='best');
            plt.savefig('./ROChold',dpi=300);
            plt.show();
              
        if secim=='KFold':
            self.tableView.clearSpans()
            self.tableView_2.clearSpans()
            self.tableView_3.clearSpans()
            self.tableView_4.clearSpans()
            self.label_10.setText("")
            testsize=self.lineEdit_2.text()
            testsize=int(testsize)
            self.comboBox_2.clear()
            for i in range(0,testsize):
                
                self.comboBox_2.addItem(str(i+1))

            xlistesi=np.array(self.descsliste)
            ylistesi=np.array(self.targetliste)
    
            i=0
            cv = KFold(n_splits=testsize, random_state = 0, shuffle=True)
            for train_index, test_index in cv.split(xlistesi):
       
                self.x_train, self.x_test, self.y_train, self.y_test = xlistesi[train_index], xlistesi[test_index], ylistesi[train_index], ylistesi[test_index]
                self.kfoldlistesi.append(self.x_train)
                self.kfoldlistesi.append(self.x_test)
                self.kfoldlistesi.append(self.y_train)
                self.kfoldlistesi.append(self.y_test)
                
                sc=StandardScaler(with_mean=False)
                x_train = sc.fit_transform(self.x_train)
                x_test = sc.fit_transform(self.x_test)
                satr,stn=x_train.shape
                #LogisticRegression
                logr=LogisticRegression(random_state=0)
                logr.fit(self.x_train,self.y_train)#egitim
                y_pred=logr.predict(self.x_test)#tahmin
                acc=accuracy_score(y_pred,self.y_test)
                cm=confusion_matrix(self.y_test, y_pred)
                acc = round(acc, 2)

                class_names=""
                i=i+1
                titles_options = [("Confusion matrix", "kfold"+str(i)) ]
                for title in titles_options:
                    disp = plot_confusion_matrix(logr, self.x_test, self.y_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues,
                                         )
                    disp.ax_.set_title(title)
        
                
                plt.savefig('./kfoldcm'+str(i),dpi=300);
                plt.show();
                
                pred_prob1=logr.predict_proba(x_test)
                fpr1, tpr1, thres1= roc_curve(self.y_test, pred_prob1[:,1], pos_label=1)
                random_probs = [0 for i in range(len(self.y_test))]
                p_fpr, p_tpr, _ = roc_curve(self.y_test, random_probs, pos_label=1)
                auc_score1 = roc_auc_score(self.y_test, pred_prob1[:,1])
                plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
                plt.title('ROC curve');
                plt.xlabel('False Positive Rate');
                plt.ylabel('True Positive rate');
                plt.legend(loc='best');
                
                plt.savefig('./KfoldRoc',dpi=300);
                
                image = Image.open('./KfoldRoc.png')
                new_image = image.resize((310, 250))
                new_image.save('./KfoldRoc.png')
                
    
   
    def kfoldlistele(self):
        secim=self.comboBox_2.currentText()
        if secim =="1":
           degisken=self.kfoldlistesi[0]
           degisken1=self.kfoldlistesi[1]
           degisken2=self.kfoldlistesi[2]
           degisken3=self.kfoldlistesi[3]
            
                 
        if secim=="2":
           degisken=self.kfoldlistesi[4]
           degisken1=self.kfoldlistesi[5]
           degisken2=self.kfoldlistesi[6]
           degisken3=self.kfoldlistesi[7]
                 
        if secim=="3":
           degisken=self.kfoldlistesi[8]
           degisken1=self.kfoldlistesi[9]
           degisken2=self.kfoldlistesi[10]
           degisken3=self.kfoldlistesi[11]      
        if secim=="4":
           degisken=self.kfoldlistesi[12]
           degisken1=self.kfoldlistesi[13]
           degisken2=self.kfoldlistesi[14]
           degisken3=self.kfoldlistesi[15]
                 
        if secim=="5":
           degisken=self.kfoldlistesi[16]
           degisken1=self.kfoldlistesi[17]
           degisken2=self.kfoldlistesi[18]
           degisken3=self.kfoldlistesi[19]
                 
        if secim=="6":
           degisken=self.kfoldlistesi[20]
           degisken1=self.kfoldlistesi[21]
           degisken2=self.kfoldlistesi[22]
           degisken3=self.kfoldlistesi[23]
          
        degisken=pd.DataFrame(degisken)
        islenmis=pandasModel(degisken)
        self.tableView.setModel(islenmis)
       
        degisken2=pd.DataFrame(degisken2)
        islenmis=pandasModel(degisken2)
        self.tableView_4.setModel(islenmis)
   
        degisken1=pd.DataFrame(degisken1)
        islenmis=pandasModel(degisken1)
        self.tableView_3.setModel(islenmis)
  
        degisken3=pd.DataFrame(degisken3)
        islenmis=pandasModel(degisken3)
        self.tableView_2.setModel(islenmis)

    def modelegitimiyap(self):
        from  keras.utils import np_utils
        x_train=pd.DataFrame(self.x_train)
        x_test=pd.DataFrame(self.x_test)
        y_train=pd.DataFrame(self.y_train)
        y_test=pd.DataFrame(self.y_test )       
        degisken2=pd.DataFrame(x_test)
        islenmis=pandasModel(degisken2)
        self.tableView_5.setModel(islenmis)
        logr=LogisticRegression(random_state=0)
        logr.fit(x_train,y_train)
        y_pred=logr.predict(x_test)#tahmin
        cm=confusion_matrix(y_test, y_pred)
        print('modeleğitimcm'+str(cm))
        class_names=""
        titles_options = [("Confusion matrix, Model egitim Sonucu ", None)
                  ]
        for title in titles_options:
            disp = plot_confusion_matrix(logr, x_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 )
            disp.ax_.set_title(title)
        plt.show()

        pred_prob1=logr.predict_proba(x_test)
        fpr1, tpr1, thres1= roc_curve(y_test, pred_prob1[:,1], pos_label=1)
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
        auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
        

        X_train = x_train.values.reshape(x_train.shape[0], 28, 28,1)
        X_test = x_test.values.reshape(x_test.shape[0], 28, 28,1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
       
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
      
        joblib.dump(X_train,"./X_train.pkl") 
        joblib.dump(y_train,"./y_train.pkl") 
        joblib.dump(X_test,"./X_test.pkl") 
        joblib.dump(y_test,"./y_test.pkl") 

        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
           
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        batch_size2 = 16
        num_epoch = int(self.lineEdit_3.text())
        #model training
        model_log =model.fit(X_train, y_train, batch_size=batch_size2, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
        scores = model.evaluate(X_test, y_test, verbose=1)
        
        loss=round(scores[0],2)
        accur=round(scores[1],2)
        self.label_13.setText('Loss:'+str(loss)+'\n'+'Acc:'+str(accur))
      

        # plotting the metrics
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(model_log.history['acc'])
        plt.plot(model_log.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        
        plt.subplot(2,1,2)
        plt.plot(model_log.history['loss'])
        plt.plot(model_log.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.tight_layout()
        plt.savefig('./modelegitimsonuc',dpi=300)
        plt.show();
        image = Image.open('./modelegitimsonuc.png')
        new_image = image.resize((310, 250))
        new_image.save('./modelegitimsonuc.png')
        self.label_20.setStyleSheet("background-image : url('./modelegitimsonuc.png')")
        # serialize model to JSON
        model_json = model.to_json()
        with open("./model_mnist.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("./model_json_mnist.h5")
       
        plt.style.use('seaborn')
        # plot roc curves
        plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
        
       
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        
        plt.legend(loc='best')
        plt.savefig('./ROCmodelegitim',dpi=300)
        plt.show();
        image = Image.open('./ROCmodelegitim.png')
        new_image = image.resize((310, 250))
        new_image.save('./ROCmodelegitim.png')
        self.label_16.setStyleSheet("background-image : url('./ROCmodelegitim.png')")


    def modeltestet(self):
        import numpy as np
        from matplotlib import pyplot as plt
        np.random.seed(123)  # for reproducibility
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.models import model_from_json
        from keras.models import model_from_yaml
        from keras.layers import Convolution2D, MaxPooling2D	
        from  keras.utils import np_utils
        import joblib
        
        from keras.datasets import mnist
         
        X_train = joblib.load("./X_train.pkl") 
        y_train = joblib.load("./y_train.pkl") 
        X_test = joblib.load("./X_test.pkl") 
        y_test = joblib.load("./y_test.pkl") 
        
        # later...
        # load json and create model
        json_file = open('./model_mnist.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./model_json_mnist.h5")
        
  
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #score = loaded_model.evaluate(X_test, y_test, verbose=0)
        #print("%s: %.2f%% on Test" % (loaded_model.metrics_names[1], score[1]*100))
        scores = loaded_model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', scores[0]) #Test loss: 0.0296396646054
        print('Test accuracy:', scores[1])
        loss=round(scores[0],2)
        accur=round(scores[1],2)
        self.label_34.setText('Test Loss:'+str(loss)+'\n'+'Test Accuracy:'+str(accur))
   
        secim=self.comboBox.currentText()
       
            
        
         
            
        alinacakresim=0
        sayac=0
        aranan=self.row()
        aranan=int(aranan)
        
        gercek=self.y_test[aranan]
        
        self.label_17.setText("Secilen görüntü gerçek değer:"+str(gercek))
        secileninresmi= self.descsliste[aranan]
        

        #goruntulecenk=self.secilenlerinismi[sayac]

        for id,test_image in enumerate(X_test):
          
            if id!=aranan:
                continue   

            test_image = np.expand_dims(test_image, axis = 0)
            
            result = loaded_model.predict(test_image)
        
            
            test_image = test_image.reshape(1,28,28, 1)
            result = loaded_model.predict(test_image)
         

            etiket=None
            for values in result:
                best_value=0
                for i,value in enumerate(values):
                    
                    if (value>best_value):
                        best_value=value
                        etiket=i            
                        
                 
            
            break
            
        self.label_18.setText("Seçilen görüntü model tahmini:"+str(etiket))
        self.label_19.setText("")
        if etiket==gercek:
            
            self.label_19.setStyleSheet("color: darkgreen; ")
            self.label_19.setText("Model doğru tahminde bulundu.")
        if etiket!=gercek:
            self.label_19.setStyleSheet("color: darkred; ")
            self.label_19.setText("Model yanlış tahminde bulundu.")
        test_images=X_test
        test_images = test_images.reshape((X_test.shape[0], 28, 28))
        
        
        
        
       
       


    
    def donustur(self):
        # Data directories
        train_dir = "./modelicin/train/yes/"
        train_dir1 = "./modelicin/train/no/"
        test_dir = "./modelicin/test/no/"
        test_dir1 = "./modelicin/test/yes/"
        # Preparing dataframes
        images = os.listdir(train_dir)
        categories = []
      
        for image in images:
            image = Image.open(train_dir+image)
            new_image = image.resize((224, 224))
            new_image.save(train_dir+image)
        images = os.listdir(train_dir1)
        categories = []
      
        for image in images:     
            image = Image.open(train_dir1+image)
            new_image = image.resize((224, 224))
            new_image.save(train_dir+image)
        images = os.listdir(test_dir)
        categories = []
      
        for image in images:     
            image = Image.open(test_dir+image)
            new_image = image.resize((224, 224))
            new_image.save(test_dir+image)
        images = os.listdir(test_dir1)
        categories = []
      
        for image in images:     
            image = Image.open(test_dir1+image)
            new_image = image.resize((224, 224))
            new_image.save(test_dir1+image)

    def vggyapalim(self):
         
        self.dosyaadi, isim=QtWidgets.QFileDialog.getOpenFileName(None,"Lütfen tahmin edilecek görüntüyü seçin.","","Veri Seti Türü(*.jpg)")
        
        self.label_22.setText(self.dosyaadi[55:])
        image = Image.open(self.dosyaadi)
        new_image = image.resize((90, 90))
        new_image.save(self.dosyaadi)
        self.label_23.setStyleSheet("background-image : url('"+self.dosyaadi+"')")
        if(self.dosyaadi == None):
            QMessageBox.question(self, 'Bilgilendirme', "İşlenecek resim seçilmedi.",QMessageBox.Yes)
        else:
            from keras.preprocessing.image import ImageDataGenerator

            train_datagen = ImageDataGenerator(rescale = 1./255,
                                               shear_range = 0.2,
                                               zoom_range = 0.2,
                                               validation_split=0.2,
                                               horizontal_flip = True)
        
            test_datagen = ImageDataGenerator(rescale = 1./255)
            
            
            training_set = train_datagen.flow_from_directory(r'./modelicin/train/', target_size = (224, 224),class_mode = 'categorical')
            
            validation_set = train_datagen.flow_from_directory(r'./modelicin/train/', target_size = (224, 224),class_mode = 'categorical')
            
            test_set = test_datagen.flow_from_directory(r'./modelicin/test/', target_size = (224, 224), class_mode = 'categorical')
            
            
            

            IMAGE_SIZE = [224, 224]
            vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
            
            
            
            for layer in vgg.layers:
             layer.trainable = False
             
            x = Flatten()(vgg.output)
            prediction = Dense(2, activation='sigmoid')(x)
            model = Model(inputs=vgg.input, outputs=prediction)
            model.compile(loss='categorical_crossentropy',
                                optimizer=optimizers.Adam(),
                                metrics=['accuracy'])
            model.summary()    
            from datetime import datetime
            from keras.callbacks import ModelCheckpoint, LearningRateScheduler
            from keras.callbacks import ReduceLROnPlateau
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=5,
                                           min_lr=0.5e-6)
            checkpoint = ModelCheckpoint(filepath='mymodel.h5', 
                                           verbose=1, save_best_only=True)
            callbacks = [checkpoint, lr_reducer]
            start = datetime.now()
            history = model.fit_generator(training_set, 
                                steps_per_epoch=1, 
                                epochs = 3, verbose=1, 
                                validation_data = validation_set, 
                                validation_steps = 1)
            duration = datetime.now() - start
            score = model.evaluate(test_set)
            print('Test Loss:', score[0])
            print('Test accuracy:', score[1])
            testloss=round(score[0],2)
            testacc=round(score[1],2)
            self.label_21.setText('Test Loss:'+ str(testloss)+'\n'+'Test accuracy:'+ str(testacc))
            import matplotlib.pyplot as plt
      
            plt.plot(history.history["acc"])
            
            plt.plot(history.history["loss"])
            
            plt.title("model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Accuracy","loss"])
            
            
            plt.savefig('./vggsonuc',dpi=300)
            plt.show();
            image = Image.open('./vggsonuc.png')
            new_image = image.resize((310, 250))
            new_image.save('./vggsonuc.png')
            self.label_11.setStyleSheet("background-image : url('./vggsonuc.png')")
            image = Image.open(self.dosyaadi)
            new_image = image.resize((224, 224))
            new_image.save(self.dosyaadi)
            
            test_image = Image.open(self.dosyaadi)
            test_image = img_to_array(test_image)
          
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
           
            res = np.argmax(result)
            dict1 = {0 : 'Tumor yok', 1: 'Tumor var'}
        
            self.label_12.setStyleSheet('color:darkblue')
            self.label_12.setText("Tahmin edilen sonuc: "+str(dict1[res]))                
    
    
    
    
    def resnetyapalim(self):
        self.dosyaadi, isim=QtWidgets.QFileDialog.getOpenFileName(None,"Lütfen tahmin edilecek görüntüyü seçin.","","Veri Seti Türü(*.jpg)")
        
        self.label_27.setText(self.dosyaadi[55:])
        image = Image.open(self.dosyaadi)
        new_image = image.resize((90, 90))
        new_image.save(self.dosyaadi)
        self.label_28.setStyleSheet("background-image : url('"+self.dosyaadi+"')")
        if(self.dosyaadi == None):
            QMessageBox.question(self, 'Bilgilendirme', "İşlenecek resim seçilmedi.",QMessageBox.Yes)
        else:


            from tensorflow.python.keras.models import Sequential
            
            from keras.applications.resnet50 import ResNet50
            from tensorflow.python.keras.layers import Dense
            IMAGE_RESIZE = 224

           

            from keras.applications.resnet50 import preprocess_input
            from keras.preprocessing.image import ImageDataGenerator
            
              
    
            train_datagen = ImageDataGenerator(rescale = 1./255,
                                                   shear_range = 0.2,
                                                   zoom_range = 0.2,
                                                   validation_split=0.2,
                                                   horizontal_flip = True)
            
            test_datagen = ImageDataGenerator(rescale = 1./255)
                
                
            training_set = train_datagen.flow_from_directory(r'./modelicin/train/', target_size = (224, 224),class_mode = 'categorical')
                
            validation_set = train_datagen.flow_from_directory(r'./modelicin/train/', target_size = (224, 224),class_mode = 'categorical')
                
            test_set = test_datagen.flow_from_directory(r'./modelicin/test/', target_size = (224, 224), class_mode = 'categorical')
                
            
            model = Sequential()
            
            
            model.add(ResNet50(include_top = False, pooling = 'avg'))
            
            
            model.add(Dense(2, activation = 'softmax'))
            
            
            model.layers[0].trainable = False
            model.summary()
            
            sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
            model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
            from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    
            cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)
            cb_checkpointer = ModelCheckpoint(filepath = 'best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto'   )
            fit_history = model.fit_generator(
            training_set,
            steps_per_epoch=1,
            epochs = 2,
            validation_data=validation_set,
            validation_steps=1,
            callbacks=[cb_checkpointer, cb_early_stopper]
            )
            model.load_weights("./best.hdf5")#en iyi kaydedilen model
            scorea = model.evaluate(test_set)
            print('Test Loss:', scorea[0])
            print('Test accuracy:', scorea[1])
            testloss=round(scorea[0],2)
            testacc=round(scorea[1],2)
            self.label_26.setText('Test Loss:'+ str(testloss)+'\n'+'Test accuracy:'+ str(testacc))
            plt.plot(fit_history.history["acc"])
            
            plt.plot(fit_history.history["loss"])
          
            plt.title("model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Accuracy","loss"])
           
                
            plt.savefig('./resnetsonuc',dpi=300) 
            plt.show();
            image = Image.open('./resnetsonuc.png')
            new_image = image.resize((310, 250))
            new_image.save('./resnetsonuc.png')
            self.label_24.setStyleSheet("background-image : url('./resnetsonuc.png')")
            
            image = Image.open(self.dosyaadi)
            new_image = image.resize((224, 224))
            new_image.save(self.dosyaadi)
                
            test_image = Image.open(self.dosyaadi)
            test_image = img_to_array(test_image)
              
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
               
            res = np.argmax(result)
            dict1 = {0 : 'Tumor yok', 1: 'Tumor var'}
            print("The predicted output is :",dict1[res]) 
            self.label_25.setStyleSheet('color:darkblue')
            self.label_25.setText("Tahmin edilen sonuc: "+str(dict1[res]))  

       
    def inceptionyapalim(self):
        self.dosyaadi, isim=QtWidgets.QFileDialog.getOpenFileName(None,"Lütfen tahmin edilecek görüntüyü seçin.","","Veri Seti Türü(*.jpg)")
        
        self.label_32.setText(self.dosyaadi[55:])
        image = Image.open(self.dosyaadi)
        new_image = image.resize((90, 90))
        new_image.save(self.dosyaadi)
        self.label_33.setStyleSheet("background-image : url('"+self.dosyaadi+"')")
        if(self.dosyaadi == None):
            QMessageBox.question(self, 'Bilgilendirme', "İşlenecek resim seçilmedi.",QMessageBox.Yes)
        else:
           self.donustur
           from keras.preprocessing.image import ImageDataGenerator
    
           train_datagen = ImageDataGenerator(rescale = 1./255,
                                                   shear_range = 0.2,
                                                   zoom_range = 0.2,
                                                   validation_split=0.2,
                                                   horizontal_flip = True)
            
           test_datagen = ImageDataGenerator(rescale = 1./255)
                
                
           training_set = train_datagen.flow_from_directory(r'./modelicin/train/', target_size = (224, 224),class_mode = 'categorical')
                
           validation_set = train_datagen.flow_from_directory(r'./modelicin/train/', target_size = (224, 224),class_mode = 'categorical')
                
           test_set = test_datagen.flow_from_directory(r'./modelicin/test/', target_size = (224, 224), class_mode = 'categorical')
            
           from keras.models import Sequential
           from keras.models import Model
           from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
           from keras import optimizers, losses, activations, models
           from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
           from keras import applications
          
           
          #önceden eğitilmiş bi ağ varsa onu donduruyor
           base_model = applications.InceptionV3(weights='imagenet', 
                                            include_top=False, 
                                            input_shape=(224, 224,3))
           base_model.trainable = False
            
           add_model = Sequential()
           add_model.add(base_model)
           add_model.add(GlobalAveragePooling2D())
           add_model.add(Dropout(0.5))
           add_model.add(Dense(2, activation='sigmoid'))
        
           model = add_model
           model.compile(loss='categorical_crossentropy', 
                      optimizer=optimizers.SGD(lr=1e-4, 
                                               momentum=0.9),
                      metrics=['accuracy'])
           model.summary()
           file_path="./inceptionweights.best.hdf5"

           checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
            
           early = EarlyStopping(monitor="acc", mode="max", patience=3)
            
           callbacks_list = [checkpoint, early] #early
            
           history = model.fit_generator(training_set, 
                                          epochs=2, 
                                          shuffle=True, 
                                          verbose=True,
                                          callbacks=callbacks_list)     
          
           scorea = model.evaluate(test_set)
           print('Test Loss:', scorea[0])
           print('Test accuracy:', scorea[1])
           testloss=round(scorea[0],2)
           testacc=round(scorea[1],2)
           self.label_31.setText('Test Loss:'+ str(testloss)+'\n'+'Test accuracy:'+ str(testacc))
           plt.plot(history.history["acc"])
         
           plt.plot(history.history["loss"])
         
           plt.title("model accuracy")
           plt.ylabel("Accuracy")
           plt.xlabel("Epoch")
           plt.legend(["Accuracy","loss"])
           
                
           plt.savefig('./inceptionsonuc',dpi=300) 
           plt.show();
           image = Image.open('./inceptionsonuc.png')
           new_image = image.resize((310, 250))
           new_image.save('./inceptionsonuc.png')
           self.label_29.setStyleSheet("background-image : url('./inceptionsonuc.png')")
           
           image = Image.open(self.dosyaadi)
           new_image = image.resize((224, 224))
           new_image.save(self.dosyaadi)
            
           test_image = Image.open(self.dosyaadi)
           test_image = img_to_array(test_image)
          
           test_image = np.expand_dims(test_image, axis = 0)
           result = model.predict(test_image)
           
           res = np.argmax(result)
           dict1 = {0 : 'Tumor yok', 1: 'Tumor var'}
          
           self.label_30.setStyleSheet('color:darkblue')
           self.label_30.setText("Tahmin edilen sonuc: "+str(dict1[res]))  
        
        















class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data
    def rowCount(self, parent=None):
        return self._data.shape[0]
    def columnCount(self, parnet=None):
        return self._data.shape[1]
        
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
                
                
          
                        
          
        
    