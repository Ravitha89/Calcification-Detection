from Model.Unet import *
from preprocess import *
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from keras.callbacks import ModelCheckpoint

Img_folder='E:/Projects/Calcification Detection/Dataset/Images'
Mask_folder='E:/Projects/Calcification Detection/Dataset/Masks'

#Preparing Input
#Read the Images and Masks from dataset
Img_list=read_Images(Img_folder)
Mask_list=read_Images(Mask_folder)
Final=zip(Img_list,Mask_list)
print('[.Loaded Images]')
######Debugger (OK)
#for(img,mask) in Final:
#    print(img)
#    print(mask)

#Read data from Imaage
#Loop over input images and masks into numpy arrays data and mask
start=time.time()
Image_List=[]
Mask_List=[]
for (img,mask) in Final:
    # Load the Image , Resize it 512,512 pixels
    image=os.path.join(Img_folder,img)
    Im=np.asarray(Image.open(image).resize((512,512)))
    Im=Im.astype(dtype='uint16')
    Image_List.append(Im)

    #Load the mask, Resize it 512,512 pixels
    Mask=os.path.join(Mask_folder,mask)
    mask=np.asarray(Image.open(Mask).resize((512,512)).convert(mode='1'))
    Im=mask.astype(dtype='uint16')
    Mask_List.append(Im)
end=time.time()
print('[..Converted to Numpy arrays]')    
print('[..It took',(end-start)/1000,'s]')
data=np.array(Image_List)
print(data.shape)
mask=np.array(Mask_List)
print(mask.shape)

######Debugger (OK)
#plt.imshow(data[0])
#plt.savefig("image.jpg")
#plt.imshow(mask[0])
#plt.savefig("Mask.jpg")

#Creating a model
model=unet()
print(model.summary())
print('[...Created Model]')

#Split the dataset into training and testing
(trainX,testX,trainY,testY)=train_test_split(data,mask,test_size=0.2)

print('[....Splitting Dataset]')
print('[....Training data size]',len(trainX))
print('[....Validation data size]',len(testX))

model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)

trainX=trainX.reshape(len(trainX),512,512,1)
trainY=trainY.reshape(len(trainY),512,512,1)
testX=testX.reshape(len(testX),512,512,1)
testY=testY.reshape(len(testY),512,512,1)
train_datagen=ImageDataGenerator(rescale=1./65536)
history=model.fit_generator(train_datagen.flow(trainX,trainY),
                    steps_per_epoch=len(trainX),
                    epochs=30,
                    validation_data=train_datagen.flow(testX,testY),
                    validation_steps=len(testX))
