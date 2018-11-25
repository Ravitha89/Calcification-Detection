import os
import pydicom as dicom
import shutil

INPUT_FOLDER='E:/research/INbreast Release 1.0/AllImages'
CALC_FOLDER='E:/research/INbreast Release 1.0/CalcificationSegmentationMasks/calcification'
COPY_FOLDER='E:/Projects/Calcification Detection/Dataset/Images'
COPY_MASK='E:/Projects/Calcification Detection/Dataset/Masks'

def read_Images(folder):
    Img_list=os.listdir(folder)
    return Img_list

def map_pid_image(folder):
    patientID=[]  # List of PatientIDs
    imagePaths=[] # List of Images
    for root,DirList,fileList in os.walk(INPUT_FOLDER):
        for file in fileList:
            if(file.partition('.')[2]=="png"): # To consider only the DICOM Images
                imagePaths.append(file)
                patientID.append(file.partition('_')[0])
    #print(len(imagePaths))
    #print(imagePaths[0])
    return(imagePaths,patientID)


def calc_pid(folder):
    calc_pid_list=[] # List of patientid from CALCIFICATION masks
    calc_images=os.listdir(folder)
    for image in calc_images:
        calc_pid_list.append(image.partition('_')[0])
    return calc_pid_list


def copy_images(patient_pid,imagePaths,calc_pid_list):
    for pid in calc_pid_list:
        index=patient_pid.index(pid)
        img=imagePaths[index]
        input_path=os.path.join(INPUT_FOLDER,img)
        output_path=os.path.join(COPY_FOLDER,img)
        shutil.copy(input_path,output_path)
        input_mask_path=os.path.join(CALC_FOLDER,pid+'_mask.png')
        output_mask_path=os.path.join(COPY_MASK,pid+'_mask.png')
        shutil.copy(input_mask_path,output_mask_path)

if __name__=="__main__":
    (imagePaths,patientID)=map_pid_image(INPUT_FOLDER)
    ###Debugging Section (OK!)
    #print(len(imagePaths))
    #print(len(patientID))
    #print(imagePaths[0])
    #print(patientID[0])
    calc_pid_list=calc_pid(CALC_FOLDER)
    ###Debugging Section (OK!)
    #print(len(calc_pid_list))
    #print(calc_pid_list[0])
    ### Copied!!
    #copy_images(patientID,imagePaths,calc_pid_list)
