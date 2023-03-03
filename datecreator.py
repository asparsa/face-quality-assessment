import os
import cv2
from blur import bluringall, focusbulr, wrongfocus, cropface
from darkness import darker, brighter
#change ir for make different part of your dataset
local_root="datasmall5\\validation\\normal\\"
part='datasmall5\\validation\\'
def creat_data(fun,target):
    image_list=[]
    list_of_files=os.listdir(local_root)
    for file in list_of_files:
        if '.jpg' or '.png' in file:
            if '.txt' not in file: ## it's passing the .txt files as it is .png!!!!!
                image_list.append(file)
    for image in image_list:
        with open(local_root+image[:-4] + '_BB.txt','r') as f:
            material = f.readline()
        temp_img=cv2.imread(local_root+image)
        if fun== brighter or fun==darker:
            temp_img=fun(temp_img,50)

        elif fun==bluringall:
            temp_img=fun(temp_img)
        else:
            temp_img=fun(temp_img,material)
        path=target+'\\'+image[:-4]+'_'+fun.__name__+'.jpg'
        #print(path)
        cv2.imwrite(target+'\\'+image[:-4]+'_'+fun.__name__+'.jpg',temp_img)

#creat_data(wrongfocus,part+'wrong_focus')
print("creating wrong focus data is completed")
#creat_data(focusbulr,part+'focus_blur')
print("creating focus blur data is completed")
creat_data(darker,part+'dark')
print("creating dark data is completed")
creat_data(brighter,part+'bright')
print("creating bright data is completed")
#creat_data(bluringall,part+'all_blur')
print("creating bluring all data is completed")