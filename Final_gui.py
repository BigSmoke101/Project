from tkinter import *
import tkinter.filedialog
from PIL import Image
from PIL import ImageTk
import time
import cv2
import math
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
LR = 1e-3
IMG_SIZE =100
disease=["basal cell carcinoma","dermatofibroma","melanoma","nevus","pigmented benign keratosis"]
im=''
tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 

convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.5) 

convnet = fully_connected(convnet, 5, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
	loss ='categorical_crossentropy', name ='targets') 

model = tflearn.DNN(convnet, tensorboard_dir ='log') 
model.load('skin.model')
str_label='new'
##



#-------------------------------------Gui part-----------------------
root = tkinter.Tk()
root.geometry("670x680")
root.title("Skin Disease Detection")
root.resizable(0,0)
#----------------------------CNN test -------------------------------------
def cNN_test(image):
    frame=cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newimg = cv2.resize(gray,(int(IMG_SIZE),int(IMG_SIZE)))
    data = newimg.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([ data])[0]
    model_out=list(model_out)
    val=max(model_out)
    idx=model_out.index(max(model_out))
    return idx
   # print(idx)
    #print(model_out)
   
    
# -----------------------------------------------------------------

def print_path():
    f = tkinter.filedialog.askopenfilename(
        parent=root, initialdir='',
        title='Choose file',
        filetypes=[('jpg image','.jpg')]
        )

    print(f)
    img = cv2.imread(f)
    newimg = cv2.resize(img,(250,250))#resize image for display on gui
    newimg = newimg[:, :, [2, 1, 0]]
    img = Image.fromarray(newimg)
    imgtk = ImageTk.PhotoImage(image=img)
    x.imgtk = imgtk
    x.configure(image=imgtk)
    xxx=cNN_test(f)
    n.set(disease[xxx])
    
    
       
       
##----------------------------------------------------------------------------------------------------------------------------

MF= Frame(root,bd=8, bg="lightgray", relief=GROOVE)
MF.place(x=0,y=0,height=50,width=669)
menu_label = Label(MF, text="Skin Disease Detection",
                    font=("times new roman", 20, "bold"),bg ="lightgray", fg="black", pady=0)
menu_label.pack(side=TOP,fill="x")

x = Label(root,image = im)
x.grid(row=1,column=1,padx=5,pady=50)

l1 = Label(root,font=("Courier", 15),text="Disease type:",fg="yellow", bg="black")
l1.grid(row=3,column=0,padx=1,pady=30)
n= StringVar()
n.set("")
l1_entry= Entry(root, font="arial 15",textvariable=n,width=25)
l1_entry.grid(row=3,column=1,padx=30)


b1 = Button(root,font=("times new roman", 20, "bold"),text='Select image', command=print_path)
b1.grid(row = 5, column = 1,padx =10, pady = 10)

root.mainloop()
