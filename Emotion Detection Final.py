#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:13:29 2018

"""

#%%



#Importing Library

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Flatten,Dropout,Conv2D
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import subprocess
from keras.preprocessing import image


#%%

model=Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True
                                 )
test_datagen = ImageDataGenerator(rescale = 1./255)   
training_set = train_datagen.flow_from_directory('Model2/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 color_mode='grayscale'
                                                 )
test_set = test_datagen.flow_from_directory('Model2/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            color_mode='grayscale')

model.fit_generator(training_set,
                         steps_per_epoch = 12 ,
                         nb_epoch = 12,
                         validation_data = test_set,
                         nb_val_samples = 5)

model.summary()

#%%

model.save("Final_Model_categories_3.model")
#%%
model = keras.models.load_model("Final_Model_categories_3.model")

    
    
#%%
#SHOW CLASS INDICES 
    
training_set.class_indices

#%%

#PLAYING SONGS
import cv2
import os


try:
choice=int(input("Which mode do you want to ?\n1.Offline\n2.Online"))

print("Take an image : \n")
cap=cv2.VideoCapture(0)


while(1):

    try:
        _,frame2=cap.read()
        frame2 = cv2.flip(frame2,1)
        cv2.imshow("frame",frame2)
        
        x=cv2.waitKey(5)
        if x==ord('c'):
        
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame2, 1.3, 5)
            for (x,y,w,h) in faces:
                frame2 = frame2[y:y+h, x:x+w]
                
            cv2.imwrite("img.jpeg",frame2)
            cv2.waitKey(5)
            break;
    except OSError:
        print("Try Again")
            
    
    
    cv2.destroyAllWindows()
    cap.release()




import numpy as np
from keras.preprocessing import image
test_image = image.load_img('img.jpeg', target_size = (64, 64),grayscale=1)

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if choice==1:

    itunes=os.listdir("SONGS")
    if result[0][0] == 1:
        y=3
        
        print("\nYou seem Excited :(\n ")
        
        prediction="Excited"
        
        songs=os.listdir("SONGS/"+itunes[y])
        print("\nSome songs for your Exciting mood :) ")
        for i in range(0,len(songs)):
            print(str(i)+". "+songs[i])
        x=int(input("What song do you want to listen to?\n"))
        subprocess.run(["open","-a","/Applications/itunes.app","SONGS/"+itunes[y]+"/"+songs[x]])
        print("Playing song " + songs[x])
        
         
        
    
    if result[0][1]==1:
        
        
        prediction="Happy"
    
        print("You seem Happy :) \n ")
            
        
        y=1
        songs=os.listdir("SONGS/"+itunes[y])
        print("\nSome songs for your happy mood :) ")
        
        for i in range(0,len(songs)):
            print(str(i)+". "+songs[i])
        x=int(input("What song do you want to listen to?\n"))
        subprocess.run(["open","-a","/Applications/itunes.app","SONGS/"+itunes[y]+"/"+songs[x]])
        print("Playing song "  + songs[x])
        
        
    if result[0][2]==1:
        
        print("\nYou seem Sad :( \n ")
        prediction="Sad"
        y=2
        songs1=os.listdir("SONGS/"+itunes[y])
        print("\nSome songs for your sad mood :( ")
        for i in range(0,len(songs1)):
            print(str(i)+". "+songs1[i])
        
        print("\nSongs to cheer up your mood : \n")
        y=1
        songs2=os.listdir("SONGS/"+itunes[y])
    
        for i in range(0,len(songs2)):
            print(str(i+len(songs1))+". "+songs2[i])
            
        x=int(input("What song do you want to listen to?\n"))
        if(x<len(songs1)):
            y=2
            subprocess.run(["open","-a","/Applications/itunes.app","SONGS/"+itunes[y]+"/"+songs1[x]])
            print("Playing song " + songs1[x])
        else:
            y=1
            subprocess.run(["open","-a","/Applications/itunes.app","SONGS/"+itunes[y]+"/"+songs2[x-len(songs1)]])
            print("Playing song " + songs2[x-len(songs1)])
elif choice==2:
    import webbrowser
     
    happy = ['POP','EDM','Jazz','IndieRock']
    sad = ['Instrumental','POP','Rock','Deep','classic']
    Excited = ['EDM','House','HIP-HOP','Disco']
    
    if result[0][1] ==1 :
        count= 1
        print('!!--You seem Happy please select your favourite genre--!!')
        for i in happy:
            print('{} : {}'.format(count,i))
            count += 1
        x = int(input('Enter Your Choice:'))
        if x == 1:
            webbrowser.open('https://www.youtube.com/watch?v=pgN-vvVVxMA&list=PLDcnymzs18LU4Kexrs91TVdfnplU3I5zs&start_radio=1')
        elif x== 2:
            webbrowser.open('https://www.youtube.com/watch?v=lTx3G6h2xyA&list=PLUg_BxrbJNY5gHrKsCsyon6vgJhxs72AH&start_radio=1')
        elif x==3:
            webbrowser.open('https://www.youtube.com/watch?v=21LGv8Cf0us&list=PLMcThd22goGYit-NKu2O8b4YMtwSTK9b9&start_radio=1')
        elif x==4:
            webbrowser.open('https://www.youtube.com/watch?v=VQH8ZTgna3Q&list=PLVAJ90ZhCcL896CZDbuIz2HGKeVekfEee&start_radio=1')
        else:
            webbrowser.open('https://www.youtube.com')
            
        print("Playing " + happy[x-1])
            
    elif result[0][2] == 1:
        count= 1
        print('!!--You Seem Sad please select your favourite genre--!!')
        for i in sad:
            print('{} : {}'.format(count,i))
            count += 1
        x = int(input('Enter Your Choice:'))
        if x == 1:
            webbrowser.open('https://www.youtube.com/watch?v=Pa__NZaRXxs&list=PLIWSikhI2_z2lNqsfjF4ahut28056cJtz')
        elif x== 2:
            webbrowser.open('https://www.youtube.com/watch?v=pgN-vvVVxMA&list=PLDcnymzs18LU4Kexrs91TVdfnplU3I5zs&start_radio=1')
        elif x==3:
            webbrowser.open('https://www.youtube.com/watch?v=6Ejga4kJUts&list=PLhd1HyMTk3f5PzRjJzmzH7kkxjfdVoPPj&start_radio=12')
        elif x==4:
            webbrowser.open('https://www.youtube.com/watch?v=UAWcs5H-qgQ&list=PLzzwfO_D01M4nNqJKR828zz6r2wGikC5a')
        elif x==5:
            webbrowser.open('https://www.youtube.com/watch?v=4Tr0otuiQuU&list=RDQMqk8OvYGJVWM&start_radio=1')
        else:
            webbrowser.open('https://www.youtube.com')
        print("Playing " + sad[x-1])
            
        
    elif result[0][0] == 1:
        count= 1
        print('!!--You seem Excited please select your favourite genre--!!')
        for i in Excited:
            print('{} : {}'.format(count,i))
            count += 1
        x = int(input('Enter Your Choice:'))
        if x == 1:
            webbrowser.open('https://www.youtube.com/watch?v=lTx3G6h2xyA&list=PLUg_BxrbJNY5gHrKsCsyon6vgJhxs72AH&start_radio=1')
        elif x== 2:
            webbrowser.open('https://www.youtube.com/watch?v=BDocp-VpCwY&list=PLhInz4M-OzRUsuBj8wF6383E7zm2dJfqZ&start_radio=1')
        elif x==3:
            webbrowser.open('https://www.youtube.com/watch?v=xTlNMmZKwpA&start_radio=1&list=PLH6pfBXQXHEC2uDmDy5oi3tHW6X8kZ2Jo')
        elif x==4:
            webbrowser.open('https://www.youtube.com/watch?v=kJQP7kiw5Fk&list=PL64E6BD94546734D8')
        else:
            webbrowser.open('https://www.youtube.com')
        print("Playing " + Excited[x-1])
    else:
        webbrowser.open('https://www.youtube.com/watch?v=aJOTlE1K90k&list=PLw-VjHDlEOgvtnnnqWlTqByAtC7tXBg6D')
        
            
        
        
            