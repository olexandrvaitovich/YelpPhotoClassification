from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing import image
import numpy as np
from random import shuffle
import pandas as pd



def pred_generator_model4(filenames, businesses, train_photo_to_biz):
    
    while True:
        
        X = []
        
        for f in filenames:
            
            businesses.append(train_photo_to_biz[train_photo_to_biz["photo_id"]==int(f[-f[::-1].index("/"):-(f[::-1].index(".")+1)])].iloc[0]["business_id"])
            
            img_path = f
            img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
            
            X.append(img)
            
            if len(X)==2:
                
                yield np.array(X)
                
                X = []
        
        yield np.array(X)

def batch_generator_model3(filenames, businesses, train_photo_to_biz, labels):
    
    #To get labels in binary form
    MB = MultiLabelBinarizer()
    MB.fit(np.array([[0,1,2,3,4,5,6,7,8]]))
    
    while True:
        
        #Every epoch
        shuffle(filenames)
        
        X = []
        Y = []
        #To identify photos later
        business = []
        
        for f in filenames:
            
            if hasDigits(f[-f[::-1].index("\\"):-(f[::-1].index(".")+1)]):
                
                bsns_id = train_photo_to_biz[train_photo_to_biz["photo_id"]==int(f[-f[::-1].index("\\"):-(f[::-1].index(".")+1)])].iloc[0]["business_id"]
                label = labels[labels["business_id"]==bsns_id].iloc[0]["labels"]
                
                if label not in [None, "NaN"] and type(label)!=float:
                    
                    label = MB.transform(np.array([list(map(lambda x:int(x),label.split()))]))[0]
                    
                    Y.append(label)
                    business.append(bsns_id)
                    
                    img_path = f
                    img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
                    
                    X.append(img)
            
            if len(X)==20:
                
                #Shiffling model input
                c = list(zip(X,Y, business))
                
                shuffle(c)
                
                X, Y, business = zip(*c)
                
                businesses+=business
                
                yield np.array(X), np.array(Y)
                
                X = []
                Y = []
                business = []
        
        yield np.array(X), np.array(Y)


#Function for checking filenames for containing photo_id    
def hasDigits(inputString):
    
    return any(i.isdigit() for i in inputString)


#Generates batches for training and validation
def batch_generator(batches, sorted_indxes, TRAIN_DIR, Y_train, businesses, img_num=10, rests_num=4):
    
    numbers = [i for i in range(len(batches))]
    beg = 0
    while True:
        #Shuffling at the begining of epoch
        shuffle(numbers)
        
        batch = []
        Y = []
        #Batch containes img_num photos for rests_num restaurants
        #labels are numbers in range rests_num which inform to what restaurant this photo belongs
        labels_for_batch = []
        
        for i in range(len(batches)):
            n = numbers[i]
            y = []
            images_for_batch = []
            
            #To train on entire dataset I take next img_num photos from sorted_indxes for each epoch
            end = beg+img_num if len(sorted_indxes[n])>=beg+img_num else len(sorted_indxes[n])
            indxes_of_photos_with_max_dist = sorted_indxes[n][beg:end]

            for j in indxes_of_photos_with_max_dist:
                
                #Business_id for every image to be able to identify results
                businesses.append(batches[n][0])
                
                img_path = TRAIN_DIR + str(batches[n][1][j]) + '.jpg'
                img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
                
                images_for_batch.append(img)
                y.append(Y_train[n])

            batch+=images_for_batch
            labels_for_batch+=[x%rests_num for x in range(len(indxes_of_photos_with_max_dist))]
            Y+=y[:len(indxes_of_photos_with_max_dist)]
            
            if (i+1)%rests_num==0:
                
                yield np.array(batch), np.array(labels_for_batch), np.array(Y)
                
                batch = []
                Y = []
                labels_for_batch = []
        
        #shift m to take next indxes from sorted_indxes
        beg+=img_num