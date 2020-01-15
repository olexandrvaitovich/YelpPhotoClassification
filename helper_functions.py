from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
from keras.preprocessing import image
from time import time
import h5py



#Input -list of ids
#Output list of photos
def read_photos(TRAIN_DIR, ids_list, plot):
    
    images = []
    
    for i in ids_list:
        
        img_path = TRAIN_DIR + str(i) + '.jpg'
        
        if plot:
            
            img = Image.open(img_path)
            img = img.resize((192, 192), Image.ANTIALIAS)
        
        else:
            
            img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
        
        images.append(img)
    
    return images


#Input photos_ids
#Output embeddings
def get_embeddings(TRAIN_DIR, model, batches, i, empty, embds = None):
    
    photos = np.array(read_photos(TRAIN_DIR, batches[i][1], False))
    
    photos_embds = model.predict(photos)
    
    if empty:
        
        return photos_embds
    
    else:
        
        return np.concatenate([embds, photos_embds], axis=0)
    
    
#Input: directory_name
#Output: list with global paths of files
def get_filenames(directory):
    
    rootdir = Path(directory)
    
    return [str(f) for f in rootdir.resolve().glob('**/*') if f.is_file() and "._" not in str(f)]


def embeddings_to_file(filename, TRAIN_DIR, batches, model, sup, beg):
    
    data_file = h5py.File(f"{filename}.h5", "w")
    
    embeddings = get_embeddings(TRAIN_DIR, model, batches, beg, True)
    
    time_list = []
    
    n = 0
    k = 0
    status = False
    
    start = time()
    
    for i in range(beg+1, len(batches)):
        
        if status:
            
            embeddings = get_embeddings(TRAIN_DIR, model, batches, i, True)
            
            status = False
        
        else:
                        
            print(embeddings.shape)            
                        
            embeddings = get_embeddings(TRAIN_DIR, model, batches, i, False, embeddings)
        
        if embeddings.shape[0]>10000:
            
            n = i
            k+=1
            print(len(batches[i][1]))
            print("Written shape: "+str(embeddings.shape))
            
            data_file.create_dataset(f"part_{n}", data=embeddings)
            
            status = True
            
            lap = time()-start
            time_list.append(lap)
            print(lap)
            
            if k==sup:
                
                print(n)
                data_file.close()
                break

    if k!=sup:

        n = 1995

        print("Written shape: "+str(embeddings.shape))
            
        data_file.create_dataset(f"part_{n}", data=embeddings)
            
        lap = time()-start
        time_list.append(lap)
        print(lap)
     
    print(n)

    data_file.close()


#Input - two pandas dataframes
#Output - list of lists with business_ids and photos_ids to this business, list of labels
def get_batches(train_photo_to_biz, lbs):
    batches = [] 

    for i in range(5000):
        #Container with all photos of each restaurant and it`s business_id
        batch = train_photo_to_biz[train_photo_to_biz["business_id"]==i]["photo_id"].to_numpy()
        if batch.size:
            batches.append((i, batch))

    #Labels and batches with existing data only
    lbs = lbs[lbs["business_id"].isin([i[0] for i in batches])]
    lbs = lbs[~lbs["labels"].isna()]

    lb = list(map(lambda x:int(x), list(lbs["business_id"])))

    batches = list(filter(lambda x:x[0] in lb, batches))

    lbs = lbs.sort_values(by="business_id")

    labels_temp = [l.split() for l in lbs["labels"]]

    lbs = np.array(list(map(lambda x:list(map(lambda y:int(y), x)), labels_temp)))
    
    return batches, lbs
