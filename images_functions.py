from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.preprocessing import image


#Plots a figure
def show_images(images, cols):
    
    n_images = len(images)
    fig = plt.figure()
    
    for n in range(len(images)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(images[n])

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def print_photos_for_every_label(labels, y_pred, y_test, batches, TRAIN_DIR):
    for l in range(len(labels)):
    
        print(labels[l]+"\n\n\n")
    
        #Getting samples for each category: fp, fn, tp, tn
        pred_labels = []
        true_labels = []
    
        #fp, fn, tp, tn
        categories = [[], [], [], []]
    
        #Values of [predicted label, true label] for each category
        categories_nums = [[0,1], [1,0], [1,1], [0,0]]
    
        for m in range(len(categories_nums)):
        
            i = 0
        
            while len(categories[m])<3:
            
                if y_pred[i][l]==categories_nums[m][0] and y_test[i][l]==categories_nums[m][1]:
                    categories[m].append(i)
                    pred_labels.append(y_pred[i])
                    true_labels.append(y_test[i])
                
                i+=1
        photos_per_category = [[batches[i][1] for i in categories[j]] for j in range(4)]
    
        photos_to_show = []
    
        #Getting 16 most different photos of each restaurant
        for arr in photos_per_category:
            for i in range(3):
            
                images = []
            
                for j in range(len(arr[i])):
                    img_path = TRAIN_DIR + str(arr[i][j]) + '.jpg'
                    img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
                    images.append(img.reshape(-1))
            
                dists = euclidean_distances(images, images)
                        
                indxs_of_photos_with_max_dist = get_max_dist_imgs(dists, [0], 16)
            
                photos_to_show.append([arr[i][indx] for indx in indxs_of_photos_with_max_dist])
    
        print("False Positive\n\n")
    
        for i in range(12):
            if i==3:
                print("False Negative\n\n")
            if i==6:
                print("True Positive\n\n")
            if i==9:
                print("True Negative\n\n")
        
            print(f"Restaurant {i}\n\n")
        
            imgs = []
        
            for x in range(16):
                im = Image.open(TRAIN_DIR + str(photos_to_show[i][x]) + '.jpg')
                im = im.resize((192, 192), Image.ANTIALIAS)
                imgs.append(im)
            
                if x==len(photos_to_show[i])-1:
                    break
        
            show_images(imgs, 4)
        
            pred_labels_str = list(map(lambda x:labels[x],[x for x in range(len(pred_labels[i])) if pred_labels[i][x]!=0]))
            true_labels_str = list(map(lambda x:labels[x],[x for x in range(len(true_labels[i])) if true_labels[i][x]!=0]))
        
            print("\n\n")
            print("\npredicted labels: ", pred_labels_str)
            print("\ntrue_labels: ", true_labels_str)
            print("\nfalse positive: ", [i for i in true_labels_str if i not in pred_labels_str])
            print("\nfalse negative: ", [i for i in pred_labels_str if i not in true_labels_str])
            print("\n\n")


#Returns matrix where rows are sorted vectors with indxes with max dist to others
def get_indxes(batches, TRAIN_DIR):
    
    indxes_list = []
    
    for i in tqdm(range(len(batches))):
        
        images = []
        
        for j in range(len(batches[i][1])):
            
            img_path = TRAIN_DIR + str(batches[i][1][j]) + '.jpg'
            img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
            images.append(img.reshape(-1))
        
        images_dists = euclidean_distances(images, images)
        
        indxes_list.append(get_max_dist_imgs(images_dists, [0], len(images)))
        
        del images
    
    return indxes_list


#Returns up to max_photos_num most different images 
def get_max_dist_imgs(dist_matrix, indx_list, max_photos_num):
    
    if len(indx_list)==max_photos_num:
        return indx_list
    
    dist_vector = dist_matrix[indx_list[0]]
    
    for i in range(1,len(indx_list)):
        dist_vector+=dist_matrix[indx_list[i]]
    
    indxes = np.array(dist_vector).argsort()[-(len(indx_list)+1):][::-1]
    
    a = [i for i in indxes if i not in indx_list]
    
    if not len(a):
        return indx_list
    
    indx_list.append(a[0])
    
    return get_max_dist_imgs(dist_matrix, indx_list, max_photos_num)

