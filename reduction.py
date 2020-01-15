from sklearn.decomposition import IncrementalPCA
import numpy as np
import h5py
pca = IncrementalPCA(n_components=1712, batch_size = 1712)
file = h5py.File("data.h5", "r")
datasets = [i for i in file]
temp_data = 0
def func(dataset, temp_data):
    data = file[dataset]
    data_size = data.shape[0]
    start = 0
    if type(temp_data)!=int:
        batch = np.concatenate((temp_data, data[:1712-temp_data.shape[0]]), axis=0)
    while data_size-start>=1712:#1712
        if type(temp_data)!=int:
            pca.partial_fit(batch)
            start+=1712-temp_data.shape[0]
            temp_data = 0
        pca.partial_fit(data[start:start+1712])
        start+=1712
    if start!=data_size:
        temp_data = data[start:]
    else:
        temp_data = 0
    return temp_data

def func2(dataset):
    data = file[dataset]
    data_size = data.shape[0]
    rem = data_size%5
    step = int((data_size-rem)/5)
    start = 0
    for part in range(5):
        res = pca.transform(data[start:start+step+rem])
        result_file.create_dataset(dataset+str(part), data=res)
        start+=(step+rem)
        rem = 0
for dataset in datasets:
    temp_data = func(dataset, temp_data)
        
result_file = h5py.File("result_data.h5", "w")

for dataset in datasets:
    func2(dataset)
    
file.close()

result_file.close()



