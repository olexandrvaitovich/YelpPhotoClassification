import h5py
import numpy as np


#Fitting big data to incremental pca
def fitting_pca(pca, data_file, dataset, temp_data, n_components):
    
    data = data_file[dataset]
    data_size = data.shape[0]
    start = 0
    
    if type(temp_data)!=int:
        
        batch = np.concatenate((temp_data, data[:n_components-temp_data.shape[0]]), axis=0)
    
    while data_size-start>=n_components:
        
        if type(temp_data)!=int:
            
            pca.partial_fit(batch)
            
            start+=n_components-temp_data.shape[0]
            
            temp_data = 0
        
        pca.partial_fit(data[start:start+n_components])
        
        start+=n_components
    
    if start!=data_size:
        
        temp_data = data[start:]
    
    else:
        
        temp_data = 0
    
    return temp_data


#Transforming big data
def pca_transform(pca, data_file, dataset, divider, result_file):
    
    data = data_file[dataset]
    data_size = data.shape[0]
    
    rem = data_size%divider
    step = int((data_size-rem)/divider)
    start = 0
    
    for part in range(divider):
        
        res = pca.transform(data[start:start+step+rem])
        
        result_file.create_dataset(dataset+str(part), data=res)
        
        start+=(step+rem)
        
        rem = 0