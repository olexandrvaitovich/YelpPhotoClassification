import numpy as np
import pandas as pd

#Input: X-features, photos-photos_id to business_id, Y-labels,
#i-index of last processed data row, train-whether it train data or test
#Output: None
def data_to_pickle(X, photos, Y, beg, end, train=True):
    
    X_df = pd.DataFrame({'features' : X})
    
    photo_to_biz = photos[beg:end]
    
    #Drop indexes to make features and photos match each other
    X_df.reset_index(drop=True, inplace=True)
    
    photo_to_biz.reset_index(drop=True, inplace=True)
    
    #Creating DataFrame with features and business_id
    df = pd.concat([X_df, photo_to_biz], axis=1)
    
    #Taking mean of every restaurant`s features
    grouped_df = pd.DataFrame(df.groupby("business_id")["features"].apply(np.mean))
    
    grouped_df.reset_index(level=0, inplace=True)
    
    #Filtering labels
    if train:
        Y_part = Y[Y["business_id"].isin(df["business_id"])]
    
        sorted_labels = Y_part.sort_values("business_id")
    
        sorted_labels.reset_index(level=0,drop=True, inplace=True)
    
        grouped_df["labels"] = sorted_labels["labels"]
    
    #Removing rows with NaNs
    nans = pd.isnull(grouped_df).any(1).nonzero()[0]
    cleared_df = grouped_df.drop(grouped_df.index[list(nans)])
        
    #Writing data to CSV
    cleared_df.to_pickle('{}_data{}.csv'.format("train" if train else "test", end))
        
    
#Reading the pickle by filename and merging it with existing pickle    
def merge_pickles(filename, pickle):
    
    temp_df = pd.read_pickle(filename)
    
    new_pickle = pickle.append(temp_df, ignore_index=True)
    
    return new_pickle