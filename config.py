import pandas as pd
#Dirs with photos
TRAIN_DIR = "D:/train_photos/"
TEST_DIR = "D:/test_photos/"

DATA_DIR = "D:/"

#CSV files with photos ids and corresponding business ids 
#test_photo_to_biz = pd.read_csv("D:/test_photo_to_biz.csv")
train_photo_to_biz = pd.read_csv("D:/train_photo_to_biz_ids.csv")

#CSV file with labels corresponding to business
labels = pd.read_csv("D:/train.csv")
max_img_amount = 10000