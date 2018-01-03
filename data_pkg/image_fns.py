import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

def return_train_test_images(eval_id):

    eval_path = os.path.join('trafficdb',eval_id)
    train_path = os.path.join(eval_path,'train')
    test_path = os.path.join(eval_path,'test')

    train_images = []
    train_labels = []

    test_images =[]
    test_labels = []

    for i in tqdm(os.listdir(train_path)):
        for j in tqdm(os.listdir(os.path.join(train_path,i))):
            path_img = os.path.join(train_path,i,j)
            train_images.append(resize(imread(path_img),(200,200)))
            train_labels.append(int(i))

    for i in tqdm(os.listdir(test_path)):
        for j in tqdm(os.listdir(os.path.join(test_path,i))):
            path_img = os.path.join(test_path, i, j)
            test_images.append(resize(imread(path_img),(200,200)))
            test_labels.append(int(i))

    return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)

