import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from glob import glob
from keras.preprocessing.image import load_img, img_to_array

files = glob('C:/Users/20182413/Documents/Year 3 Quartile 3/BIA project/8p361-project-imaging-master/assignments/test/*.tif')
#files[:10]

count =0
for file in files:
    if file[-5] == '1':
        count+=1
#print(count)

def show_img(files):
    plt.figure(figsize=(10, 10))
    ind = np.random.randint(0, len(files), 25)
    i = 0
    for loc in ind:
        plt.subplot(5, 5, i + 1)
        sample = load_img(files[loc], target_size=(150, 150))
        sample = img_to_array(sample)
        plt.axis("off")
        plt.imshow(sample.astype("uint8"))
        i += 1

show_img(files[:10])

import seaborn as sns


