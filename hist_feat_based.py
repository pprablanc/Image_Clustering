#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import cv2

# Sklearn
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

# Keras
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Dense
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard

# Plot
import matplotlib.pyplot as plt

path = os.getcwd()
raw_data_directory = os.path.join(path, 'Raw_data')
list_filename = os.listdir(raw_data_directory)
list_filename_full = [os.path.join(path, 'Raw_data', f) for f in list_filename]


def extract_hist(f, scale_percent = 50):
    hist_size = 64
    color = ('b','g','r')
    im = cv2.imread(f)
    width = int( np.shape(im)[1] * scale_percent / 100 )
    height = int( np.shape(im)[0] * scale_percent / 100 )
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_NEAREST)
    hist_channel = [cv2.calcHist([im],[i],None,[hist_size],[0,hist_size]) for i, col in enumerate(color)]
    hist = np.concatenate(hist_channel)
    #hist = sum(hist_channel, 1)
    return hist


hist_data = 'load'
N_images = np.size(list_filename) # All images

if hist_data == 'compute':
    # Compute histograms in parallel
    pool = mp.Pool(mp.cpu_count())
    hist = pool.starmap(extract_hist, [(f, 100) for f in list_filename_full[0:N_images]])
    hist = np.squeeze(np.array(hist)) / 255.0
    pool.close()

    # Sauver ces histograms
    np.save('histogram.npy', hist)

elif hist_data == 'load':
    # Charger les histograms sauvegardés
     hist = np.load('histogram.npy')


M = hist
ndim_red = 32
dim_red_method = "autoencoder"

if dim_red_method == "pca":
    M = preprocessing.scale(hist)
    Mp = pd.DataFrame(M, index=list_filename)

    pca = PCA()
    Y = pca.fit_transform(Mp)

    print(pca.explained_variance_ratio_[0:20])

elif dim_red_method == "autoencoder":
    x_train, x_test = train_test_split(M, test_size=0.2, random_state=123)

    input_vec = Input(shape= (M.shape[1],) )
    encoded = Dense(ndim_red, activation='relu')(input_vec)
    decoded = Dense(M.shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input_vec, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Encoder is needed to get only the encoded data
    encoder = Model(input_vec, encoded)

    # Decoder
    input_decoder = Input(shape=(ndim_red), )
    decoder_layer = Dense(M.shape[1], activation='sigmoid')(input_decoder)
    decoder = Model(input_decoder, decoder_layer)

    # Training on data
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='./hist_based/')])



N_CLUSTERS = 3
# Les données sont centrées/réduites
clf_kmeans = make_pipeline(preprocessing.StandardScaler(), KMeans(n_clusters=N_CLUSTERS, random_state=0))
clf_kmeans.fit(Y[:, 0:ndim_red])
# clf_hc = make_pipeline(preprocessing.StandardScaler(), AgglomerativeClustering(n_clusters=N_CLUSTERS))
# clf_hc.fit(Y)
colors = [str(elt) for elt in clf_kmeans['kmeans'].labels_]
#colors = [str(elt) for elt in clf_hc['agglomerativeclustering'].labels_]
Ydf = pd.DataFrame(Y[:,0:2])
Ydf['class'] = colors
Ydf['id'] = list_filename
Ydf.columns = ['0', '1', 'class', 'id']

#######################
### CREATE FOLDERS ####
#######################

# from shutil ios.rmport copyfile
import shutil

for c in pd.unique(clf_kmeans['kmeans'].labels_):
    d = os.path.join('Results', str(c))
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.mkdir(d)


for i in range(len(Ydf)):
    f_src = os.path.join(path, 'Raw_data', Ydf['id'][i])
    f_tar = os.path.join(path, 'Results', str(Ydf['class'][i]), Ydf['id'][i])
    #copyfile(f_src, f_tar)
    os.symlink(f_src, f_tar)
    


#######################
#### PLOTS ############
#######################

import pandas as pd
import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.palettes import d3
bpl.output_file("line.html")

source = bpl.ColumnDataSource.from_df(Ydf)

# use whatever palette you want...
palette = d3['Category10'][len(Ydf['class'].unique())]
color_map = bmo.CategoricalColorMapper(factors=Ydf['class'].unique(), 
                                       palette=palette)

# create figure and plot
p = bpl.figure(plot_width=1400, plot_height=1000)
p.scatter(x='0', y='1',
          color={'field': 'class', 'transform': color_map},
          legend='class', source=source)
bpl.show(p)





filename_sombre = os.path.join(path, 'Raw_data', "DSC_5259.jpg")
filename_clair = os.path.join(path, 'Raw_data', "D81_5004.jpg")

H_sombre = extract_hist(filename_sombre)
H_clair = extract_hist(filename_clair)

# plt.figure()
# plt.plot(H_sombre)
# plt.plot(H_clair)



