
from resnet.Resnet_feature_extractor import getResNet50Model

import numpy as np
import h5py
from scipy import spatial

class Resnet:
    def __init__(self):
        # # read features database (h5 file)
        h5f = h5py.File("ResnetFeatures.h5",'r')
        self.feats = h5f['dataset_1'][:]
        self.imgPaths= h5f['dataset_2'][:]
        h5f.close()

        # init Resnet50 model
        self.model = getResNet50Model()

    def query_image(self, queryImg,threshold):
        print(" searching for similar images")
        # #Extract Features
        X = self.model.extract_feat(queryImg)
        # Compute the Cosine distance between 1-D arrays
        imlist = []
        for i in range(self.feats.shape[0]):
            score = 1-spatial.distance.cosine(X, self.feats[i])
            if score>=threshold:
                imlist.append(self.imgPaths[i])
        return imlist
