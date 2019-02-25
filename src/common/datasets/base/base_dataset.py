import numpy as np
import chainer
import os
import glob
from chainer import cuda, optimizers, serializers, Variable
import json
import h5py
from PIL import Image
from .dataset_augmentor import DatasetAugmentor
from chainer.dataset import dataset_mixin
import Augmentor.ImageUtilities 

class BaseDataset(dataset_mixin.DatasetMixin):
    def __init__(self, datasetConfig, prefix=None, _additional_augmentor_obj=None, return_meta=False, **kwargs):
        print("Create Dataset ...")
        print("config: ")
        #datasetConfig = json.load(open(datasetConfigPath, 'r'))
        print(datasetConfig)
        print("Prefix: ", prefix, "Others: ", _additional_augmentor_obj)
        self.augmentor = None 
        self._additional_augmentor_obj = _additional_augmentor_obj
        self._metaFilepath = []
        self._readArray = False # False for folder/json, True for hdf5/lmdb
        self._datasetConfig = datasetConfig
        if prefix is None and 'prefix' not in datasetConfig:
            self._isMultiLevelMetadata = False
        else:
            if prefix is not None:
                self._prefix = prefix
                self._isMultiLevelMetadata = True
            else:
                self._prefix = datasetConfig['prefix']
                self._isMultiLevelMetadata = True
        
            if 'string_before_prefix' in datasetConfig:
                self._prefix = datasetConfig['string_before_prefix'] + self._prefix 

        sourcePath = datasetConfig['source']
        if datasetConfig['type'] == 'folder':
            if self._isMultiLevelMetadata:
                sourcePath = sourcePath + '/' + self._prefix
            self._metaFilepath = Augmentor.ImageUtilities.scan_directory(sourcePath)

        elif datasetConfig['type'] == 'json':
            self._metaFilepath = json.load(open(sourcePath, 'r'))
            if self._isMultiLevelMetadata:
                self._metaFilepath = self._metaFilepath[self._prefix]

        elif datasetConfig['type'] == 'hdf5':
            self._h5fp = h5py.File(sourcePath, 'r', libver='latest')
            if self._isMultiLevelMetadata:
                self._data = self._h5fp[self._prefix]
            else:
                self._data = None
            if 'metaSource' in datasetConfig:
                self._metaFilepath = json.load(open(datasetConfig['metaSource'], 'r'))
                if self._isMultiLevelMetadata:
                    self._metaFilepath = self._metaFilepath[self._prefix]
            else:
                self._metaFilepath = [str(_) for _ in range(len(self._data))]
            self._readArray = True
            self._data = None
            self._h5fp.close()
            self._h5fp = None
        else:
            raise NotImplementedError
        
        self.weight = datasetConfig.get('weight', 1)
        self.return_meta = return_meta
        self._metaBasename = [os.path.basename(i) for i in self._metaFilepath]

        # Specify whitelist
        if 'whitelist' in datasetConfig:
            whitelist = set=json.load(open(datasetConfig['whitelist'], 'r'))
            _indexMapping = {}
            for i, file_ in enumerate(self._metaBasename):
                if file_ in whitelist:
                    _indexMapping[len(_indexMapping)] = i
            self._indexMapping = _indexMapping
            self._len = len(self._indexMapping)
            print('Whitelist is effective. Found %d out of %d data instances and %d whitelist entries' % (self._len, len(self._metaBasename), len(whitelist)))
        else:
            self._indexMapping = {i: i for i in range(len(self._metaFilepath))}
            self._len = len(self._metaFilepath)

        super().__init__(**kwargs)
    
    def get_example(self, i):
        if self._readArray:
            if self._data is None:
                sourcePath = self._datasetConfig['source'] 
                self._h5fp = h5py.File(sourcePath, 'r', libver='latest')
                if self._isMultiLevelMetadata:
                    self._data = self._h5fp[self._prefix]
                else:
                    self._data = self._h5fp
            image = Image.fromarray(self._data[i].transpose((1, 2, 0)))
        else:    
            image = Image.open(self._metaFilepath[self._indexMapping[i]])
        if self.augmentor is None:
            self.augmentor = DatasetAugmentor(self._datasetConfig, self._additional_augmentor_obj)
        image = self.augmentor.augment(image)
        if self.return_meta:
            return image, self._metaBasename[self._indexMapping[i]]
        else:
            return image

    def get_random_example(self):
        i = np.random.randint(0, self.__len__())
        return self.get_example(i)

    def get_random_images(self, batch_size):
        results = []
        for j in range(batch_size):
            i = np.random.randint(0, self.__len__())
            img = self.get_example(i)
            if isinstance(img, tuple):
                img, _ = img
            results.append(img)
        return np.asarray(results)
        
    def __len__(self):
        return self._len
        
    def close(self):
        try:
            if self._h5fp is not None:
                self._h5fp.close()
        except:
            pass
            