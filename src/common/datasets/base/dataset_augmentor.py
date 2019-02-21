
import numpy as np
import Augmentor
from PIL import Image

class DatasetAugmentor():
    def __init__(self, 
            dataset_config=None,
            additional_augmentor_obj=None
        ):
        self.p = Augmentor.Pipeline()

        if dataset_config is not None and 'pipeline' in dataset_config:
            for pipeline in dataset_config['pipeline']:
                method_to_call = getattr(self.p, pipeline[0])
                parameters = pipeline[1]
                method_to_call(**parameters)
            if additional_augmentor_obj is not None:
                for pipeline in additional_augmentor_obj:
                    method_to_call = getattr(self.p, pipeline[0])                    
                    parameters = pipeline[1]
                    method_to_call(**parameters)

        self.transform = self.p.torch_transform()
        if dataset_config is not None and 'scaling' in dataset_config:
            self.scaling = dataset_config['scaling']
        else:
            self.scaling = 'tanh'

    def _scaling_tanh(self, img):
        img = img / 127.5 - 1
        return img

    def _scaling_sigmoid(self, img):
        img = img / 255.0
        return img

    def augment(self, image, isArray=False):
        if isArray: # if the input is a numpy array, convert back to PIL
            image = Image.fromarray(image)
        image = self.transform(image)
        image = np.asarray(image).astype('f')
        w, h = image.shape[0], image.shape[1]
        if np.ndim(image) == 2:
            ch = 1
        else:
            ch = np.shape(image)[2]
        image = image.reshape(w, h, ch)
        image = image.transpose((2, 0, 1))
        if self.scaling == 'none':
            return image 
        elif self.scaling == 'sigmoid':
            return self._scaling_sigmoid(image)
        elif self.scaling == 'tanh':
            return self._scaling_tanh(image)
        else:
            raise NotImplementedError
            