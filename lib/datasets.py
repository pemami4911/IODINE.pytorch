import torch
import torchvision
import numpy as np
import h5py
from pathlib import Path
from sacred import Ingredient
import cv2
from PIL import Image

ds = Ingredient('dataset')

@ds.config
def cfg():
    data_path = ''  # base directory for data
    h5_path = '' # dataset name
    masks = False
    preprocess_style = 'basic'


class StaticHdF5Dataset(torch.utils.data.Dataset):
    """
    Dataset class for reading seqs of images from an HdF5 file
    """
    @ds.capture
    def __init__(self, data_path, h5_path, masks, preprocess_style, d_set='train'):
        super(StaticHdF5Dataset, self).__init__()
        self.h5_path = str(Path(data_path, h5_path))
        self.d_set = d_set.lower()
        self.masks = masks
        self.preprocess_style = preprocess_style


    def preprocess(self, img):
        """
        """
        PIL_img = Image.fromarray(np.uint8(img))
        if self.preprocess_style == 'clevr-large':
            # square center crop of 192 x 192
            PIL_img = PIL_img.crop((64,29,256,221))
            # reshape to 128 x 128  # TODO: change back to 128 later
            PIL_img = PIL_img.resize((128,128))
        elif self.preprocess_style == 'clevr-small':
            # square center crop of 192 x 192
            PIL_img = PIL_img.crop((64,29,256,221))
            # reshape to 128 x 128  # TODO: change back to 128 later
            PIL_img = PIL_img.resize((96,96))

        img = np.transpose(np.array(PIL_img), (2,0,1))
        img = img / 255.  # to 0-1
        
        return img

    def preprocess_mask(self, mask):
        """
        [objects, h, w, c]
        """
        #import pdb; pdb.set_trace()
        o,h,w,c = mask.shape
        masks = []
        for i in range(o):
            mask_ = mask[i,:,:,0]
            PIL_mask = Image.fromarray(mask_, mode="F")
            # square center crop of 192 x 192
            PIL_mask = PIL_mask.crop((64,29,256,221))
            # reshape to 128 x 128  # TODO: change back to 128 later
            #PIL_mask = PIL_mask.resize((96,96), resample=1)
            masks += [np.array(PIL_mask)[...,None]]
        mask = np.stack(masks)  # [o,h,w,c]
        mask = np.transpose(mask, (0,3,1,2))
        return mask    
    

    def __len__(self):
        with h5py.File(self.h5_path,  'r') as data:
            data_size, _, _, _ = data[self.d_set]['imgs'].shape
            return data_size


    def __getitem__(self, i):
        with h5py.File(self.h5_path,  'r') as data:
            outs = {}
            outs['imgs'] = self.preprocess(data[self.d_set]['imgs'][i].astype('float32')).astype('float32')
            if self.masks:
                if self.preprocess_style == 'clevr-large' or self.preprocess_style == 'clevr-small':
                    outs['masks'] = self.preprocess_mask(data[self.d_set]['masks'][i].astype('float32'))
                else:
                    outs['masks'] = np.transpose(data[self.d_set]['masks'][i].astype('float32'), (0,3,1,2))

            return outs
