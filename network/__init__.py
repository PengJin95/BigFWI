from .fcn import *
from collections import OrderedDict

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers')
              .replace('Resizelayers', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

model_dict = {
    'FCN4_Deep_2': FCN4_Deep_2, # BigFWI-B
    'FCN4_Deep_2L': FCN4_Deep_2L, # BigFWI-M
    'FCN4_Deep_2XL': FCN4_Deep_2XL, # BigFWI-XL
    'FCN4_Deep_2XL_2': FCN4_Deep_2XL_2, # BigFWI-L
}
