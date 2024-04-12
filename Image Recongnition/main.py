from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

x  = 'image_data/seg_train/'
path = Path(x)


np.random.seed(40)
#data = ImageDataLoaders.from_folder(path, train = '.', valid_pct=0.2,
#                                  ds_tfms=aug_transforms(), size=224,
#                                  num_workers=4).normalize(imagenet_stats)

#data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2,
#                                    ds_tfms=aug_transforms(), size=224,
#                                    num_workers=4,
#                                    batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])

data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2,
                                  ds_tfms=aug_transforms(), size=224,  # Resize to a fixed size (e.g., 224x224)
                                  num_workers=4,
                                  item_tfms = Resize(244),
                                  batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])




#data.show_batch(rows=3, figsize=(7,6),recompute_scale_factor=True)
data.show_batch(max_n=3, figsize=(7, 6))



learn = vision_learner(data, models.resnet18, metrics=[accuracy], model_dir = Path('working/'),path = Path("."))
learn.lr_find()
#learn.recorder.plot(suggestions=True)
lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(40,slice(lr1,lr2))
learn.unfreeze()
learn.fit_one_cycle(20,slice(1e-4,1e-3))
#learn.recorder.plot_losses()
#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix()

img = load_image('image_data/seg_test/glacier/21982.jpg')
print(learn.predict(img)[0])

learn.save('working/model.pkl')