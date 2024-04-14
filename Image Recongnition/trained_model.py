from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate


learn = load_learner('export.pkl')

img = load_image('image_data/seg_test/glacier/21982.jpg')
prediction = learn.predict(img)[0]
print(prediction)
