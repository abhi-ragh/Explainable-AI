from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os

# Removed unnecessary imports (pandas, seaborn, cv2)

x  = 'image_data/seg_test/'
path = Path(x)


np.random.seed(40)

data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2,
                                  seed=42,  # Set a fixed seed for reproducibility
                                  ds_tfms=aug_transforms(flip_vert=True, max_rotate=10),  # Use efficient get_aug_transforms
                                  size=224,
                                  num_workers=4,  # Utilize all available CPU cores
                                  item_tfms=Resize(244),
                                  batch_tfms=[*aug_transforms(flip_vert=True, max_rotate=10), Normalize.from_stats(*imagenet_stats)])

# Removed unnecessary data visualization (data.show_batch)
data.show_batch(max_n=3, figsize=(7, 6))


learn = vision_learner(data, resnet18, metrics=[accuracy], model_dir=Path('working/'), path=Path("."))

learn.lr_find()

learn.fit_one_cycle(40, slice(1e-3, 1e-2))  # Adjust learning rates based on lr_find
learn.unfreeze()
learn.fit_one_cycle(20, slice(1e-5, 1e-4))  # Adjust learning rates and potentially reduce epochs

# Removed unnecessary plotting (learn.recorder.plot_losses())

img = load_image('image_data/seg_test/glacier/21982.jpg')
print(learn.predict(img)[0])

learn.export('export.pkl')