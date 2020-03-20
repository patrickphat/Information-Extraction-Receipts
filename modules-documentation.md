The documentation shows some small and useful modules that helps the progress of the project

# 1. Layout detection

This module is inherited from the repository [here](https://github.com/eragonruan/text-detection-ctpn)

Checkpoint training model(50k iter): [here]()

Data for training: [here](https://drive.google.com/drive/folders/1OTl3BRs3iBCTLyCGMHno1VFxyxZzxs4Z?usp=sharing)

a) Setup environment

```
cd Layout/text-detection-ctpn/utils/bbox

chmod +x make.sh

./make.sh
```

b) Training

Download pretrained VGG model and put it in data/vgg_16.ckpt link: [tensorflow/models](https://github.com/tensorflow/models/tree/1af55e018eebce03fb61bba9959a04672536107d/research/slim

Download format_dataset from [here](https://drive.google.com/drive/folders/1OTl3BRs3iBCTLyCGMHno1VFxyxZzxs4Z?usp=sharing) and
put format_data folder inside Layout folder

Download checkpoint for model [here](https://drive.google.com/drive/folders/1k4bYCBJUlPzsFhEJGIv-5lk9tZQX3tvv?usp=sharing) and put all checkpoint record inside folder checkpoints_mlt
Run following script to prepare training data

```
cd ../../

python ./utils/prepare/split_label.py
```

and then train

```
python ./main/train.py
```

c) Demo

If you want to see demo results on test data of SROIE, go to following link and download it [here](https://drive.google.com/drive/folders/1TYJDmql_ahQk_CyrfnpVnB_agZVcqTp3?usp=sharing), then put it under folder data inside Layout folder

Download pretrained model [here](https://drive.google.com/drive/folders/1k4bYCBJUlPzsFhEJGIv-5lk9tZQX3tvv?usp=sharing) and put all checkpoint record inside folder checkpoints_mlt


Move demo image inside folder text-detection-ctpn/data/demo, then run

```
python ./main/demo.py
```


# 2. OCR
Download prepared pickled dataset at [here](https://drive.google.com/file/d/1-0bRc91c-50S38oC3JYE9BcWwogheiRg/view?usp=sharing)

Also the pickled labels for the dataset at here [here](https://drive.google.com/file/d/1-5jkZ7YT23tCd1-P_5AvKmR3cTyQIJ4n/view?usp=sharing)

Use module from `/helpers/FilePickling` to load pickled file, for example:

```python
from helpers.FilePickling import pkl_load

# This return a list of cropped image patches
img_patches = pkl_load("patches.pkl") 

# This return ground truth for each patches
img_patches_labels = pkl_load("labels.pkl") 
```

# 3. KV

# 4. Explainer
