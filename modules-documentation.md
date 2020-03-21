The documentation shows some small and useful modules that helps the progress of the project

# 1. Layout detection

This module is inherited from the repository [here](https://github.com/eragonruan/text-detection-ctpn)

## a) Setup environment

```
cd Layout/text-detection-ctpn/utils/bbox

chmod +x make.sh

./make.sh
```

## b) Training

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

## c) Demo

If you want to see demo results on test data of SROIE, go to following link and download it [here](https://drive.google.com/drive/folders/1TYJDmql_ahQk_CyrfnpVnB_agZVcqTp3?usp=sharing), then put it under folder data inside Layout folder

Download pretrained model [here](https://drive.google.com/drive/folders/1k4bYCBJUlPzsFhEJGIv-5lk9tZQX3tvv?usp=sharing) and put all checkpoint record inside folder text-detection-ctpn/checkpoints_mlt


Move demo image inside folder text-detection-ctpn/data/demo, then run

```
python ./main/demo.py
```

Some results after running demo script

<img src="/Layout/text-detection-ctpn/data/res/X51005757342.jpg" width=320 height=480 /><img src="/Layout/text-detection-ctpn/data/res/X51005764154.jpg" width=320 height=480 />


# 2. OCR
- Download prepared dataset at [here](https://drive.google.com/file/d/1H8BWQz1z4Y93hqv3DdPI_qQi6dI0Mhs5/view?usp=sharing)
- Unzip the file ```unzip data.zip```
- Start training using CRNN module in `/crnn-ocr` by calling `python training.py`
# 3. KV

# 4. Explainer
