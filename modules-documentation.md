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

## Data preparation

- Download prepared dataset at [here](https://drive.google.com/file/d/1H8BWQz1z4Y93hqv3DdPI_qQi6dI0Mhs5/view?usp=sharing)
- Unzip the file ```unzip data.zip```
- Start training using CRNN module in `/crnn-ocr` by calling `python training.py`

Folder structure:
```
└── data_train
    ├── 00000.jpg
    ├── 00001.jpg
    ├── 00002.jpg
    ├── ...
    └── 34054.jpg
└── data_test
    ├── 00000.jpg
    ├── 00001.jpg
    ├── 00002.jpg
    ├── ...
    └── 34054.jpg
└── labels_train.txt
└── labels_test.txt
```
For `labels_train` and `labels_test`, each lines represent the labels for each image in `data_train` and `data_test` respectively.

## Training
This module is inherited from the [repository](https://github.com/eragonruan/text-detection-ctpn)

### Install dependencies:
```
git clone https://github.com/patrickphatnguyen/Optical-Character-Recognition-KV.git

cd crnn-ocr/task2
# installing torch and lmdb
pip install torch
pip install lmdb
# setup warp-ctc
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build
cd build
cmake ..
make
cd pytorch_binding
python setup.py install
```
### Training

+ Download dataset at put in under data_train and data_valid.

 - [data_train](https://drive.google.com/drive/folders/1--rbdb48OWxJi6m0blesTkJbSNm95Peb?usp=sharing)
 
 - [data_valid](https://drive.google.com/drive/folders/16DyQmbTg4YwhczR26X1u8lFyKwaTFXvI?usp=sharing)
+ Run create_dataset.py(should run with python2 environment) to prepare dataset in lmdb format. Prepared dataset after running this script will be found at dataset/

+ Training
```
python train.py --adadelta --trainroot dataset/train --valroot dataset/val --cuda
```
### Demo

+ Before recognizing optical character, an image annotated by its text bounding box must be specified. Put bounding box txt file inside boundingbox/ and demo image in data_test

+ Download pretrained checkpoint with 100 epoch [here](https://drive.google.com/drive/folders/17KPbFetUaPWvcG0xJUNoDMXJC7WSeqAQ?usp=sharing) and put checkpoint file under expr/ folder

Some example:


#### Demo image: <img src="/crnn-ocr/version2/data_test/X51009568881.jpg" width=320 height=480 />
#### Results:
```
thank you ! & please come egain !
gst summary
0.59
rounding adjj
0.00
sr sr
1.00 5.00 sr
qty descipton n
cds & noan gst tmara)
0.00
09 3w aumnuroo 5do 5sr
59 tvoe 2 tota:
goodsod arendu runabnl3 reundaretangu!
10.40
total inclusive gst:
cashiey cash1-
discount:
10.400
price total(rm :
0.59
m# : c2 - 0
10.40
56100 cheras, kuala lumpur.
total
5 pvc wallplug
10.40
9.81
10.40
+603-9130 2672
9.81
gst raeg : 001125220352
no 57, jalan kanis 7, taman segar,
21/09/2017 10:2037am
hon hwa hardware trading
taxinvoice
ccb#: 87870
company reg.no. : 001055194x
((50pcs)
cash
```

#### Demo image: <img src="/crnn-ocr/version2/data_test/X510056849111.jpg" width=320 height=480 />

#### Result:
```
rounding
gst ta amt
cash
rm0.42
gst 6%
c01
total
tel : 03-4043 7678
returnable
3180203
:::****0*:*****.**"
-0.02
210038-k
co reg no
tax invoice
s
goods sold are not
000517095424
western eastern
gst id
stationery sdn. bhd
1 no
kl 001 035857
rm7.40
reg 26-02-2018 14:27
42-46, jln sultan azlan
rm7.42
rm7.00
rm7.42
tax invoice of gst
clr p.s a4/a3
shah 51200 kuala lumpur
```
# 3. KV

# 4. Explainer
