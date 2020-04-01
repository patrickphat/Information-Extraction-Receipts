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

## Modeling
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
## 3.1. Data Preparation 

**Load bbox coordinates and text lines in the bbox**

Use the `coords_labels_loader` [here](https://github.com/patrickphatnguyen/Optical-Character-Recognition-KV/blob/master/helpers/coords_labels_loader.py) to load coords and text labels from `.txt` ground-truth file on SROIE19

```python
from helpers import coords_labels_loader
coords_per_samples = []
labels_per_samples = []

for filename in text_filename:
  list_coords, list_labels = coords_labels_loader.load_from_file("task1/" + filename)
  coords_per_samples.append(list_coords)
  labels_per_samples.append(list_labels)
```

**From coordinates create graph (adjacency matrix)**

The input is list of all coordinates for bbox. Each coord is a dictionary having 4 keys: `x_min`,`y_min`,`x_max` and `y_max`.

```python
from helpers import GraphBuilder

adjacency_tensors = []
for coords in coords_per_sample):
  adjacency_tensor = build_graph_from_coords(coords)
  adjacency_tensors.append(adjacency_tensor)
```

**Sentence feature extraction with pretrained RoBerta**

```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
sentences_embeddings = model.encode(["Product Price",
                                     "Cost",
                                     "Money"])
```


## 3.2. Download and load prepared data
Get prepared data here
- [List of adjacency tensors](https://drive.google.com/file/d/1--E7mci8JFHBNRqVJWQY_5X9jwuIuET5/view?usp=sharing) Type: list. Shape m x [N_i,N_i,5]
- [Node features](https://drive.google.com/file/d/10aTONod0P8SEldh63MotAoSMwNPD-821/view?usp=sharing) Type: list. Shape mx[Ni,784]
- [Node labels](https://drive.google.com/open?id=1-3Do0qpiR1oMVLE60dNEL0AdqFUewqy7) Type: list. shape mx[N_i]

Notation:
- m: number of datapoints
- N_i: Number of nodes of i-th sample i

```python
# Module to load pkl file
from helpers.FilePickling import pkl_load

all_adjacency_tensors = pkl_load("all_adjacency_tensors.pkl")
all_node_features = pkl_load("all_node_features.pkl")
all_node_labels = pkl_load("all_node_labels.pkl")
```

## 3.3 Train and run demo

Access all necessary data [here](https://drive.google.com/open?id=1v3-ybmhEHxuG44wyskBRw2YJHGifxvh4)

Notebooks for training [here](https://colab.research.google.com/drive/1N9UTbUglxISw2XIePPRXQUohLAsxh47V)

Explanation of notebook:

+ Rebuild coor: Load the coordinates to build the adjacency matrix.

+ Rebuild nodes: Load the corresponding text containing each bbox(considered as one node)

+ Rebuild graph: Build the adjacency matrix from loading coordinates

+ Process text, Process all label, Process all labels_per_samples: Preprocess text of each bbox by using token to represent special character(using regular expression)

+ BOW, Extracted feature: using bag of words to build feature vector of each node(limited to 768 words)

+ Load and train: Load all above preprocessed data for processing in KV

+ Padding, Load padded data: Padding to feature, node labels and adjacency matrix with respect to the maximum number of nodes(160 in this case)

+ Demo: Run demo from pretrained model which is saved in GCN_final.pt file
# 4. Explainer
```python
from explain import explain
import matplotlib.pyplot as plt 

# Load pretrained GCN
GCN = torch.load("GCN_final.pt")

# Parameters
IDX = 1
IMG_PATHS = img_paths_test
NODE = 1
LR = 1e-1
PRINT_EVERY = 50
EPOCHS = 300
THRESH = 0.1
BASE_LINE_THICKNESS = 2
BASE_BOX_THICKNESS = 2

# Explain GCN result 
img = explain(model = GCN,
            A_s = A_s_test,
            V_s = V_s_test,
            IMG_PATHS = img_paths_test ,
            IDX = IDX,
            NODE = NODE,
            LR = LR,
            PRINT_EVERY = PRINT_EVERY,
            EPOCHS = EPOCHS,
            THRESH = THRESH,
            BASE_LINE_THICKNESS = BASE_LINE_THICKNESS,
            BASE_BOX_THICKNESS = BASE_LINE_THICKNESS)
            
# Show the explaining image            
plt.imshow(img)
```
In which:
- `A_s`is a tensor of adjacency tensors shape (m,N,N,L)
- `V_s`is a tensor of vector features shape (m,N,F)
- `IDX` is the index of image in the dataset
- `NODE` is the index of node of that given image
- `img_paths_test` is a list of path to load images (accordingly to A_s and V_s)
- `EPOCHS` is number of training epochs for explainer
- `LR` is number of learning rate for explainer
- if a probability of an edge >`THRESH`, that thresh is considered to be important
```
