# Temporal Sentence Grounding with Relevance Feedback in Videos
This is a repository contains the implementation of our NeurIPS'24 paper "Temporal Sentence Grounding with Relevance Feedback in Videos" 
![overview](/figures/framework-RaTSG.jpg)


## Table of Contents
* [Environments](#environments)
* [Datasets](#datasets)
* [Preparation](#preparation)
* [Training](#training)
* [Evaluation](#evaluation)

## Environments
- **Ubuntu** 20.04
- **CUDA** 11.7
- **Python** 3.7

Install other required packages by
```sh
pip install -r requirements.txt
```

## Datasets
This paper has reconstructed the validation and test sets of two widely used datasets in the TSG domain: Charades-STA and ActivityNet Captions, to construct a testing environment for TSG-RF task., i.e., Charades-STA-RF, ActivityNet Captions-RF. The reconstructed dataset is located in the `./data/dataset` directory.

## Preparation
The details about how to prepare the `Charades-STA`, `ActivityNet Captions` features are followed previous work: [VSLNet Datasets Preparation](https://github.com/26hzhang/VSLNet/tree/master/prepare). Alternatively, you can download the prepared visual features from  [Mega](https://mega.nz/folder/gv93jDSI#U9Qf1ZuKdP8cIJj5sdK0bw), and place them to the `./data/features/` directory.
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to 
`./data/features/` directory.

## Training
**Train** 
```shell script
# train RaTSG on Charades-STA-RF dataset
bash charades_RF_train.sh
# train RaTSG on ActivityNet Captions-RF dataset
bash activitynet_RF_train.sh
```

## Evaluation
Run the following script to test on the trained models:
**Test** 
```shell script
# test RaTSG on Charades-STA-RF dataset
bash charades_RF_test.sh
# test RaTSG on ActivityNet Captions-RF dataset
bash activitynet_RF_test.sh
```

We release several pretrained checkpoints, please download and put them into `./ckpt/`
- RaTSG on Charades-STA-RF: [RaTSG_charades_RF_i3d_128](https://drive.google.com/drive/folders/1TQyojFEEhXsDg6GSChfGrmCcoKesigI5?usp=sharing)
- RaTSG on Activitynet Captions-RF: [RaTSG_activitynet_RF_i3d_128](https://drive.google.com/drive/folders/1TQyojFEEhXsDg6GSChfGrmCcoKesigI5?usp=sharing)


