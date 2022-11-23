SATRE with Data-efficiency and Computational Efficiency
==========

This code introduces Self-attention over Tree for Relation Extraction (SATRE) for the large scale sentence-level relation extraction task (TACRED).


See below for an overview of the model architecture:

![SATRE](fig/satre.png"SATRE")

  

## Requirements

Our model was trained  on two Nvidia GTX 1080Ti graphic cards.

- Python 3 (tested on 3.7.6)

- Pytorch (tested on 1.2.0)
- CUDA （tested on 10.2.89）
- tqdm

- unzip, wget (for downloading only)



## Preparation

 The code requires that you have access to the TACRED dataset (LDC license required). Once you have the TACRED data, please put the JSON files under the directory `dataset/tacred`.

 First, download and unzip GloVe vectors:

```
chmod +x download.sh; ./download.sh
```

  

Then prepare vocabulary and initial word vectors with:

```
python3 prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

  

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

  

## Training

 To train the SATRE model, run:

```
bash train.sh 0 1
```

Model checkpoints and logs will be saved to `./saved_models/1`.

For details on the use of other parameters, please refer to `train.py`.

  

## Evaluation

Our trained model is saved under the dir saved_models/1. To run evaluation on the test set, run:

```
bash  eval.sh 0 1 test
```

  

## Related Repo

Codes are adapted from the repo of the EMNLP18 paper [Graph Convolution over Pruned Dependency Trees Improves Relation Extraction](https://nlp.stanford.edu/pubs/zhang2018graph.pdf) and the repo of the ACL19 paper [Attention Guided Graph Convolutional Networks for Relation Extraction](https://aclanthology.org/P19-1024/).

