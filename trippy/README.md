## Introduction

This is the code repository for exercise 4 of the Spoken Dialogue System Course at Heinrich-Heine University. The task is to complete the implementation in modeling_dst.py according to the instruction in the exercise sheet.
The dataset used is the simplemultiwoz21 in unified format, provided within the ConvLab3 toolkit.

TripPy is an approach to dialogue state tracking (DST) that makes use of various copy mechanisms to fill slots with values. Our model has no need to maintain a list of candidate values. Instead, all values are extracted from the dialog context on-the-fly.
A slot is filled by one of three copy mechanisms:
1. Span prediction may extract values directly from the user input;
2. a value may be copied from a system inform memory that keeps track of the system’s inform operations;
3. a value may be copied over from a different slot that is already contained in the dialog state to resolve coreferences within and across domains.
Our approach combines the advantages of span-based slot filling methods with memory methods to avoid the use of value picklists altogether. We argue that our strategy simplifies the DST task while at the same time achieving state of the art performance on various popular evaluation sets including MultiWOZ 2.1.

## How to run

One example script are provided for how to use TripPy. `DO.simple_mini` will train and evaluate a model with bert_mini. 


## Datasets


The ```--task_name``` is
- 'unified', for ConvLab-3's unified data format

With the provided run script and correct implementation, you should expect the average JGA of ~39%

## ConvLab-3

If you want to train your own TripPy model for ConvLab-3 from scratch, you can do so by using this code, setting ```--task_name='unified'```. The ```--data_dir``` parameter will be ignored in that case. Pick the file for ```--dataset_config``` according to the dataset you want to train for. For simplemultiwoz21, this would be 'dataset_config/unified_simplemultiwoz21.json'.

## Requirements

- torch (tested: 1.8.0)
- transformers (tested: 4.18.0)
- tensorboardX (tested: 2.1)

## Citation

This work is published as [TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://www.aclweb.org/anthology/2020.sigdial-1.4/)

If you use TripPy in your own work, please cite our work as follows:

```
@inproceedings{heck2020trippy,
    title = "{T}rip{P}y: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking",
    author = "Heck, Michael and van Niekerk, Carel and Lubis, Nurul and Geishauser, Christian and
              Lin, Hsien-Chin and Moresi, Marco and Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 21st Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    pages = "35--44",
