# CS598-Deep-Learning-in-Healthcare
This repository is to store all information and code bases that used for the class 

## Week 1 - Experience with PyTorch

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

### Week 2

### Week 7 - RNN Model

Supporting Package:

**Pickling** - https://pythonprogramming.net/pickle-data-analysis-python-pandas-tutorial/

Python has a module called Pickle, which will convert your object to a byte stream, or the reverse with unpickling. What this lets us do is save any Python object. That machine learning classifier? Yep. Dictionary? Yessir. Dataframe? Yep! Now, it just so happens that Pandas has pickles handled in its IO module, but you really should know how to do it with and without Pandas, so let's do that!

.pkl : File extension for "Pickle file" in Python

''' python
# Common pickle file extensions
'model.pkl'      # Most common
'data.pickle'    # Alternative extension
'state.pickled'  # Less common variant
'''

```python
# rb is "read binary" which required by pickle 
pids = pickle.load(open(os.path.join(DATA_PATH,'train/pids.pkl'), 'rb'))
```
**Pytorch**
[Pytorch Tutorial](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

**Neutral Model**

''' python
import torch
from torch import nn  #'''

**1. EmBedding**

'''

# Basic usage
embedding_layer = nn.Embedding(
    num_embeddings=10,  # vocabulary size
    embedding_dim=5     # dimension of embedding vectors
)

# Example
input_indices = torch.tensor([1, 4, 3])
embedded = embedding_layer(input_indices)
'''

Common Parameter of Embedding
'''
embedding = nn.Embedding(
    num_embeddings=1000,  # size of vocabulary
    embedding_dim=100,    # size of each embedding vector
    padding_idx=None,     # if specified, index for padding token
    max_norm=None,        # max norm of embedding vectors
    norm_type=2.0,       # p of p-norm for normalization
    scale_grad_by_freq=False  # scale gradients by frequency
)
'''

Real World Example

'''python
# Simple word embedding model
vocab_size = 5000
embedding_dim = 300

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Convert word IDs to embeddings
word_ids = torch.tensor([1, 2, 3])
word_vectors = embedding_layer(word_ids)
'''
**2. Loss and Optimized**

'''python

'''
**Seg2Seq**

### Week 8: Xray data and CNN Model to classify Pneumonia
https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5 

### AUTO ENCODER:

Autoencoders, a group of architectures used for encoding compact representations of model inputs and then reconstructing them.

When Running any Jupyter Note do


'''python -m venv venv'''



