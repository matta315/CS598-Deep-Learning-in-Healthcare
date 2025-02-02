## 1. Tensors

Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on *GPUs* or other hardware accelerators. Tensors are also optimized for automatic differentiation (we’ll see more about that later in the Autograd section). 

pip install tourch

pip install numpy matplotlib
### 1.1 Shape:
shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

### 1.2 Operation of Tensor:

**Standard numpy-like indexing and slicing**

tensor = torch.arange(12).reshape(3, 4).float()
print(tensor)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[:, -1])

**Joining tensor**

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

**Arithmetic Operations**

(1) 
y1 = tensor @ tensor.T
print(y1)

y2 = tensor.matmul(tensor.T)
print(y2)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)

### 1.3 GPU

GPU (Graphics Processing Unit) is a specialized electronic circuit designed for parallel processing. Here's a clear breakdown: https://developer.mozilla.org/en-US/docs/Glossary/GPU

### 1.4 Implementing Sigmoid Function

**Implement the Sigmoid function on your own.**

$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$

Note that you should not use existing PyTorch implementation.

Hint: try `torch.exp()`.

def sigmoid(x):
    # your code here
    return 1/(1 + torch.exp(-x))
    raise NotImplementedError

**Implement a Softmax function on your own.**

softmax(𝐗)𝑖𝑗=exp(𝐗𝑖𝑗)∑𝑘exp(𝐗𝑖𝑘)

Note that you should not use existing PyTorch implementation.

Hint: try torch.exp() and torch.sum().

def softmax(X):
    # your code here
    return torch.exp(X)/torch.sum(torch.exp(X), dim=1, keepdim=True)
    raise NotImplementedError

## 2. Loss

When presented with some training data, our untrained network is likely not to give the correct answer. Loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. To calculate the loss we make a prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include `nn.MSELoss` (Mean Square Error) for regression tasks, and `nn.NLLLoss` (Negative Log Likelihood) for classification. `nn.CrossEntropyLoss` combines `nn.LogSoftmax` and `nn.NLLLoss`. `nn.BCELoss` is specially designed for binary classification.

## Mean Square Error

This is the most popular function. When our prediction for an example $i$ is $\hat{y}^{(i)}$ and the corresponding true label is ${y}^{(i)}$, the squared error is given by:

$$l^{(i)} = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

To measure the quality of a model on the entire dataset of $n$ examples, we simply average (or equivalently, sum) the losses on the training set.

$$L =\frac{1}{n}\sum_{i=1}^n l^{(i)}.$$

Let us see how to implement this.

## Cross Entropy Loss
Best use for binary classification

$$l^{(i)} = - \sum_{j=1}^q y_j^{(i)} \log \hat{y}_j^{(i)},$$

$$L =\frac{1}{n}\sum_{i=1}^n l^{(i)}.$$

where $\mathbf{y}^{(i)}$ is a one-hot vector of length $q$, the sum over all its coordinates $j$ vanishes for all but one term.

Note that you should not use existing PyTorch implementation.

Hint: try `torch.log()`.