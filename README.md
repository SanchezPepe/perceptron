
## From Biology to AI: The Perceptron

This is the Python Jupyter Notebook for the [Medium article](https://towardsdatascience.com/from-biology-to-ai-the-perceptron-81abfdc788bf) on how to implement the Perceptron algorithm in Python.

It has been a long standing task to create machines that can act and reason in a similar fashion as humans do. And while there has been lots of progress in artificial intelligence (AI) and machine learning in recent years some of the groundwork has already been laid out more than 60 years ago. In this Jupyter notebook we will explore how to implement one of the earliest attempts of creating machine intelligence: The Perceptron. This algorithm is one of the past milestones on the way to the modern deep learning algorithms.

We'll start by importing some libraries for generating a dataset and visualizing our results. The Perceptron implementation will be in NumPy.


```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
```

So now that we imported everything we need for our implementation of the Perceptron algorithm we should generate some data to work with. Luckily scikit learn provides a function called make_blobs that does exactly this for us.


```python
# Generate dataset
X, Y = make_blobs(n_features=2, centers=2, n_samples=1000, random_state=18)

# Visualize dataset
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], c=Y)
ax.set_title('ground truth', fontsize=20)
plt.show()
```


![png](images/perceptron_4_0.png)


Ok by looking at the above plot we can see that both clusters can be linearly separated. So lets start coding out the Perceptron.

First we add a bias (ones) term to the inputs.


```python
# Add a bias to the X1 vector
X_bias = np.ones([X.shape[0], 3])
X_bias[:, 1:3] = X
```

Now we initialize the weights with zeros.


```python
# Initialize weights with zeros
w = np.zeros([3, 1])
```

Finally, before we train the Perceptron let's define some functions that will make our life easier.

1) An activation function that returns either 1 or 0


```python
def activation(x):
    return 1 if x >= 1 else 0
```

2) A function to calculate the unit vector of our weights vector


```python
def calc_unit_vector(x):
    return x.transpose() / np.sqrt(x.transpose().dot(x))
```

3) A function that returns values that lay on the hyperplane


```python
def calc_hyperplane(X, w):
    return np.ravel([-(w[0] + x * w[1]) / w[2] for x in X])
```

Last thing to do is writing the Perceptron algorithm and running it.


```python
# Apply Perceptron learning rule
for _ in range(10):
    for i in range(X_bias.shape[0]):
        y = activation(w.transpose().dot(X_bias[i, :]))

        # Update weights
        w = w + ((Y[i] - y) * X_bias[i, :]).reshape(w.shape[0], 1)
```

Determine the class of each data point according to the weight vector our perceptron found


```python
# Calculate the class of the data points with the weight vector
result = [w.transpose().dot(x) for x in X_bias]
result_class = [activation(w.transpose().dot(x)) for x in X_bias]
```

Calculate the unit vector of the weight vector


```python
# Calculate unit vector
w = calc_unit_vector(w).transpose()
```

Plot the results


```python
# Visualize results
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(X[:, 0], X[:, 1], c=Y)
ax[0].set_title('ground truth', fontsize=20)

ax[1].scatter(X[:, 0], X[:, 1], c=result_class)
ax[1].plot([-20, 20], calc_hyperplane([-20, 20], w), lw=3, c='red')
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
ax[1].set_yticks([])
ax[1].set_title('Perceptron classification with hyperplane', fontsize=20)

plt.show()
```


![png](images/perceptron_22_0.png)
