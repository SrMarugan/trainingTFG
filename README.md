# trainingTFG
In this repository we are going to show how to train two types of models using tensorflow and keras. This repository is a complementary and an extra explanation of my TFG AIOT system based on Kendryte K210 SoCs

# Prepare the environment
First of all we install the anaconda environment from https://www.anaconda.com/
Then open Anaconda Navigator and launch Jupiter Notebook

![image](https://user-images.githubusercontent.com/115635629/219171276-c97e65b0-d4a7-4821-a322-5b56536ab012.png)

Now create a conda working environment in the cmd or terminal of your computer

```
conda create --name <Name>
```

![image](https://user-images.githubusercontent.com/115635629/219172275-e83fea9f-8ca9-4204-bc83-c58ae5119ec2.png)

Now with conda activate <Name> and conda deactivate you can change your working environment.
Why we do that? Because when we are training a model we need the dependencies of tensorflow and more dependencies with a specific version or with the same name that others. With the conda environment you protect yourself from other types of versions of dependencies that you may work with if you usually code in python.
  
To open the new environment we use:
  
```
conda activate <Name>
```
  
As you can see in the capture I create the environtment in the Power Shell of Windows 11, but I use the cmd to work in the environment. If you are using Windows 10 you have to use the cmd. In Linux you can do the same but using the terminal. In Mac the same that Linux.
  
Now install the dependencies of tensorflow using:
```
conda install tensorflow
```
  
And the tensorflow datasets using:
```
pip install tensorflow-datasets
```

To acces to this dependencies in the notebook we have to install ipywidgets using:
```
conda install -c conda-forge ipywidgets
```

Also I use some libraries to help me in the visualization of the MNIST results predictions, you can install using:
```
pip install matplotlib
pip install opencv-python
```
Remember you have to install in the conda environtment that you create (use before **conda activate <Name>**) and you have to activate everytime you reboot your computer or close the cmd program/terminal.

Now we're gonna create our own MNIST model using the datasets of tensorflow. I create it with two differents forms, using the tensorflow datasets and keras datasets. The difference? I don't know, but the tensorflows datasets needs more preprocessing before we train it. I load the two codes in the repository, but we have to see how it works.

Add libraries and dependencies:
```
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
```

It's very important the first two dependencies, without them we can't train our model. Remember if you don't have install you have to do it, how?, I explain that in the beginning.

# Preparing the dataset

First we prepare the MNIST dataset. The dataset that we use is the same to the both models that we train.

We load the data of the dataset and we reshape the images in a 32x32 format and transform the "matrix" in a float32 type and normalize (divide by 255) between 0 and 1. We do that with the training and the test images.

```
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
```
We use this libraries to train the models and load the MNIST images.

```
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()


x_train = np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0)
x_test = np.pad(x_test,((0,0),(2,2),(2,2)),'constant',constant_values=0)

x_train = x_train.astype('float32')

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0],32,32,1)
x_test = x_test.reshape(x_test.shape[0],32,32,1)
```

Now we have the MNIST images ready to train the models.

# Simple model
We design a simple MNIST model using two types of layers, the first is the Flatten neuronal layers. We use this layer to convert the multidimensional tensor into an unidimensional tensor. This neuronal layer is useful to convert images and other data with a similar structure in a type that can be proccessing by dense layers and other neuronals nets layers.

The next layer that we use is the Dense neuronal layers. We use this layer to make conections between neurons and put weights between that. This layer has an input size and an output size, and the layer before that has to have the same output size that the after input layer. But we don't have to put the output size in the layer, that is making automatically. We do that exactly the same with all the layers in the model. Then we have the different types of activation, we use the relu activation and for the last layer we use the softmax.

The relu activation gets the maximum value with this simple formula f(x) = max(0, x). The last layer use the softmax activation that is more complex than relu, it applies a probability distribution with that formula f(x_i) = e^(x_i) / Sum(e^(x_j)) we use it to have in the output a probability of the prediction. As we use a digit recognition only have 10 outputs (0-9, 10 digits).

```
model =  tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 1)),
    tf.keras.layers.Dense(124, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(124, activation=tf.nn.relu),
    tf.keras.layers.Dense(496, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])
```

Now we have the model design, this neuronal network is enough to detect digits, but is inefficient because it weigh a lot when it's not neccesary but it's very simple the layers we use and it's very fast the training.

# Complex model
We design a complex MNIST model adding two new differents layers. The first layer is the Conv2D layer that applies a convolutional filter. This layer is more complex and requires much more training time to determinate the weights and to predict the image, because the operation is a little more complex than a simple multiplication. 

The next layer is the AveragePooling2D that is a layer usually go after the Conv2D layer. It applies an average to a pixel moving window, the output is a reduced representation of the input that preserves the characteristics more important of the data.

```
model =  tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 6, kernel_size = (3,3), padding = 'valid', activation = tf.nn.relu, input_shape = (32,32,1)),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'),
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid', activation = tf.nn.relu),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 120, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 84, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)
])
```

# Final training
Once we have the struct of the two different structures of the models we proceed to train the models. You can see it in the entire code in TrainingFinalTFG, also you can see how to save it in a .h5 format and in a .tflite format. Last there are two more programs, one save the train images in a directory of our PC and the other is an inferer using the train images, but you can use other different images, so, you can't use the function evaluate unless you put correctly the results (in an array of numbers with the correct answer).
