# Assignment 2

In this assignment you are asked to:

1. Implement a convolutional neural network to classify images of 'Cats of the Wild';
2. Implement a fully connected feed forward neural network to classify images from the 'Cats of the Wild' dataset.

Both requests are very similar to what we have seen during the labs. However, you are required to follow **exactly** the assignment's specifications.

Once completed, please submit your solution on the iCorsi platform following the instructions below. 


## Tasks

### T1. Image Classification with Fully Connected Feed Forward Neural Networks (FFNN)

In this task, we will try and build a classifier for the provided dataset. This task, we will use a classic Feed Forward Neural Network.

1. Download and load the dataset using the following link 'https://drive.switch.ch/index.php/s/XSnhQDNar7y46oQ'. The dataset consist of 7 classes with a folder for each class images. The classes are 'CHEETAH' ,'OCELOT', 'SNOW LEOPARD', 'CARACAL', 'LIONS', 'PUMA', 'TIGER'. Check `src/utils.py` to find the ready and implemented function to load the dataset. 
2. Preprocess the data:
    - Normalize each pixel of each channel so that the range is [0, 1];
3. Flatten the images into 1D vectors. You can achieve that by using [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape) or by prepending a [Flatten layer](https://keras.io/api/layers/reshaping_layers/flatten/) to your architecture; if you follow this approach this layer will not count for the rules at point 3.
4. Build a Feed Forward Neural Network of your choice, following these constraints:
    - Use only Dense layers.
    - Use no more than 3 layers, considering also the output one.
    - Use ReLU activation for all layers other than the output one.
5. Draw a plot with epochs on the x-axis and with two graphs: the train accuracy and the validation accuracy (remember to add a legend to distinguish the two graphs!).
6. Assess and comment on the performances of the network on the test set loaded in point 1, and provide an estimate of the classification accuracy that you expect on new and unseen images. 
7. **Bonus** (Optional) Train your architecture of the FFNN of choice (you are allowed to change the input layer dimensionality!) following the same procedure as above, but, instead of the flattened images, use any feature of your choice as input. You can think of these extracted features as a conceptual equivalent of the Polynomial Features you saw in Regression problems, where the input data were 1D vectors. Remember that images are just 3D tensors (HxWxC) where the first two dimensions are the Height and Width of the image and the last dimension represents the channels (usually 3 for RGB images, one for red, one for green and one for blue). You can compute functions of these data as you would for any multi-dimensional array. A few examples of features that can be extracted from images are:
    - Mean and variance over the whole image.
    - Mean and variance for each channel.
    - Max and min values over the whole image.
    - Max and min values for each channel.
    - Ratios between statistics of different channels (e.g. Max Red / Max Blue)
    - [Image Histogram](https://en.wikipedia.org/wiki/Image_histogram) (Can be compute directly on [TF Tensors](https://www.tensorflow.org/api_docs/python/tf/histogram_fixed_width) or by temporarely converting to numpy arrays and using [np.histogram](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html))
But you can use anything that you think may carry useful information to classify an image.

**N.B.** If you carry out point 7 also consider the obtained model and results in the discussion of point 6.


### T2. Follow our recipe

Implement a multi-class classifier (CNN model) to identify the class of the images: 'CHEETAH' ,'OCELOT', 'SNOW LEOPARD', 'CARACAL', 'LIONS', 'PUMA', 'TIGER'.

1. Follow steps 1 and 2 from T1 to prepare the data.
2. Build a CNN of your choice, following these constraints: 
 - use 3 convolutional layers
 - use 3 pooling layers
 - use 3 dense layers (output layer included).
3. Train and validate your model. Choose the right optimizer and loss function.
4. Follow steps 5 and 6 of T1 to assess performance.
5. Qualitatively and **statistically** compare the results obtained in T1 with the ones obtained in T2. Explain what you think the motivations for the difference in performance may be.
6. Apply image manimpulation/augmentation techniques in order to improve the performance of your models. Evaluate the performance of the model using the new images and compare the results with the previous evaluation performed in part 3. Provide your observations and insights.
7. **Bonus** (Optional) Tune the model hyperparameters with a **grid search** to improve the performances (if feasible).
    - Perform a grid search on the chosen ranges based on hold-out cross-validation in the training set and identify the most promising hyper-parameter setup.
    - Compare the accuracy on the test set achieved by the most promising configuration with that of the model obtained in point 4. Are the accuracy levels **statistically** different?


## Instructions

### Tools

Your solution must be entirely coded in **Python 3** ([not Python 2](https://python3statement.org/)).
We recommend to use Keras from TensorFlow2 that we seen in the labs, so that you can reuse the code in there as reference. 

All the required tasks can be completed using Keras. On the [documentation page](https://www.tensorflow.org/api_docs/python/tf/keras/) there is a useful search field that allows you to smoothly find what you are looking for. 
You can develop your code in Colab, where you have access to a GPU, or you can install the libraries on your machine and develop locally.


### Submission

In order to complete the assignment, you must submit a zip file named `as2_surname_name.zip` on the iCorsi platform containing: 

1. A report in `.pdf` format containing the plots and comments of the two tasks. You can use the `.tex` source code provided in the repo (not mandatory).
2. The best models you find for both the tasks (one for the first task, one or two for the second task, in case you completed the bonus point). By default, the keras function to save the model outputs a folder with several files inside. If you prefer a more compact solution, just append `.h5` at the end of the name you use to save the model to end up with a single file.
3. A working example for T1 and T2 `tasks.ipynb` that loads the dataset, preprocesses the data, loads the trained model from file and evaluates the accuracy. Note that the notebook should contain only 5 cells and each cell shou be run independatly from the other cells.
 - 1 cell for loading the data (already given)
 - 1 cell for task 1
 - 1 cell for task 1 bonus (if any)
 - 1 cell for task 2
 - 1 task for task 2 bonus (if any)
4. A folder `src` with all the source code you used to build, train, and evaluate your models.

The zip file should eventually looks like as follows

```
as2_surname_name/
    report_surname_name.pdf
    deliverable/
        example.ipynb #your solution
        # your saved files (i.e ending with h5)
    src/
        utils.py
        ...
```


### Evaluation criteria

You will get a positive evaluation if:

- your code runs out of the box (i.e., without needing to change your code to evaluate the assignment);
- your code is properly commented;
- the performance assessment is conducted appropriately;

You will get a negative evaluation if: 

- we realize that you copied your solution;
- your code requires us to edit things manually in order to work;
- you did not follow our detailed instructions in tasks T1 and T2.

Bonus parts are optional and are not required to achieve the maximum grade, however they can grant you extra points.

