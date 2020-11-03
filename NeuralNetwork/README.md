This is a documentation of my progress in further solidifying my knowledge on Neural Networks following on from Andrew Ng's coursera course.  
## Neural Network with no Hidden Layer
See [Link](https://github.com/JanThan/LearningML/blob/master/NeuralNetwork/NN_basic.py) for implementation of NN without hidden layers. As this was my first effort, I thought it best to start an implementation without a hidden layer. As this would only be able to use linear boundaries I created this very simple dataset:  
![LinearDataSet](https://github.com/JanThan/LearningML/blob/master/NeuralNetwork/images/LinearDataset.PNG)  
From this dataset, it is obvious that there is a positive correlation between obesity and being diabetic.   
I trained it with a learning rate of 0.05 over 20000 iterations and was able to get a error of 0.0039.  
I then tested this model on two test data points: [1 0 0] and [0 1 0].  
The model correctly gave [1 0 0] a confidence of 1.78% and [0 1 0] a confidence of 99.53%

## Neural Network (One Hidden Layer)
See [Link](https://github.com/JanThan/LearningML/blob/master/NeuralNetwork/NN_withHiddenLayer.py) for my first implementation of a Neural Network with a single hidden layer from scratch.  
I created a non-linear dataset to test this on using 'make moons' from the Scikit library:  
![dataset](https://github.com/JanThan/LearningML/blob/master/NeuralNetwork/images/Data.png)  
I was able to reach a loss of 0.06 using a learning rate of 0.01 over just 200 iterations.  
![lossgraph](https://github.com/JanThan/LearningML/blob/master/NeuralNetwork/images/NN_with_hidden_cost.png)
## Neural Network (One Hidden Layer) - slightly neater implementation
See [Link](https://github.com/JanThan/LearningML/blob/master/NeuralNetwork/NN_4_alt.py) for a slightly altered implementation of the previous. It has the advantage of being more flexible to use for different purposes and overall easier to read and understand, whilst being similar in performance to previous.
