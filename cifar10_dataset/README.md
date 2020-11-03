Now that I understood the fundamentals, I thought it was needed for me to understand how to quickly build and deploy NN models.  
I was recommended the Keras python library, and applied it to the Cifar-10 dataset. 

# Summary of dataset
- 50k training examples
- 10k testing examples
- Images are 32x32 pixels
The dataset consists of 10 classes: [airplane, car, bird, cat, deer, dog, frog, horse, ship, truck]  
# Visualisation of dataset
![cifar10](https://github.com/JanThan/LearningML/blob/master/cifar10_dataset/cifar10_plot.png)
# Evaluation of model
I was able to achieve an accuracy of ~80% over 100 epochs with a learning rate of 0.001. 
![training](https://github.com/JanThan/LearningML/blob/master/cifar10_dataset/cifarNN.py_plot.png)  
I used a VGG-based architecture with some drop out regularisation, batch normalisation and weight decay.  
For script see [Link](https://github.com/JanThan/LearningML/blob/master/cifar10_dataset/cifarNN.py)
