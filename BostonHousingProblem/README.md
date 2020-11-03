See [BostonHousing.py](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/BostonHousing.py) to see my approach to this common problem.  

# Visualisation of dataset
As always, I first visualised the dataset to get a better understanding of it...  
A general description of the dataset (click image if it is unclear):    
![description](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/descr.png)
What the first 5 data points looks like:  
![head](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/head.PNG)  

## Heatmap plot
I used a heatmap of the dataset to quickly identify strong correlations between features:
![heat_map](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/heatmap.png)
Strong correlations can be found between the output MEDV (Median value of owner-occupied homes in $1000's) and the two features:   LSTAT (% lower status of the population) and RM (average number of rooms per dwelling). These were the two features I used in my model going forward.  
## MEDV and LSTAT correlation 
![MEDV_LSTAT](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/MEDV_LSTAT.png)
## MEDV and RM correlation 
![MEDV_RM](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/MEDV_RM.png)

# Evaluation 
After normalising to avoid bias between the two features used, I split the data into training and test sets and performed gradient descent to find the best linear regression model as can be seen in the code.  
I used a learning rate of 0.001 over 15000 iterations.  
## Loss over iterations
![graph of loss over iterations](https://github.com/JanThan/LearningML/blob/master/BostonHousingProblem/images/loss_graph.png)  
From the graph, it can be seen that I can get away with using only ~5000 iterations which would help reduce the time taken. It is not too much of an issue here due to not being that complex of a task.  
  
I used the RMSE (root mean squared error) method of evaluating my model:  
RMSE = 6.80  
