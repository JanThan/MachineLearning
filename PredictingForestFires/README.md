# About this dataset
Problem is to try to predict the burned area of forest firest in the northeast region of Portugal. 
Data was found [here](https://archive.ics.uci.edu/ml/datasets/forest+fires)  

## Citation Request: 
This dataset is public available for research.  
The details are described in [Cortez and Morais, 2007].  
Please include this citation if you plan to use this database:  
P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data.  
In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. 
Available at: http://www.dsi.uminho.pt/~pcortez/fires.pdf 

## Attribute Information
For more information, read [Cortez and Morais, 2007]. 
1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 
2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 
3. month - month of the year: 'jan' to 'dec' 
4. day - day of the week: 'mon' to 'sun' 
5. FFMC - FFMC index from the FWI system: 18.7 to 96.20 
6. DMC - DMC index from the FWI system: 1.1 to 291.3 
7. DC - DC index from the FWI system: 7.9 to 860.6 
8. ISI - ISI index from the FWI system: 0.0 to 56.10 
9. temp - temperature in Celsius degrees: 2.2 to 33.30 
10. RH - relative humidity in %: 15.0 to 100 
11. wind - wind speed in km/h: 0.40 to 9.40 
12. rain - outside rain in mm/m2 : 0.0 to 6.4 
13. area - the burned area of the forest (in ha): 0.00 to 1090.84 
(this output variable is very skewed towards 0.0, thus it may make 
sense to model with the logarithm transform).

## Approach 
Due to the very skewed nature of the 'area' attribute, I made this into a logistic regression task in attempt to class whether or whether not a forest fire would occur.  
The workflow for this problem was as follows:
- load dataset
- visualise dataset
- perform some data preprocessing
- fit simple logistic regression model to training data
- predict for test inputs and compare results against test output    
- I also attempted to use feature importance to optimise the dataset, however found it to not be particularly impactful. 
This is easier recognised within the [source code.](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/forestFires2.py)  
The various outputs from this workflow can be found below... 

## Variables Summary
The Rain variable was found to be incredibly skewed as can be seen and so was removed from the dataset. 
### Distribution of inputs
![X](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/X.png)
![Y](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/Y.png)
![FFMC](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/FFMC.png)
![DMC](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/DMC.png)
![DC](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/DC.png)
![ISI](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/ISI.png)
![Temp](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/temp.png)
![RH](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/RH.png)
![Wind](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/wind.png)
![Rain](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/rain.png)
### Distribution of Compressive Strength of Concrete
![Area](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/area.png)

### Feature Importances
![Feature Importances](https://github.com/JanThan/LearningML/blob/master/PredictingForestFires/images/Feature%20Importances.png)
Due to the month data not classing very highly in feature importance, I removed them to see what effect it would have on the logistic regression model. It was found that it had very minimal impact.  
After further learning about other optimisation methods, I plan to return to this project to re-attampt its optimisation as there must be further optimisations I can make to improve its current state. 

## Logistic Regression Model Summary
### Before with months data  
Accuracy Score : 0.59  
Precision Score : 0.59  
Recall Score : 1.0  
F1 Score : 0.74  
### After removing months data  
Accuracy Score : 0.56  
Precision Score : 0.63  
Recall Score : 0.59  
F1 Score : 0.61  
