Dependencies:
sys
os
pandas
numpy
matplotlib
seaborn
calendar
pickle
scikit-learn
lightgbm


1. Install the above dependencies
2. Open and run eda.py. Each plot generated will get saved in the Plots directory.
3. Open and run train.py. This will output Validation Results.csv which will contain the model evaluation of each model. 
Also, it will save models casual_model.sav and registered_model.sav
4. Open and run prediction.py. This will use the models saved in the previous step to predict on test_data.csv and generate y_pred.csv
5. Open evaluate_predictions.py. This will use y_pred.csv generated in the previous step and y_true.csv to evaluate the final predictions.
The MAD and MAPE will be printed.


About the files:

1. process_data.py

	- A utility file which has functions used by both train.py and prediction.py
	- get_features(): This function is called to preprocess the data. It in turns calls the following functions:
		-get_peaks(): Creates 2 flag columns which are checked when there is a peak hour
		-convert_categorical(): It is called only if we are requesting for dummies. Might come in handy if LinearRegression needs to be tried in future
	- y_log(): Used to log-transform "casual" and "registered"
	-xySplit(): Split data into x and y. Special handling for train as the y variables in train need to be log-transformed

2. train.py
	
	- Python file to compare models and save the best model
	- load_data(): loads the train data
	- train_valid_split(): Splits the data into train and valid
	- init_models(): Initializes the models
	- compare_models(): Uses the models initialized by init_models() and then calls the following for each model:
		- evaluate_model(): Trains each model on train and evaluates on the holdout set and stores the result in data frame results
	-save_model(): Trains the best model on the whole data and saves the models along with the dataframe results

3. prediction.py
	
	- Python file to predict using the models saved in the previous step. Can predict for multiple days as well as single days
	- make_prediction_data(): Used to create test data. No need to execute this. 
	- predict(): Imports the models and returns y_pred

3. evaluate_predictions.py

	- Python file to print MAD and MAPE of test data
	- evaluate_predictions(): Returns MAD and MAPE

References:
[1] Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
