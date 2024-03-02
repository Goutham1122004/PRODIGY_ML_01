1.Data Loading and Preprocessing:
  *Reads a CSV file named "data.csv" containing information about houses.
  *Splits the data into features (X) and target variable (y).
  *Scales the features using StandardScaler.

2.Train-Test Split:
  *Splits the data into training and testing sets using train_test_split function from scikit-learn.

3.Model Training:
  *Uses a linear regression model (LinearRegression) from scikit-learn to train on the training data.

4.Model Evaluation:
  *Calculates the R-squared score on both the training and testing sets to evaluate the model's performance.

5.Prediction Function:
  *Defines a function predict_price that takes the number of bedrooms, bathrooms, and square footage of living area as input, scales the input data using the same     scaler used for training data, predicts the price using the trained linear regression model, and returns the predicted price.

6.User Interaction:
  *Asks the user to input the number of bedrooms, bathrooms, and square footage of living area for a house.
  *Uses the predict_price function to predict the price of the house based on the user input.
  *Prints the predicted price.
  
To run this code, you need to ensure the following:

Create a CSV file named "data.csv" containing the relevant information about houses (e.g., bedrooms, bathrooms, square footage of living area, price).
Make sure the dataset directory is correctly specified in the code.
Customize the dataset by adjusting the column names and data accordingly in the CSV file.
To run the code, create a Python file named "main.py" and paste the provided code into it. Then, execute the "main.py" file using a Python interpreter. Make sure that the "data.csv" file is in the same directory as the "main.py" file. Follow the instructions provided by the program to input the relevant information about the house for which you want to predict the price.
