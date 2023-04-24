# Amazon_ML_Challenge_2023
Team Name       : 	Astra

Group Members : 
Kartik Shetty[Leader],
Manthan Dhole,
Kashyap Chavhan,
Hrishikesh Bade,

Approach Method:
This is a Python code for a machine learning model that uses linear regression to predict the length of a product based on its tags. Here is an overview of the code:

The code imports necessary libraries such as pandas, numpy, re, nltk, and sklearn.
The code loads a dataset from a CSV file using pandas' read_csv function.
The code drops any rows with missing values in the "TITLE" column using the dropna function.
The code fills any remaining missing values with empty strings using the fillna function.
The code combines the "TITLE", "BULLET_POINTS", and "DESCRIPTION" columns into a single "TAGS" column.
The code preprocesses the text in the "TAGS" column by converting it to lowercase, removing punctuation and digits, removing stop words and "dirt" words (a custom list of characters), and stemming the remaining words.
The code drops unnecessary columns from the dataset.
The code splits the dataset into training and testing sets using train_test_split function from sklearn.
The code converts the text in the "TAGS" column to a matrix of word counts using CountVectorizer function from sklearn.
The code trains a linear regression model on the training set using the LinearRegression function from sklearn.
The code evaluates the model's performance on the training set and testing set using mean squared error metric.
Finally, the code prints the training and testing mean squared error.
In summary, this code preprocesses text data, converts text to numerical vectors, trains a linear regression model, and evaluates its performance using mean squared error.
