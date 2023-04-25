# Amazon_ML_Challenge_2023
Team Name       : 	Astra
## Final Rank          124

Group Members : 
Kartik Shetty[Leader],
Manthan Dhole,
Kashyap Chavhan,
Hrishikesh Bade,


## PROBLEM STATEMENT
Product length prediction

Max. score: 100

+ In this hackathon, the goal is to develop a machine learning model that can predict the length dimension of a product. Product length is crucial for packaging and storing products efficiently in the warehouse. Moreover, in many cases, it is an important attribute that customers use to assess the product size before purchasing. However, measuring the length of a product manually can be time-consuming and error-prone, especially for large catalogs with millions of products.

+ You will have access to the product title, description, bullet points, product type ID, and product length for 2.2 million products to train and test your submissions. Note that there is some noise in the data.

## Task

+ You are required to build a machine learning model that can predict product length from catalog metadata.

## Dataset description

The dataset folder contains the following files:

train.csv: 2249698 x 6
test.csv: 734736 x 5
sample_submission.csv: 734736 x 2
The columns provided in the dataset are as follows:

PRODUCT_ID: Represents a unique identification of a product
TITLE: Represents the title of the product
DESCRIPTION: Represents the description of the product
BULLET_POINTS: Represents the bullet points about the product
PRODUCT_TYPE_ID: Represents the product type
PRODUCT_LENGTH: Represents the length of the product
Evaluation metric

score = max( 0 , 100*(1-metrics.mean_absolute_percentage_error(actual,predicted)))

## Result submission guidelines

The index is "PRODUCT_ID" and the target is the "PRODUCT_LENGTH" column.
The submission file must be submitted in .csv format only.
The size of this submission file must be 734736 x 2.
Note: Ensure that your submission file contains the following:

Correct index values as per the test file
Correct names of columns as provided in the sample_submission.csv file

## Approach :
This is a Python code for a machine learning model that uses linear regression to predict the length of a product based on its tags. Here is an overview of the code:

+ The code imports necessary libraries such as pandas, numpy, re, nltk, and sklearn.
- The code loads a dataset from a CSV file using pandas' read_csv function.
+ The code drops any rows with missing values in the "TITLE" column using the dropna function.
+ The code fills any remaining missing values with empty strings using the fillna function.
+ The code combines the "TITLE", "BULLET_POINTS", and "DESCRIPTION" columns into a single "TAGS" column.
+ The code preprocesses the text in the "TAGS" column by converting it to lowercase, removing punctuation and digits, removing stop words and "dirt" words (a custom list of characters), and stemming the remaining words.
+ The code drops unnecessary columns from the dataset.
+ The code splits the dataset into training and testing sets using train_test_split function from sklearn.
+ The code converts the text in the "TAGS" column to a matrix of word counts using CountVectorizer function from sklearn.
+ The code trains a linear regression model on the training set using the LinearRegression function from sklearn.
+ The code evaluates the model's performance on the training set and testing set using mean squared error metric.
+ Finally, the code prints the training and testing mean squared error.
+ In summary, this code preprocesses text data, converts text to numerical vectors, trains a linear regression model, and evaluates its performance using mean squared error.

# Project Final Result Details
<img width="410" alt="image" src="https://user-images.githubusercontent.com/90518833/234085796-81c9b08a-8086-4026-bc70-db40d6830f05.png">
<img width="675" alt="image" src="https://user-images.githubusercontent.com/90518833/234086089-13dea69d-a42b-4911-b6a1-57b675765c0d.png">
<img width="690" alt="image" src="https://user-images.githubusercontent.com/90518833/234086178-8d2ac519-6d49-4b2f-955a-5382c867d5fd.png">
<img width="765" alt="image" src="https://user-images.githubusercontent.com/90518833/234191762-c581444c-21c1-44c4-a475-0ac4e45df23e.png">
