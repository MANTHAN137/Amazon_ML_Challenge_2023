import pandas as pd
import numpy as np
data=pd.read_csv('data.csv')
data.reindex()
data.head(5)
data.dropna(subset=['TITLE'],inplace=True)
data.dropna()
data['TITLE'].isnull().sum()
data.fillna(" ",inplace=True)
data['TAGS']=data['TITLE']+" "+data['BULLET_POINTS']+" "+data['DESCRIPTION']

data['TAGS'] = data['TAGS'].astype(str)

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

stemmer = PorterStemmer()
data['TAGS']=data['TAGS'].apply(lambda x: x.lower())

dirt=[']','_','[','(',')',':',';',"''",'{','}','<','>','/','\\','|','\'','@','#','$','%','^','&','*','+','=','~','`','1','2','3','4','5','6','7','8','9','0','-']
def dirtRemover(text):
    text=text.replace('[^\w\s]','').replace('\s\s+', ' ')
    text=re.sub(r'-?\d+\.\d+', '', text)
    text=re.sub('-?\d+',' ',text)
    text=re.sub(r'[^\w\s]','',text).replace("_","")
    
    # # POS tagging
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    filtered_words = []
    for word, pos_tag in pos_tags:
        if pos_tag in ['NN', 'NNS']:
            filtered_words.append(word.lower())

    # Remove stop words and dirt words
    filtered_words = [word for word in filtered_words if word not in stop_words]
    filtered_words = [word for word in filtered_words if word not in dirt]

    # Stemming
    stemmed_tags = []
    for tag in filtered_words:
        if(len(tag)>3):
            stemmed_tag = stemmer.stem(tag)
            stemmed_tags.append(stemmed_tag)
    return stemmed_tags

data['TAGS']=data['TAGS'].apply(dirtRemover)
data=data.drop(columns=['TITLE','BULLET_POINTS','DESCRIPTION','PRODUCT_ID','Unnamed: 0'],axis=1)
data['TAGS'] = data['TAGS'].apply(lambda x: ' '.join(x))

data.head(25)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = data['TAGS']
y = data['PRODUCT_LENGTH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert tags to vectors using CountVectorizer
cv = CountVectorizer(max_features=50000, stop_words='english')
X_train_vectors = cv.fit_transform(X_train['TAGS']).toarray()
X_test_vectors = cv.transform(X_test['TAGS']).toarray()

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_vectors, y_train)

# Evaluate model on training set
y_train_pred = model.predict(X_train_vectors)
mse_train = mean_squared_error(y_train, y_train_pred)
print('Training MSE:', mse_train)

# Evaluate model on testing set
y_test_pred = model.predict(X_test_vectors)
mse_test = mean_squared_error(y_test, y_test_pred)
print('Testing MSE:', mse_test)