# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:12:01 2023

@author: sharma
"""

##Import All Relevant Packages

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traceback

import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
#from textblob import TextBlob
#from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')


#Import Data
try:    
    df = pd.read_excel(r'C:\Assignment\Movie Review.xlsx')
    print("\n Number of rows of data fetched is:", len(df))    
except Exception:
    traceback.print_exc()
#43,564

#Check df
df.head()

#Describe DF
desc = df.describe() 

##Review Duplicacy Seen
 

#Review count
df['SentimentValue'].value_counts()
# Positive    26550
# Negative    17014

#Check Data Duplicacy and Event Rate
df.duplicated().sum()
##Observation - 2,819 Data Duplicates present. Make it unique

#Subset Dataframe without Duplicates to be used for Model Training
df2 = df.drop_duplicates()
#207 Duplicates 
df2.count()
#Review count
df2['SentimentValue'].value_counts()
# Positive    24698
# Negative    16047


#Treatment of Reviews to create features

# The necessary steps include (but aren’t limited to) the following:

# Tokenizing sentences to break text down into sentences, words, or other units
# Removing stop words like “if,” “but,” “or,” and so on
# Normalizing words by condensing all forms of a word into a single form
# Vectorizing text by turning the text into a numerical representation for consumption by your classifier


#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')


#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Apply function on review column
df2['review_2']=df2['review'].apply(denoise_text)


#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
df2['review_2']=df2['review_2'].apply(remove_special_characters)


#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
df2['review_2']=df2['review_2'].apply(simple_stemmer)

#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
df2['review_2']=df2['review_2'].apply(remove_stopwords)


#TRAIN & TEST REVIEW
#split the dataset  
#train dataset
test = df2.sample(frac=0.2,random_state=1)
##Remove Validation set from main dataframe
train = pd.concat([df2, test, test]).drop_duplicates(keep=False).reset_index(drop=True)

test['SentimentValue'].value_counts()
# Positive    4912
# Negative    3237

train['SentimentValue'].value_counts()
# Positive    19784
# Negative    12810

norm_train_reviews=train.review_2
# norm_train_reviews[0]
norm_test_reviews=test.review_2


#BAG OF WORDS
#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
#transformed test reviews
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names


#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(norm_train_reviews)
#transformed test reviews
tv_test_reviews=tv.transform(norm_test_reviews)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)


#labeling the sentiment data
lb=LabelBinarizer()
#transformed sentiment data

sentiment_data=lb.fit_transform(df2['SentimentValue'])
print(sentiment_data.shape)

#Spliting the sentiment data
train_sentiments=lb.fit_transform(train.SentimentValue)
test_sentiments=lb.fit_transform(test.SentimentValue)
print(train_sentiments)
print(test_sentiments)


#Training Sentiment Model using Logistic Regression - Bags of Words & TF-IDF Attributes
#LR
# lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
lr=LogisticRegression(max_iter=500)

#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
print(lr_tfidf)

y_train_pred = lr_tfidf.predict(tv_train_reviews)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(lr_tfidf.score(tv_train_reviews, train_sentiments)))
#Accuracy of logistic regression classifier on train set: 1

##Predicting the test set results and calculating the accuracy
y_pred = lr_tfidf.predict(tv_test_reviews)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_tfidf.score(tv_test_reviews, test_sentiments)))
##Accuracy of logistic regression classifier on test set: 0.63

##Confusion Matrix
from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(test_sentiments, y_pred)
# print(confusion_matrix)
cm_train = confusion_matrix(train_sentiments, y_train_pred)
print(cm_train)


##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(test_sentiments, y_pred))
print(classification_report(train_sentiments, y_train_pred))


##ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_sentiments, lr_tfidf.predict(tv_test_reviews))
fpr, tpr, thresholds = roc_curve(test_sentiments, lr_tfidf.predict_proba(tv_test_reviews)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#Combine BOW & TF_IDF Sparse Matrix to include in Model Train
from scipy.sparse import hstack
train_2 = hstack((cv_train_reviews, tv_train_reviews))
test_2 = hstack((cv_test_reviews, tv_test_reviews))

#Fitting LR with both Bag of Words + TD_IDF 
lr_2=lr.fit(train_2,train_sentiments)
print(lr_bow)

y_train_pred = lr_2.predict(train_2)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(lr_2.score(train_2, train_sentiments)))
#Accuracy of logistic regression classifier on train set: 1

##Predicting the test set results and calculating the accuracy
y_pred = lr_2.predict(test_2)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_2.score(test_2, test_sentiments)))
##Accuracy of logistic regression classifier on test set: 0.63

##Confusion Matrix
from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(test_sentiments, y_pred)
# print(confusion_matrix)
cm_train = confusion_matrix(train_sentiments, y_train_pred)
print(cm_train)


##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(test_sentiments, y_pred))
print(classification_report(train_sentiments, y_train_pred))



#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 10)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(cv_train_reviews, train_sentiments)
  
# performing predictions on the test dataset
y_pred = clf.predict(cv_test_reviews)
y_train_pred = clf.predict(cv_train_reviews)
  

##Confusion Matrix
from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(test_sentiments, y_pred)
# print(confusion_matrix)
cm_train = confusion_matrix(train_sentiments, y_train_pred)
print(cm_train)

##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(test_sentiments, y_pred))
print(classification_report(train_sentiments, y_train_pred))


##ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_sentiments, clf.predict(cv_test_reviews))
fpr, tpr, thresholds = roc_curve(test_sentiments, clf.predict_proba(cv_test_reviews)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# using the feature importance variable
# import pandas as pd
# feature_imp = pd.Series(clf.feature_importances_, index = tv_train_reviews.columns).sort_values(ascending = False)
# feature_imp


##Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=2, random_state=0).fit(tv_train_reviews, train_sentiments)
# performing predictions on the test dataset
y_pred = clf.predict(tv_test_reviews)
y_train_pred = clf.predict(tv_train_reviews)
  

##Confusion Matrix
from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(test_sentiments, y_pred)
# print(confusion_matrix)
cm_train = confusion_matrix(train_sentiments, y_train_pred)
print(cm_train)

##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(test_sentiments, y_pred))
print(classification_report(train_sentiments, y_train_pred))


##ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_sentiments, clf.predict(tv_test_reviews))
fpr, tpr, thresholds = roc_curve(test_sentiments, clf.predict_proba(tv_test_reviews)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='GBM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()






