from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
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
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np
from numpy import array,asarray,zeros

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup



def cleaning(data):
  
  # removing tags
  phrase = remove_tags(data)
  
  # remove special characters i.e r
  # remove anything thats not a to z or A-Z
  
  phrase = re.sub('[^a-zA-Z]', ' ', phrase)
  
  # remove multiple spaces
  phrase = re.sub(r'\s+', ' ',phrase)
  
  return phrase


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
  return TAG_RE.sub('', text)

X = []
phrases = list(movie['review'])
for data in phrases:
  X.append(cleaning(data))

train_reviews=movie.review[:40000]
train_sentiments=movie.sentiment[:40000]

#test dataset
test_reviews=movie.review[40000:]
test_sentiments=movie.sentiment[40000:]

y = movie['sentiment']
y = np.array(list(map(lambda x: 1 if x=='positive' else 0,y)))
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
print(stopword_list)

from bs4 import BeautifulSoup
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
movie['review']=movie['review'].apply(denoise_text)

#function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
movie['review']=movie['review'].apply(remove_special_characters)


def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
movie['review']=movie['review'].apply(remove_special_characters)


#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
movie['review']=movie['review'].apply(simple_stemmer)

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
movie['review']=movie['review'].apply(remove_stopwords)

#normalized train reviews
norm_train_reviews=movie.review[:40000]
norm_train_reviews[100]

#Normalized test reviews
norm_test_reviews=movie.review[40000:]
norm_test_reviews[45005]


#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data=lb.fit_transform(movie['sentiment'])
print(sentiment_data.shape)

#Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]