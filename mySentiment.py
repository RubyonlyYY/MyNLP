'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

# import libraries
import pandas as pd
import seaborn as sns
import flair

# configure size of heatmap
sns.set(rc={'figure.figsize':(35,3)})

# function to visualize
def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Sentence").T,center=0, annot=True, cmap = "PiYG")

# model
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

# text
sentence = "To inspire and guide entrepreneurs is where I get my joy of contribution"

# sentiment
s = flair.data.Sentence(sentence)
flair_sentiment.predict(s)
total_sentiment = s.labels
total_sentiment

# tokenize sentiments
tokens = [token.text for token in s.tokens]
ss = [flair.data.Sentence(s) for s in tokens]
[flair_sentiment.predict(s) for s in ss]
sentiments = [s.labels[0].score * (-1,1)[str(s.labels[0]).split()[0].startswith("POS")] for s in ss]

# heatmap
visualise_sentiments({
      "Sentence":["SENTENCE"] + tokens,
      "Sentiment":[total_sentiment[0].score *(-1,1)[str(total_sentiment[0]).split()[0].startswith("POS")]] + sentiments,
}
######################################################################################another
# Data processing
import pandas as pd
# Hugging Face model
from transformers import pipeline
# Import flair pre-trained sentiment model
from flair.models import TextClassifier
classifier = TextClassifier.load('en-sentiment')

# Import flair Sentence to process input text
from flair.data import Sentence
# Import accuracy_score to check performance
from sklearn.metrics import accuracy_score
# Define a function to get Flair sentiment prediction score
# Read in data
amz_review = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt', sep='\t', names=['review', 'label'])

# Take a look at the data
amz_review.head()
def score_flair(text):
  # Flair tokenization
  sentence = Sentence(text)
  # Predict sentiment
  classifier.predict(sentence)
  # Extract the score
  score = sentence.labels[0].score
  # Extract the predicted label
  value = sentence.labels[0].value
  # Return the score and the predicted label
  return score, value

# Get sentiment score for each review
amz_review['scores_flair'] = amz_review['review'].apply(lambda s: score_flair(s)[0])

# Predict sentiment label for each review
amz_review['pred_flair'] = amz_review['review'].apply(lambda s: score_flair(s)[1])

# Check the distribution of the score
amz_review['scores_flair'].describe()

# Change the label of flair prediction to 0 if negative and 1 if positive
mapping = {'NEGATIVE': 0, 'POSITIVE': 1}
amz_review['pred_flair'] = amz_review['pred_flair'].map(mapping)

# Take a look at the data
amz_review.head()
# Compare Actual and Predicted
accuracy_score(amz_review['label'],amz_review['pred_flair'])