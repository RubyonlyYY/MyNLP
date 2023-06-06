'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

from flair.models import TARSClassifier
from flair.data import Sentence

# 1. Load our pre-trained TARS model for English
tars = TARSClassifier.load('tars-base')

# 2. Prepare a test sentence
#sentence = Sentence("I am so glad you liked it!")
#sentence = Sentence("This dries my hair faster that bigger, more powerful models.")
sentence = Sentence("I am glad to see you")
# 3. Define some classes that you want to predict using descriptive names
#classes = ["pos", "neg","neu"]
classes = ["happy", "sad"]
#4. Predict for these classes
tars.predict_zero_shot(sentence, classes)

# Print sentence with predicted labels
print(sentence)