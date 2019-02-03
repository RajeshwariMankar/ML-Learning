# import necessary functions from python files
from preprocess import preprocess
from classifier import classifier

# Clasify the below as spam or ham
 

# 1. preprocess the data
# 2. Use the classify function
 
msg1 = '''add any spam msg'''
msg2 = ''' add some non spam msg'''
msg3 = '''add some more msg'''

features = preprocess(msg1)
print("Message 1 is :" ,classifier.classify(features))

features = preprocess(msg2)
print("Message 2 is :" ,classifier.classify(features))

features = preprocess(msg3)
print("Message 3 is :" ,classifier.classify(features))
