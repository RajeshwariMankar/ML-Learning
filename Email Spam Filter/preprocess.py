import os 
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

rootdir = r"/Spam Filter"

def preprocess(data):

    # converting data to lowercase
    l_data = data.lower()

    #remove numbers from the data
    a_data = re.sub(r'\d+','',l_data)

    # remove punctuations
    translate_data = dict((ord(char), None) for char in string.punctuation)
    str_data = a_data.translate(translate_data)
    #str_data = a_data.translate(str.maketrans("",""))

    #remove whiteaspaces
    str_data = str_data.strip()

    #remove stopwords and tokenize the words
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(str_data)
    tokenized_data = [word for word in tokens if not word in stop_words]

    #create dictionary that returns true for each word to pass it to the Naive Bayes Classifier
    filtered_data = dict([(words, True) for words in tokenized_data ])

    return(filtered_data)


def get_data(folder):
    ham_list = []
    spam_list = []

    for directories, subdirs, files in os.walk(rootdir):
        if (os.path.split(directories)[1]  == 'ham'):
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    #print(filename)
                    data = f.read()
                    ham_list.append((preprocess(data),"ham"))
        if (os.path.split(directories)[1]  == 'spam'):
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    data = f.read()

                    spam_list.append((preprocess(data), "spam"))
    print(ham_list[0])
    print(spam_list[0])

    return ham_list,spam_list

ham_list, spam_list = get_data(rootdir)



