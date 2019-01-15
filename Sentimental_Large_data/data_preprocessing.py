import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
import re

lemmatizer = WordNetLemmatizer()

def init_process(fin,fout):
    outfile = open(fout,'a')
    with open(fin,buffering=200000,encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                elif initial_polarity == '4':
                    initial_polarity = [0,1]

                tweet = line.split(',')[-1]
                # pat1 = r'@[A-Za-z0-9_]+'
                # pat2 = r'https?://[^ ]+'
                # combine = r'|'.join((pat1, pat2))
                # tweet2 = re.sub(combine,'',tweet)
                # print(tweet2==tweet)
                put = str(initial_polarity)+':::'+tweet
                outfile.write(put)

        except Exception as e:
            print(str(e))

    outfile.close()

init_process('train.csv','train2.csv')
init_process('test.csv','test2.csv')

def create_lexicon(fin):
    lexicon = []
    with open(fin,'r',buffering=100000,encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter+=1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content = ' ' + tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon+words))
                    print(counter,len(lexicon))

        except Exception as e:
            print(str(e))

    with open('lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f)

create_lexicon('train2.csv')

def convert_to_vec(fin,fout,lexicon_pickle):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout,'a')
    with open(fin,buffering=20000,encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter+=1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet)
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)

        print(counter)

convert_to_vec('test2.csv','processed-test-set2.csv','lexicon.pickle')

def shuffle(fin):
    df = pd.read_csv(fin,error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('train_shuffle.csv',index=False)

shuffle('train2.csv')

def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin,buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter+=1
            except:
                pass
        print(counter)
        feature_sets = np.array(feature_sets)
        labels = np.array(labels)

create_test_data_pickle('processed-test-set2.csv')



