import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re

train = pd.read_csv("IA3-train.csv")
test = pd.read_csv("IA3-dev.csv")


def free_tweet(dataset):
    twt_list = list()
    for i in range(dataset.shape[0]):
        tweets = dataset["text"].iloc[i].lower()
        tweets = re.sub("@[A-Za-z0-9_]+", "", str(tweets))
        tweets = re.sub("#[A-Za-z0-9_]+", "", str(tweets))
        tweets = re.sub(r"http\S+", "", tweets)
        tweets = re.sub(r"www.\S+", "", tweets)
        tweets = re.sub('[()!?]', ' ', tweets)
        tweets = re.sub('\[.*?\]', ' ', tweets)
        tweets = re.sub("[^a-z0-9]", " ", tweets)
        twt_list.append(tweets)

    return twt_list


def count_tweets(train, twt_list, vect, vectorize):
    train["tweets"] = twt_list
    twt_df = pd.DataFrame(vect.toarray(), columns=vectorize.get_feature_names_out())
    twt_df = twt_df.drop(["sentiment", "text"], axis=1)

    training = pd.concat([train, twt_df], axis=1)

    neg = training.loc[training["sentiment"] == 0]
    pos = training.loc[training["sentiment"] == 1]

    posword_count = pos.iloc[:, 3:]
    negword_count = neg.iloc[:, 3:]

    poshigh = posword_count.T.sum(axis=1)
    neghigh = negword_count.T.sum(axis=1)

    poshigh_10 = np.argpartition(poshigh, -10)[-10:]
    neghigh_10 = np.argpartition(neghigh, -10)[-10:]

    positive_top10 = posword_count.columns[[poshigh_10]]
    negative_top10 = negword_count.columns[[neghigh_10]]
    return positive_top10, negative_top10


def preprocessing(train, test, label):
    training_clean = free_tweet(train)
    testing_clean = free_tweet(test)

    if label == 0:
        # CountVectorizer
        vectorize = CountVectorizer()
        train_vect = vectorize.fit_transform(training_clean)
        tst_vect = vectorize.transform(testing_clean)
        train_pos, train_neg = count_tweets(train, training_clean, train_vect, vectorize)
        # test_pos, test_neg = count_tweets(test, testing,tst_vect)


    else:
        # Tfidvectorizer
        vectorize = TfidfVectorizer(use_idf=True)
        train_vect = vectorize.fit_transform(training_clean)
        tst_vect = vectorize.transform(testing_clean)
        # train_pos, train_neg = count_tweets(train, training,train_vect)
        # test_pos, test_neg = count_tweets(test, testing,tst_vect)

    # return train_vect, tst_vect,train_pos,train_neg,test_pos,test_neg
    return train_vect, tst_vect, train_pos


trn, tst, dff = preprocessing(train, test, 0)

print(dff)
