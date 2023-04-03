import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns
import re

def load_data(train_url, test_url):
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)
    return train, test

def free_tweet(dataset):
    twt_list = list()
    for i in range(dataset.shape[0]):
        tweets = dataset["text"].iloc[i].lower()
        tweets = re.sub("@[A-Za-z0-9_]+","",str(tweets))
        tweets = re.sub("#[A-Za-z0-9_]+","",str(tweets))
        tweets = re.sub(r"http\S+","",tweets)
        tweets = re.sub(r"www.\S+","",tweets)
        tweets = re.sub('[()!?]', ' ', tweets)
        tweets = re.sub('\[.*?\]',' ', tweets)
        tweets = re.sub("[^a-z0-9]"," ", tweets)
        twt_list.append(tweets)
        
    return twt_list

def count_tweets(train,twt_list, vect, vectorize):
    train["tweets"] = twt_list    
    twt_df = pd.DataFrame(vect.toarray(), columns=vectorize.get_feature_names_out())
    twt_df = twt_df.drop(["sentiment","text"], axis=1)

    training = pd.concat([train, twt_df], axis=1)

    neg = training.loc[training["sentiment"]==0]
    pos = training.loc[training["sentiment"]==1]

    posword_count = pos.iloc[:,3:]
    negword_count = neg.iloc[:,3:]

    poshigh = posword_count.T.sum(axis=1)
    neghigh = negword_count.T.sum(axis=1)

    poshigh_10 = np.argpartition(poshigh, -10)[-10:]
    neghigh_10 = np.argpartition(neghigh, -10)[-10:]

    positive_top10 = posword_count.columns[[poshigh_10]]
    negative_top10 = negword_count.columns[[neghigh_10]]
    return positive_top10, negative_top10
      

def preprocessing(train,test,label):
    

    training_clean = free_tweet(train) 
    testing_clean = free_tweet(test)
        
    if label==0:
        #CountVectorizer
        vectorize = CountVectorizer()
        train_vect = vectorize.fit_transform(training_clean)
        tst_vect = vectorize.transform(testing_clean)
        train_pos, train_neg = count_tweets(train, training_clean,train_vect,vectorize)
        test_pos, test_neg = count_tweets(test, testing_clean,tst_vect,vectorize)
        
        
    else:
        #Tfidvectorizer
        vectorize = TfidfVectorizer(use_idf=True)
        train_vect = vectorize.fit_transform(training_clean)
        tst_vect = vectorize.transform(testing_clean)
        train_pos, train_neg = count_tweets(train, training_clean,train_vect,vectorize)
        test_pos, test_neg = count_tweets(test, testing_clean,tst_vect,vectorize)
    
    return train_vect, tst_vect,train_pos,train_neg,test_pos,test_neg
    

def SVM_function(y_train, y_test, train_vec, test_vec, limits,label,gam):
    train_acc = np.zeros((len(limits),2))
    test_acc = np.zeros((len(limits),2))
    coefs = np.empty((len(limits),2))
    rbf_coefs = np.empty((len(limits),len(gam)))
    rbf_train_acc = np.zeros((len(limits),len(gam)))
    rbf_test_acc = np.zeros((len(limits),len(gam)))
    for j in range(len(label)):
        for i in range(len(limits)):
            if label[j]=="rbf":
                for k in range(len(gam)):
                    lsvm = svm.SVC(kernel="rbf", C=10**(limits[i]), gamma = 10**gam[k], probability=True)
                    
                    rbf_fit = lsvm.fit(train_mat, y_train)
                    rbf_coefs[i,k] = (lsvm.support_vectors_).shape[0]
                    train_pred = lsvm.predict(train_vec)
                    test_pred = lsvm.predict(test_vec)

                    rbf_train_acc[i,k] = accuracy_score(y_train, train_pred)
                    rbf_test_acc[i,k] = accuracy_score(y_test, test_pred)
            else:
                if label[j]== "poly":
                    lsvm = svm.SVC(kernel="poly",degree=2, C=10**(limits[i]), probability=True)
                else:
                    lsvm = svm.SVC(kernel="linear", C=10**(limits[i]), probability=True)
        
                linear_fit = lsvm.fit(train_mat, y_train)
                coefs[i,j] = (lsvm.support_vectors_).shape[0]
                train_pred = lsvm.predict(train_vec)
                test_pred = lsvm.predict(test_vec)

                train_acc[i,j] = accuracy_score(y_train, train_pred)
                test_acc[i,j] = accuracy_score(y_test, test_pred)
    return train_acc, test_acc, coefs,rbf_coefs,rbf_train_acc,rbf_test_acc

def svm_plots(support_vectors,train_a, test_a,ranges,tag):
    
    for i in range(train_a.shape[1]):
        fig = plt.figure()  
        plt.plot(ranges,train_a[:,i],"o",label="Training")
        plt.plot(ranges,test_a[:,i],"o", label="Validation")
        plt.xlabel("C values 10^i")
        plt.ylabel("Accuracy")
        plt.title(tag[i]+" SVM")
        plt.legend()
        #plt.show()
        plt.savefig("accuracy"+tag[i]+".pdf", format="pdf")
        plt.close(fig)

      
    for i in range(train_a.shape[1]):  
        fig = plt.figure()
        plt.plot(ranges, support_vectors[:,i],"o")
        plt.xlabel("C values 10^i")
        plt.ylabel("Support Vectors")
        plt.title(tag[i]+" Support Vectors")
        #plt.show()
        plt.savefig("supportvector"+tag[i]+".pdf", format="pdf")
        plt.close(fig)

def rbf_plots(support_vectors, train, validation, ranges, gammas):
    
    fig1 = plt.figure()
    train_heat = sns.heatmap(data=train, xticklabels =gammas , yticklabels=ranges ,annot=True)
    train_heat.set(xlabel="Gamma 10^i", ylabel="C values 10^i")
    plt.title("Training Accuracy for RBF Kernel")
    plt.savefig("rbftraining.pdf", format="pdf")
    plt.close()
    
    fig2 = plt.figure()
    test_heat = sns.heatmap(data=validation, xticklabels =gammas , yticklabels=ranges, annot=True)
    test_heat.set(xlabel="Gamma 10^i", ylabel="C values 10^i")
    plt.title("Validation Accuracy for RBF Kernel")
    plt.savefig("rbftesting.pdf", format="pdf")
    plt.close()
    
    fig3 = plt.figure()
    support_heat = sns.heatmap(data=support_vectors, xticklabels =gammas , yticklabels=ranges ,annot=False, linewidth=1)
    plt.title("Number of Support Vectors for RBF Kernel")
    plt.savefig("support.pdf", format="pdf")
    plt.close()
    
    fig4 = plt.figure()
    plt.plot(ranges, support_vectors[:,4],"o")
    plt.xlabel("C 10^i")
    plt.ylabel("Number of support vectors")
    plt.title("Support Vectors as a function of C for gamma = 0.1")
    plt.savefig("support_gamma01.pdf", format="pdf")
    plt.close()
    
    fig5 = plt.figure()
    plt.plot(gamma, support_vectors[5,:],"o")
    plt.xlabel("Gamma 10^i")
    plt.ylabel("Number of support vectors")
    plt.title("Support Vectors as a function of gamma for c = 10")
    plt.savefig("support_c10.pdf", format="pdf")
    plt.close()
    



#File path
train_location = "IA3-train.csv"
test_location = "IA3-dev.csv"


#Loading data
train, test = load_data(train_location,test_location)

#Data Preprocessing using CountVectorizer and TfidVectorizer

train_mat0, test_mat0, train_pos_100, train_neg_100,test_pos_100, test_neg_100 = preprocessing(train,test,0)

train_mat, test_mat, train_pos_10, train_neg_10,test_pos_10, test_neg_10 = preprocessing(train,test,1)


#Fitting Linear, Quadratic and RBF kernel SVMs

lims  = [-4,-3,-2,-1,0,1,2,3,4]
gamma = [-5,-2,-3,-2,-1,0,1]
train_a, test_a, weights, rbf_weights, rbf_acc, rbf_validation = SVM_function(train["sentiment"],test["sentiment"],train_mat,test_mat,lims,["linear","poly","rbf"],gamma)

#SVM plots
svm_plots(weights,train_a, test_a,lims,["Linear","Quadratic"])

rbf_plots(rbf_weights, rbf_acc, rbf_validation, lims, gamma)
