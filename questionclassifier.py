# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:09:11 2018

@author: Aashi agarwal
"""
import pandas
import spacy
import scipy
import sklearn

from sklearn import svm
from sklearn.svm import LinearSVC
from scipy import sparse
from scipy.sparse import csr_matrix

def remove_irrelevant_features(df_question):
    df_question_class = df_question.pop('Class')

    df_question.pop('Question')
    df_question.pop('WH-Bigram')

    return df_question_class

def pre_process(dta):
    return pandas.get_dummies(dta)

def transform_data_matrix(df_question_train, df_question_predict):

    df_question_train_columns = list(df_question_train.columns)
    df_question_predict_columns = list(df_question_predict.columns)

    df_question_trans_columns = list(set(df_question_train_columns + df_question_predict_columns))

    trans_data_train = {}

    for feature in df_question_trans_columns:
        if feature not in df_question_train:
            trans_data_train[feature] = [0 for i in range(len(df_question_train.index))]
        else:
            trans_data_train[feature] = list(df_question_train[feature])

    df_question_train = pandas.DataFrame(trans_data_train)
    df_question_train = csr_matrix(df_question_train)

    trans_data_predict = {}

    for feature in trans_data_train:
        if feature not in df_question_predict:
            trans_data_predict[feature] = 0
        else:
            trans_data_predict[feature] = list(df_question_predict[feature])  # KeyError

    df_question_predict = pandas.DataFrame(trans_data_predict)
    df_question_predict = csr_matrix(df_question_predict)

    return df_question_train, df_question_predict

def get_question_predict_data(en_doc):
    sentence_list = list(en_doc.sents)[0:1]
    en_nlp = spacy.load("en_core_web_md")

    question_data_frame = []

    for sentence in sentence_list:

        wh_bi_gram = []
        root_token, wh_pos, wh_nbor_pos, wh_word = [""] * 4
        for token in sentence:

        # if token is of WH question type
            if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
                wh_pos = token.tag_
                wh_word = token.text
                wh_bi_gram.append(token.text)
                wh_bi_gram.append(str(en_doc[token.i + 1]))
                wh_nbor_pos = en_doc[token.i + 1].tag_
    
            # if token is the root of sentence
            if token.dep_ == "ROOT":
                root_token = token.tag_
    
        question_data_frame_obj = {'WH': wh_word, 'WH-POS': wh_pos, 'WH-NBOR-POS': wh_nbor_pos, 'Root-POS': root_token}
        question_data_frame.append(question_data_frame_obj)
        
        df_question = pandas.DataFrame(question_data_frame)

    return df_question

def support_vector_machine(df_question_train, df_question_class, df_question_predict):
    lin_clf = LinearSVC()
    lin_clf.fit(df_question_train, df_question_class)
    prediction = lin_clf.predict(df_question_predict)
    return prediction, lin_clf

def classify_question(en_doc):
    training_data_path = "D:/Qclassifier/qclassify.csv"
    df_question_train = pandas.read_csv(training_data_path, sep='|', header=0, encoding='cp1252')
    
    df_question_class = remove_irrelevant_features(df_question_train)
    df_question_predict = get_question_predict_data(en_doc)
    df_question_train = pre_process(df_question_train)
    df_question_predict = pre_process(df_question_predict)
    
    df_question_train, df_question_predict = transform_data_matrix(df_question_train, df_question_predict)
    predicted_class, svc_clf = support_vector_machine(df_question_train, df_question_class, df_question_predict)
    print (predicted_class)
    
if __name__ == "__main__":

    en_nlp_l = spacy.load("en_core_web_md")
    question = input("What would you like to know?")
    en_doc_l = en_nlp_l(u'' + question)
    question_class = classify_question(en_doc_l)
