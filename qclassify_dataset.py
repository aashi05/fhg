# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:46:04 2018

@author: Aashi agarwal
"""
import spacy
import csv
en_nlp = spacy.load("en_core_web_md")

def clean_old_data(training_data_path):

    question_features = ['Question', 'WH', 'WH-Bigram', 'WH-POS', 'WH-NBOR-POS', 'Root-POS', 'Class']

    with open(training_data_path, 'w', newline='') as csv_fp:
        csv_fp_writer = csv.writer(csv_fp, delimiter='|')
        csv_fp_writer.writerow(question_features)
        csv_fp.close()

def read_input_file(raw_data_file, training_data_path, en_nlp):

    with open(training_data_path, 'a', newline='') as csv_fp:
        csv_fp_writer = csv.writer(csv_fp, delimiter='|')
        f = open(raw_data_file,'r')
        f1=f.readlines()
        for row in f1:
            list_row = row.split(" ")
            question_class_list = list_row[0].split(":")
            question = " ".join(list_row[1:len(list_row)])
            question = question.strip("\n")
            question_class = question_class_list[0]
           
            process_question(question, question_class, en_nlp, training_data_path, csv_fp_writer)

        csv_fp.close()


def process_question(question, question_class, en_nlp, training_data_path, csv_fp_writer):
    en_doc = en_nlp(u'' + question)
    sentence_list = list(en_doc.sents)

    # Currently question classifier classifies only the 1st sentence of the question
    sentence = sentence_list[0]

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

    if wh_word != "" and " ".join(wh_bi_gram) != "" and wh_pos != "" and wh_nbor_pos != "":
        csv_fp_writer.writerow([question, wh_word, " ".join(wh_bi_gram), wh_pos, wh_nbor_pos, root_token, question_class])    
