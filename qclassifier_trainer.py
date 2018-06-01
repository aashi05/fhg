import pandas
import sklearn
import scipy

from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.svm import LinearSVC

def pre_process(dta):
    return pandas.get_dummies(dta)

def transform_data_matrix(df_question_train):

    # Generate Compressed Sparse Row matrix:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    df_question_train = csr_matrix(df_question_train)

    return df_question_train


def remove_irrelevant_features(df_question):
    df_question_class = df_question.pop('Class')

    df_question.pop('Question')
    df_question.pop('WH-Bigram')

    return df_question_class

def support_vector_machine(df_question_train, df_question_class):
    lin_clf = LinearSVC()
    lin_clf.fit(df_question_train, df_question_class)
    
    return lin_clf

def train_question_classifier(training_data_path):
    """
    Question Classifier based on its feature.
    CSV: Training Data `qclassifier_trainer.csv`
        #Question|WH|WH-Bigram|WH-POS|WH-NBOR-POS|Root-POS|Class
    Using: Linear Support Vector Machine
    Model: Saved as Pickle `question_classifier.pkl`
    """

    df_question = pandas.read_csv(training_data_path, sep='|', header=0, encoding='cp1252')

    df_question_class = remove_irrelevant_features(df_question)

    df_question_train = pre_process(df_question)

    df_question_train = transform_data_matrix(df_question_train)

    support_vector_machine(df_question_train, df_question_class)



