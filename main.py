import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def import_data():
    df = pd.read_csv("Symptoms_Dataset.csv")
    
    return df


def convert_data_values(df):
    data = {'Yes': 1}
    for column in df.columns:
        if column != 'Class Label':
            df[column] = df[column].map(data)
            df[column] = df[column].fillna(0)
    
    return df


def split_dataset(df, test_size, random_state):
    X = df.values[:, 0:28]
    Y = df.values[:, 28]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def train_using_gini(max_depth, X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=max_depth)
    clf_gini.fit(X_train, y_train)
    
    return clf_gini


def train_using_naive_bayes(X_train, y_train):
    gnb = GaussianNB(var_smoothing=0.1)
    gnb.fit(X_train, y_train) 
    
    return gnb


def predict_output(X_test, model):
    y_pred = model.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    
    return y_pred
    

def evaluate_accuracy(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion)
    
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=np.unique(y_pred))
    display.plot()
    plt.show()
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy * 100)
    
    report = classification_report(y_test, y_pred, labels=np.unique(y_pred))
    print("Report:\n", report)


def main(test_size=None, random_state=None, max_depth=None, model=None):
    df = import_data()
    converted_df = convert_data_values(df)
    X_train, X_test, y_train, y_test = split_dataset(converted_df, test_size, random_state)
    
    arr = []
    count = 0
    
    if model == 'clf':
        print("=============================================================================================")
        print("Experiment using Decision Tree with test size = %.2f, random state = %d and max depth = %d\n" %(test_size, random_state, max_depth))
        
        clf_gini = train_using_gini(max_depth, X_train, y_train)
        
        print("Results of using Gini:")
        y_pred_gini = predict_output(X_test, clf_gini)
        evaluate_accuracy(y_test, y_pred_gini)
       
        for i in range(len(y_test)):
            if y_test[i] == y_pred_gini[i]:
                arr.append((y_test[i], y_pred_gini[i]))
                
                for j in range(len(arr)):
                    if arr[j] == (y_test[i], y_pred_gini[i]):
                        count += 1
                    
                plt.scatter(y_test[i], y_pred_gini[i], s=(count * 50), color='blue')
                count = 0
            elif y_test[i] != y_pred_gini[i]:
                arr.append((y_test[i], y_pred_gini[i]))
                
                for j in range(len(arr)):
                    if arr[j] == (y_test[i], y_pred_gini[i]):
                        count += 1
                    
                plt.scatter(y_test[i], y_pred_gini[i], s=(count * 50), color='red')
                count = 0
       
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()
        print("=============================================================================================\n")
    elif model == 'gnb':
        print("=============================================================================================")
        print("Experiment using Naive Bayes with test size = %.2f and random state = %d\n" %(test_size, random_state))
        
        gnb = train_using_naive_bayes(X_train, y_train)
        
        print("Results of using Gaussian Naive Bayes:")
        y_pred_nb = predict_output(X_test, gnb)
        evaluate_accuracy(y_test, y_pred_nb)
        
        for i in range(len(y_test)):
            if y_test[i] == y_pred_nb[i]:
                arr.append((y_test[i], y_pred_nb[i]))
                
                for j in range(len(arr)):
                    if arr[j] == (y_test[i], y_pred_nb[i]):
                        count += 1
                    
                plt.scatter(y_test[i], y_pred_nb[i], s=(count * 50), color='blue')
                count = 0
            elif y_test[i] != y_pred_nb[i]:
                arr.append((y_test[i], y_pred_nb[i]))
                
                for j in range(len(arr)):
                    if arr[j] == (y_test[i], y_pred_nb[i]):
                        count += 1
                    
                plt.scatter(y_test[i], y_pred_nb[i], s=(count * 50), color='red')
                count = 0
                
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()
        print("=============================================================================================\n")


if __name__ == '__main__':
    main(test_size=0.2, random_state=50, max_depth=15, model='clf')
    main(test_size=0.2, random_state=50, model='gnb')