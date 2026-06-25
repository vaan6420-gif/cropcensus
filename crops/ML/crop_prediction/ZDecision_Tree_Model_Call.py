
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import sys



header = ['State_Name', 'District_Name', 'Season', 'Crop'] 

class Question:
    def __init__(self,column,value):
        self.column =column
        self.value=value
    def match(self,example):
        val = example[self.column]
        return val == self.value
    def match2(self,example):
        if example == 'True' or example == 'true' or example == '1':
            return True
        else:
            return False
    def __repr__(self):
        return "Is %s %s %s?" %(
            header[self.column],"==",str(self.value))
            
            
def class_counts(Data):
    counts= {}
    for row in Data:
        label =row[-1]
        if label not in counts:
             counts[label] = 0
        counts[label] += 1
    return counts


class Leaf:
    def __init__(self,Data):
        self.predictions = class_counts(Data)



class Decision_Node:
    def __init__(self,question,true_branch,false_branch):
        self.question=question
        self.true_branch = true_branch
        self.false_branch = false_branch




def print_tree(node,spacing=""):
    if isinstance(node,Leaf):
        print(spacing + "Predict",node.predictions)
        return
    print(spacing+str(node.question))
    print(spacing + "--> True:")
    print_tree(node.true_branch,spacing + " ")

    print(spacing + "--> False:")
    print_tree(node.false_branch,spacing + " ")




def print_leaf(counts):
    total = sum(counts.values())*1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] =str(int(counts[lbl]/total * 100)) + "%"
    return probs




def classify(row,node):
    if isinstance(node,Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row,node.true_branch)
    else:
        return classify(row,node.false_branch)


def get_best_prediction(predictions):
    # predictions is a dict like {'Rice': '70%', 'Wheat': '30%'}
    # Convert percentages to integers and pick the max
    best_crop = max(predictions.items(), key=lambda x: int(x[1].replace('%','')))
    return best_crop[0], best_crop[1]


dt_model_final= joblib.load('ML/crop_prediction/filetest2.pkl') 



state =sys.argv[1]
district=sys.argv[2]
season=sys.argv[3]


testing_data = [[state,district,season]]



for row in testing_data:
    prediction = classify(row, dt_model_final)
    probs = print_leaf(prediction)
    print("Input: State=%s, District=%s, Season=%s" % (row[0], row[1], row[2]))
    print("Predicted Crop Probabilities:", probs)
    Predict_dict = probs.copy()

    # Get best crop
    best_crop, confidence = get_best_prediction(Predict_dict)
    print(f"Most likely crop: {best_crop} ({confidence})")


