import pandas as pd
import utils
from sklearn import tree,model_selection
import numpy as np

test=pd.read_csv("test.csv")
utils.clean_data(test)


train=pd.read_csv("train.csv")
utils.clean_data(train)

target=train["Survived"].values
feautures_names=["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
feautures=train[feautures_names].values

generalized_tree=tree.DecisionTreeClassifier(
    random_state=1,
    max_depth=7,
    min_samples_split=2
)
generalized_tree_predict=generalized_tree.fit(feautures,target)

feautures_test=test[feautures_names].values
predictions=generalized_tree_predict.predict(feautures_test)
utils.write_prediction(predictions,"naive_decision_Tree_prediction.csv")
