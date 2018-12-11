import pandas as pd
import utils
from sklearn import tree,model_selection
train=pd.read_csv("train.csv")
utils.clean_data(train)

target=train["Survived"].values
feautures_names=["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
feautures=train[feautures_names].values

decision_tree=tree.DecisionTreeClassifier(random_state=1)
decision_tree_predict=decision_tree.fit(feautures,target)

print(decision_tree_predict.score(feautures,target))

scores=model_selection.cross_val_score(decision_tree,feautures,target,scoring='accuracy',cv=50)
print(scores)
print(scores.mean())

generalized_tree=tree.DecisionTreeClassifier(
    random_state=1,
    max_depth=7,
    min_samples_split=2
)
generalized_tree_predict=generalized_tree.fit(feautures,target)

print(generalized_tree_predict.score(feautures,target))

scores=model_selection.cross_val_score(generalized_tree,feautures,target,scoring='accuracy',cv=50)
print(scores)
print(scores.mean())

tree.export_graphviz(generalized_tree_predict,feature_names=feautures_names,out_file="tree.dot")
