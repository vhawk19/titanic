import pandas as pd
import utils
from sklearn import linear_model, preprocessing
train=pd.read_csv("train.csv")
utils.clean_data(train)
target=train["Survived"].values
feautures=train[["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]].values

classifier=linear_model.LogisticRegression()

predictions=classifier.fit(feautures,target)
print(predictions.score(feautures,target))


poly=preprocessing.PolynomialFeatures(degree=2)
poly_features=poly.fit_transform(feautures)

predictions=classifier.fit(poly_features,target)
print(predictions.score(poly_features,target))
