import pandas as pd
import matplotlib.pyplot as plt
df_train=pd.read_csv("train.csv")
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
df_train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df_train.Survived,df_train.Age,alpha=0.05)
plt.title("Age wrt Survived")

plt.subplot2grid((2,3),(0,2))
df_train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("class")

plt.subplot2grid((2,3),(1,0),colspan=2)
for x in [1,2,3]:
    df_train.Age[df_train.Pclass == x].plot(kind="kde")
plt.title("Class wrt Age")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((2,3),(1,2))
df_train.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("embarked")

plt.show()
