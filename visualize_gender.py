import pandas as pd
import matplotlib.pyplot as plt

female_color="#FA0000"

df_train=pd.read_csv("train.csv")
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,0))
df_train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3,4),(0,1))
df_train.Survived[df_train.Sex=="male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Male Survived")

plt.subplot2grid((3,4),(0,2))
df_train.Survived[df_train.Sex=="female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=female_color)
plt.title("Female Survived")

plt.subplot2grid((3,4),(0,3))
df_train.Sex[df_train.Survived == 1].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=[female_color,'b'] )
plt.title("Sex of survived")

plt.subplot2grid((3,4),(1,0),colspan=4)
for x in [1,2,3]:
    df_train.Survived[df_train.Pclass == x].plot(kind="kde")
plt.title("Class wrt Survived")
plt.legend(("1st","2nd","3rd"))
plt.title("Female Survived")

plt.subplot2grid((3,4),(2,0))
df_train.Survived[(df_train.Sex=="male") & (df_train.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Rich Male Survived")

plt.subplot2grid((3,4),(2,1))
df_train.Survived[(df_train.Sex=="male")&(df_train.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Poor Male Survived")


plt.subplot2grid((3,4),(2,2))
df_train.Survived[(df_train.Sex=="female") & (df_train.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=female_color)
plt.title("Rich FeMale Survived")

plt.subplot2grid((3,4),(2,3))
df_train.Survived[(df_train.Sex=="female")&(df_train.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=female_color)
plt.title("Poor FeMale Survived")

plt.show()
