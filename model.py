import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from warnings import filterwarnings
filterwarnings('ignore')

data = pd.read_csv("/Users/harshulsinghal/Desktop/Dataset_spine.csv.xls")
data.head()
data.info()
data["Unnamed: 13"].unique()
data.Class_att.unique()

df = data.drop("Unnamed: 13", axis=1)
df.rename(columns = {
    "Col1" : "pelvic_incidence", 
    "Col2" : "pelvic_tilt",
    "Col3" : "lumbar_lordosis_angle",
    "Col4" : "sacral_slope", 
    "Col5" : "pelvic_radius",
    "Col6" : "degree_spondylolisthesis", 
    "Col7" : "pelvic_slope",
    "Col8" : "direct_tilt",
    "Col9" : "thoracic_slope", 
    "Col10" :"cervical_tilt", 
    "Col11" : "sacrum_angle",
    "Col12" : "scoliosis_slope", 
    "Class_att" : "target"}, inplace=True)

df["target"].value_counts().sort_index().plot.bar()
plt.subplots(figsize=(12,8))
sns.heatmap(df.corr())
sns.pairplot(df,hue="target")
dataset = df[["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis","target"]]
y = dataset.target
X = dataset.drop("target", axis=1)
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LogisticRegression()
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = classification_report(y_test,predictions,output_dict=True)["accuracy"]
print(accuracy)
pickle.dump(clf,open('model.pkl','wb'))
modle = pickle.load(open('model.pkl','rb'))