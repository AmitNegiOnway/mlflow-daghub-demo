import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='amitnegionway', repo_name='mlflow-daghub-demo', mlflow=True)


mlflow.set_tracking_uri('https://dagshub.com/amitnegionway/mlflow-daghub-demo.mlflow') 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
max_depth = 10
n_estimator =100

# apply mlflow 

mlflow.set_experiment('iris-rf') # this is of the option for experiment write a experiment name but also do experiment_id='479759578844802238'but for this u manually create new experiment in mlflow tool and then copy their experiment_id but in option 1 u dont need to do this.

with mlflow.start_run(): # this is called contaxt manager
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimator)

    rf.fit(X_train,y_train)

    y_pred=dt.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth',max_depth)
    mlflow.lod_param('n_estimator',n_estimator)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    # save the plot as an artifact
    plt.savefig('confusion_matrix.png')

    # mlflow artifact 
    mlflow.log_artifact('confusion_matrix.png')

    # mlflow code (hum code ko bhi log kar skte hai par normally dvc ye kar deta hai par mlflow mai ye ho jata hai.)
    mlflow.log_artifact(__file__) # here code is also a artifact..

    # how can log the model (means - this decision tree model.)
    mlflow.sklearn.save_model(rf,"random_forest") # you do little modify of these 2 line of code i u used dagshub for mlflow tracking server...
    mlflow.log_artifact("random_forest")

    # mlflow.log_model(dt,"decision tree") this one also used but if u used previous one they provided more meta data...
    
    # mlflow tags
    mlflow.set_tag('author',' rahul')
    mlflow.set_tag('model',' random forest')
    mlflow.set_tag('day','6/26/2025')

    print('accuracy',accuracy)

