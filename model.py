import os
import sys
import numpy as np
import pandas as pd
import warnings

import dagshub
dagshub.init(repo_owner='sudipta.mahato.ece25',
             repo_name='DeepLearrning-With-MLFlow',
             mlflow=True)


warnings.filterwarnings("ignore")
from urllib.parse import urlparse

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import logging
logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

def evaluate(Y_test,Y_pred):
    r2=r2_score(Y_test,Y_pred)
    mae=mean_absolute_error(Y_test,Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
    return r2,mae,rmse


csv_url=("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")
try:
    df=pd.read_csv(csv_url,sep=";")
except Exception as e:
    logging.exception("FAILED TO READ THE DATA %S",e)


train,test=train_test_split(df)
X_train=train.drop(["quality"],axis=1)
X_test=test.drop(["quality"],axis=1)
Y_train=train["quality"]
Y_test=test["quality"]


alpha=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
l1=float(sys.argv[2]) if len(sys.argv)>1 else 0.5

with mlflow.start_run():
    lr=ElasticNet(alpha=alpha,l1_ratio=l1)
    lr.fit(X_train,Y_train)

    Y_preds=lr.predict(X_test)

    (r2,mae,rmse)=evaluate(Y_test,Y_preds)

    print("ElasticNet model(alpha={:f},l1_ratio={:f})".format(alpha,l1))
    print("R2_score: %s"%r2)
    print("mae: %s"%mae)
    print("rmse: %s"%rmse)

    mlflow.log_param("alpha",alpha)
    mlflow.log_param("l1_ratio",l1)
    mlflow.log_metric("r2_score",r2)
    mlflow.log_metric("mae",mae)
    mlflow.log_metric("rmse",rmse)

    remote_server_uri="https://dagshub.com/sudipta.mahato.ece25/DeepLearrning-With-MLFlow.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme    

    if tracking_url_type_store!="file":
        mlflow.sklearn.log_model(lr,"model",registered_model_name="ElasticNetWineModel")
    else:
        mlflow.sklearn.log_model(lr,"model")