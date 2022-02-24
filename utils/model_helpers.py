#!/usr/bin/env python3

import logging

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, confusion_matrix, roc_auc_score, roc_curve


def model_metrics(input_pipe, X_train, X_test, y_train, y_test):
    y_pred_test = input_pipe.predict(X_test)
    y_pred_train = input_pipe.predict(X_train)

    logging.info("Training Accuracy %s", (input_pipe.score(X_train, y_train)) * 100)
    logging.info("Test Accuracy %s", (input_pipe.score(X_test, y_test)) * 100)
    logging.info("MAE train %s", mean_absolute_error(y_train.astype('int'),
                                        y_pred_train.astype('int')))
    logging.info("MAE test %s", mean_absolute_error(y_test.astype('int'),
                                         y_pred_test.astype('int')))
    logging.info("AUC train %s", roc_auc_score(y_train, y_pred_train))
    logging.info("AUC test %s", roc_auc_score(y_test, y_pred_test))
    return True

def roc_auc_metrics(pipes):
    """ To plot a combined ROC plot. Takes a list of pipe model dictionaries as so:

        pipes = [
    {
        'label':'Dummy Classifier', 
        'pipe': dummy_pipe, 
    }, 
    {
        'label':'Random Forest', 
        'pipe': randomforest_pipe,
    },
    {
        'label':'XGBoost', 
        'pipe': xgb_pipe,
    }]
    """ 

    plt.figure(0).clf()

    for p in pipes:
        pipe = p['pipe']
        y_pred=pipe.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (p['label'], auc))
    
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    filename = "outgoing/roc_auc_curve.png"
    plt.savefig(filename)
    logging.info("ROC AUC curve plot saved to %s", filename)
    plt.close()
    return True