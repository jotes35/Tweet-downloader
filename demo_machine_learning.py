# generating roc curve and auc in machine learning



from matplotlib import pyplot as plt


#classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,f1_score

# generate a 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

# split into train & test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)

# fit and train the model
def fit_model(model,trainX,trainy,testX,testy):
   
    model.fit(trainX, trainy)

    # predict probabilities
    probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate AUC
    auc = roc_auc_score(testy, probs)
    predictedy=model.predict(testX)
    accuracy=accuracy_score(testy,predictedy)
    f1=f1_score(testy,predictedy)
    #print('AUC: %.2f' % auc)
    # calculate roc curve and generate false positive rates and true positive rates
    # note: 1-specificity is false positive rate and sensitivity is true positive rate
    fpr, tpr, thresholds = roc_curve(testy, probs)
    return(fpr,tpr,thresholds,auc,accuracy,f1)

def plot_auc(fpr,tpr,auc):
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model and label with AUC
    plt.plot(fpr, tpr, marker='.',label='AUC: %0.2f'%auc)
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate / 1-Specifity')
    plt.ylabel('True Positive Rate / Sensitivity')
#title of plot
    plt.title('Receiver Operating Characteristic (ROC) Curve')
# show the plot
    plt.show()

def plot_several_auc():
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    knn= KNeighborsClassifier(n_neighbors=3)
    fpr_k,tpr_k,thresholds,auc_k,accuracy,f1=fit_model(knn,trainX,trainy,testX,testy)
    
    nb=GaussianNB()
    fpr_nb,tpr_nb,thresholds,auc_nb,accuracy,f1=fit_model(nb,trainX,trainy,testX,testy)

    
    rf=RandomForestClassifier()
    fpr_rf,tpr_rf,thresholds,auc_rf,accuracy,f1=fit_model(rf,trainX,trainy,testX,testy)
    
# plot the roc curve for the model and label with AUC
    #knearestneighbor
    plt.plot(fpr_k, tpr_k, marker='.',label='KNN-AUC: %0.2f'%auc_k)

    #naive bayes
    plt.plot(fpr_nb, tpr_nb, marker='.',label='NAIVE BAYES-AUC: %0.2f'%auc_nb)
    
    #plot for random forest
    plt.plot(fpr_rf, tpr_rf, marker='.',label='RANDOM FOREST-AUC: %0.2f'%auc_rf)

    
    
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate / 1-Specifity')
    plt.ylabel('True Positive Rate / Sensitivity')
#title of plot
    plt.title('Receiver Operating Characteristic (ROC) Curve')
# show the plot
    plt.show()

#classification with K Nearest Neighbor
def plot_knn():
    knn=KNeighborsClassifier(n_neighbors=3)
    fpr,tpr,thresholds,auc,accuracy,f1=fit_model(knn,trainX,trainy,testX,testy)
    print('accuracy:%s'%accuracy)
    print('f1:%s'%f1)
    plot_auc(fpr,tpr,auc)



#plot_knn()
plot_several_auc()


