from evaluation import accuracy_score, confusion_matrix
from naive_bayes import GaussianNB, MultinomialNB
from preprocessing import Preprocessing
# TODO @Waldek I would not create this file, your code is rather library and this stuff below is rather some experiment. 
# You can add it as a separate notebook
def main():
    prp = Preprocessing()
    prp.get_datasets_annotations()
    X_train, y_train, X_test, y_test = prp.get_dataset()

    gnb = GaussianNB()
    y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
    cm_gnb = confusion_matrix(y_test.ravel(),y_pred_gnb.ravel())
    acc_gnb = accuracy_score(y_test.ravel(),y_pred_gnb.ravel())
    print("Gaussian Naive Bayes - confusion matrix")
    print(cm_gnb)
    print(f"Accuracy: {acc_gnb * 100} %")
    mnb = MultinomialNB(bins=37)
    y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)
    cm_mnb = confusion_matrix(y_test.ravel(),y_pred_mnb.ravel())
    acc_mnb = accuracy_score(y_test.ravel(),y_pred_mnb.ravel())
    print("Multinomial Naive Bayes - confusion matrix")
    print(cm_mnb)
    print(f"Accuracy: {acc_mnb * 100} %")

if __name__ == "__main__":
    main()
