from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def cal_confusion(y_true, y_pred, labels=[-1, 1]):
    # Get confusion matrix
    confusion_mc = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    # Plot heatmap
    plt.figure(figsize=(5, 3))
    c = sns.heatmap(confusion_mc, annot=True, fmt='g')
    c.set(xticklabels=labels, yticklabels=labels)
    plt.show()

    # Printing the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, labels=labels))
