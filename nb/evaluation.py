def confusion_matrix(actual, predicted):

    unique = sorted(set(actual))
    matrix = [[0 for _ in unique] for _ in unique]
    imap   = {key: i for i, key in enumerate(unique)}
    # Generate Confusion Matrix
    for p, a in zip(predicted, actual):
        matrix[imap[p]][imap[a]] += 1
    return matrix

def accuracy_score(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    accuracy= 0
    for i in range(len(cm)):
        accuracy += cm[i][i]
    
    accuracy /= len(actual)

    return accuracy