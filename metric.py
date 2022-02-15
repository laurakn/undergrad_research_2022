import numpy as np


def Demographic_parity_prob(group_labels, prediction_labels):
    """
        Input:
            group_labels-np array
            prediction_labels-np array
        Output:
            return P(Y_hat = 1 | A = 0), P(Y_hat = 1 | A = 1).
    """
    return np.sum(prediction_labels[group_labels == 0] == 1) / np.sum(group_labels == 0), np.sum(prediction_labels[group_labels == 1] == 1) / np.sum(group_labels == 1)

def Demographic_parity_worst_group(group_labels, prediction_labels):
    """
        Input:
            group_labels-np array
            prediction_labels-np array
        Output:
            return the disadvantaged group.
            larger than 0: group 1
            smaller than 0: group 0
            equal to 0: fair
    """
    return np.sum(prediction_labels[group_labels == 0] == 1) / np.sum(group_labels == 0) - np.sum(prediction_labels[group_labels == 1] == 1) / np.sum(group_labels == 1)

def Equal_opportunity_prob(group_labels, prediction_labels, classification_labels):
    """
        Input:
            group_labels-np array
            prediction_labels-np array
            classification_labels-np array
        Output:
            return P(Y_hat = 1 | A = 0, Y = 1), P(Y_hat = 1 | A = 1, Y = 1).
    """
    group0 = group_labels == 0
    group0_prediction = prediction_labels[group0]
    group0_classification = classification_labels[group0]
    group0_PosClass = group0_classification == 1
    # P(Y_hat = 1 | A = 0, Y = 1)
    P1 = np.sum(group0_prediction[group0_PosClass] == 1) / np.sum(group0_PosClass)
    # P(Y_hat = 1 | A = 1, Y = 1)
    group1 = group_labels == 1
    group1_prediction = prediction_labels[group1]
    group1_classification = classification_labels[group1]
    group1_PosClass = group1_classification == 1
    P2 = np.sum(group1_prediction[group1_PosClass] == 1) / np.sum(group1_PosClass)
    return P1, P2

def Equal_opportunity_worst_group(group_labels, prediction_labels, classification_labels):
    """
        Input:
            group_labels-np array
            prediction_labels-np array
            classification_labels-np array
        Output:
            return the disadvantaged group.
            larger than 0: group 1
            smaller than 0: group 0
            equal to 0: fair
    """
    group0 = group_labels == 0
    group0_prediction = prediction_labels[group0]
    group0_classification = classification_labels[group0]
    group0_PosClass = group0_classification == 1
    # P(Y_hat = 1 | A = 0, Y = 1)
    P1 = np.sum(group0_prediction[group0_PosClass] == 1) / np.sum(group0_PosClass)
    # P(Y_hat = 1 | A = 1, Y = 1)
    group1 = group_labels == 1
    group1_prediction = prediction_labels[group1]
    group1_classification = classification_labels[group1]
    group1_PosClass = group1_classification == 1
    P2 = np.sum(group1_prediction[group1_PosClass] == 1) / np.sum(group1_PosClass)
    return P1 - P2

def Equal_odds_prob(group_labels, prediction_labels, classification_labels):
    """
        Input:
            group_labels-np array
            prediction_labels-np array
            classification_labels-np array
        Output:
            return P(Y_hat = 1 | A = 0, Y = 1), P(Y_hat = 1 | A = 1, Y = 1), P(Y_hat = 1 | A = 0, Y = 0), P(Y_hat = 1 | A = 1, Y = 0).
    """
    group0 = group_labels == 0
    group0_prediction = prediction_labels[group0]
    group0_classification = classification_labels[group0]
    group0_PosClass = group0_classification == 1
    # P(Y_hat = 1 | A = 0, Y = 1)
    P1 = np.sum(group0_prediction[group0_PosClass] == 1) / np.sum(group0_PosClass)
    
    group1 = group_labels == 1
    group1_prediction = prediction_labels[group1]
    group1_classification = classification_labels[group1]
    group1_PosClass = group1_classification == 1
    # P(Y_hat = 1 | A = 1, Y = 1)
    P2 = np.sum(group1_prediction[group1_PosClass] == 1) / np.sum(group1_PosClass)


    group0_NegClass = group0_classification == -1
    # P(Y_hat = 1 | A = 0, Y = -1)
    print(np.sum(group0_NegClass))
    P3 = np.sum(group0_prediction[group0_NegClass] == 1) / np.sum(group0_NegClass)

    group1_NegClass = group1_classification == -1
    # P(Y_hat = 1 | A = 1, Y = -1)
    P4 = np.sum(group1_prediction[group1_NegClass] == 1) / np.sum(group1_NegClass)
    return P1, P2, P3, P4

def Equal_odds_worst_group(group_labels, prediction_labels, classification_labels):
    """
        Input:
            group_labels-np array
            prediction_labels-np array
            classification_labels-np array
        Output:
            The disadvantaged group
            larger than 0: group 1
            smaller than 0: group 0
            equal to 0: fair
    """
    group0 = group_labels == 0
    group0_prediction = prediction_labels[group0]
    group0_classification = classification_labels[group0]
    group0_PosClass = group0_classification == 1
    # P(Y_hat = 1 | A = 0, Y = 1)
    P1 = np.sum(group0_prediction[group0_PosClass] == 1) / np.sum(group0_PosClass)
    
    group1 = group_labels == 1
    group1_prediction = prediction_labels[group1]
    group1_classification = classification_labels[group1]
    group1_PosClass = group1_classification == 1
    # P(Y_hat = 1 | A = 1, Y = 1)
    P2 = np.sum(group1_prediction[group1_PosClass] == 1) / np.sum(group1_PosClass)


    group0_NegClass = group0_classification == -1
    # P(Y_hat = 1 | A = 0, Y = -1)
    print(np.sum(group0_NegClass))
    P3 = np.sum(group0_prediction[group0_NegClass] == 1) / np.sum(group0_NegClass)

    group1_NegClass = group1_classification == -1
    # P(Y_hat = 1 | A = 1, Y = -1)
    P4 = np.sum(group1_prediction[group1_NegClass] == 1) / np.sum(group1_NegClass)
    return (P1 + P3) - (P2 + P4)

def Overall_Accuracy_prob(group_labels, prediction_labels, classification_labels):
    """
    Input:
            group_labels-np array
            prediction_labels-np array
            classification_labels-np array
        Output:
            return overall accuracy for group 0 and group 1.
    """
    accuracy0 = np.sum(prediction_labels[group_labels == 0] == classification_labels[group_labels == 0]) / np.sum(group_labels == 0)
    accuracy1 = np.sum(prediction_labels[group_labels == 1] == classification_labels[group_labels == 1]) / np.sum(group_labels == 1)
    return accuracy0, accuracy1

def Overall_Accuracy_worst_group(group_labels, prediction_labels, classification_labels):
    """
    Input:
            group_labels-np array
            prediction_labels-np array
            classification_labels-np array
        Output:
            return the disadvantaged group.
            larger than 0: group 1
            smaller than 0: group 0
            equal to 0: fair
    """
    accuracy0 = np.sum(prediction_labels[group_labels == 0] == classification_labels[group_labels == 0]) / np.sum(group_labels == 0)
    accuracy1 = np.sum(prediction_labels[group_labels == 1] == classification_labels[group_labels == 1]) / np.sum(group_labels == 1)
    return accuracy0 - accuracy1



# def main():
#     group_labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1])
#     classification_labels = np.array([1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1])
#     prediction_labels = np.array([1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1])

#     print(Overall_Accuracy_prob(group_labels, prediction_labels, classification_labels))

# if __name__ == '__main__':
#     main()
