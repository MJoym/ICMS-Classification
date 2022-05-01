import torch
import torch.nn.functional as F
from torch import stack
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np
from Functions.plt_loss_accuracy import plot_confusion_mat, build_confusion_mat, plot_roc, build_roc


# Paths for files that will be saved in script
path2saveCM = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/SGD_Model/FinalTestConfusionMat_45epochs_SGD1.png'
path2saveROC = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/SGD_Model/Final_TEST_ROC_45epochs_SGD1.png'


def test(model, test_dl):
    # initialize a dictionary to store testing history
    history = {"test_loss": [], "test_acc": []}
    totalTestLoss = []
    testCorrect = 0
    testCorrect2 = 0
    correct = []
    incorrect = []
    y_true = []
    y_pred = []
    y_pred_prob = []
    classes_names = ["Blank", "ICMS"]
    ndate = datetime.now()
    daten = ndate.strftime('%d_%m_%y')
    i = 0

    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the test set
        for (x, y) in test_dl:
            x = x[:, None, :, :, :] # Add fifth dimension - number of channels = 1 (gray scale image)
            x = x.type(torch.FloatTensor)

            # make the predictions and calculate the validation loss
            prob_pred = F.sigmoid(model(x))
            test_loss = F.cross_entropy(prob_pred, y)
            totalTestLoss.append(test_loss)

            # calculate the number of correct predictions
            argmax_preds = prob_pred.argmax(1)

            testCorrect += (argmax_preds == y).type(torch.float).sum().item()
            # testCorrect2 += (check_preds1 == y).type(torch.float).sum().item()

            # Params for confusion mat, ROC/AUC and precision-recall curves
            output = argmax_preds.data.cpu().numpy()
            max_prob_preds = prob_pred.cpu().numpy()
            max_prob_preds = [item[1] for item in max_prob_preds]
            true_labels = y.data.cpu().numpy()
            y_true.extend(true_labels)
            y_pred.extend(output)
            y_pred_prob.extend(max_prob_preds)

            # Append correct and incorrect file names for text files creation:
            for index in range(len(argmax_preds)):
                ind = test_dl.dataset.indices[i]
                if argmax_preds[index] == y[index]:
                    # correct.append(test_dl.dataset.samples[i])
                    correct.append(test_dl.dataset.dataset.samples[ind])
                else:
                    incorrect.append(test_dl.dataset.dataset.samples[ind])
                i = i + 1

        # Create a files with all the incorrect and correct predictions and save them:
        title_in = "Incorrect_preds_"+daten+"_SGD_fixed.txt"
        textfile = open(title_in, "w")
        for element in incorrect:
            textfile.write(element[0])
            textfile.write("\n")
        textfile.close()

        title_corr = "Correct_preds_"+daten+"_SGD_fixed.txt"
        textfile = open(title_corr, "w")
        for element in correct:
            textfile.write(element[0])
            textfile.write("\n")
        textfile.close()

        # avgTestLoss = torch.mean(stack(totalTestLoss))
        avgTestLoss = torch.mean(stack(totalTestLoss))
        testlen = len(test_dl.dataset)
        testAccuracy = testCorrect / testlen

        # history["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        history["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        history["test_acc"].append(testAccuracy)

        print("\n##########################################################################")
        print("##########################################################################\n")
        print("Test loss: {:.6f}, Test accuracy: {:.4f}".format(avgTestLoss, testAccuracy))
        print("\n")
        print("##########################################################################")
        print("##########################################################################\n")

        # Build and plot confusion matrix for test set:
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 100, index=[i for i in classes_names], columns=[i for i in classes_names])
        build_confusion_mat(df_cm, path2saveCM)

        # Build and plot ROC and AUC curve:
        fpr, tpr, thr = roc_curve(y_true, y_pred_prob, pos_label=1)
        print('Thresholds: ', thr)

        # roc curve for tpr = fpr
        random_probs = [0 for i in range(len(y_true))]
        p_fpr, p_tpr, _ = roc_curve(y_true, random_probs, pos_label=1)

        build_roc(fpr, tpr, p_fpr, p_tpr, path2saveROC)
        # calculate AUC:
        auc = roc_auc_score(y_true, y_pred_prob)
        print('AUC: %.3f' % auc)

        return history

