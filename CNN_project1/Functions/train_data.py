import pickle
import torch
from torch import stack
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from Functions.plt_loss_accuracy import plot_confusion_mat, build_confusion_mat, plot_roc, build_roc, save_acc_loss_fig

def train(model, train_dl, val_dl, epochs, start_epoch, loss, opt):
    """This function trains our model according to the hyperparameters given"""

    # initialize a dictionary to store training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "y_true": [], "y_pred": [], "class1_probs": []}
    checkpoints = 1  # Every 5 epochs save the model
    model_num = 60  # Parameter to save the model every 5 epochs

    #Loops over epochs:
    for epoch in range(start_epoch, epochs):

        model.train()
        totalTrainLoss = []
        totalValLoss = []
        trainCorrect = 0
        valCorrect = 0

        # Paths for files that will be saved in script
        path2saveCheckp = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/ADAM_Model/the_model_'+str(model_num)+'_checkp.pt'
        path2saveModel = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/ADAM_Model/the_model_'+str(model_num)+'.pt'
        path2saveCM = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/ADAM_Model/ConfusionMat_'+str(model_num)+'.png'
        path2saveROC = '/home/joy/Documents/Neuroscience Master/Neural Networks/CNN_project1/Model saved/100Epochs/ADAM_Model/ROC_'+str(model_num)+'.png'

        y_true = []
        y_pred = []
        y_pred_prob = []
        classes_names = ["Blank", "ICMS"]

        # loop over the training set
        for (x, y) in train_dl:
            x = x[:, None, :, :, :]  # Add fifth dimension - number of channels = 1 (gray scale image)
            x = x.type(torch.FloatTensor)

            # zero the parameter gradients
            opt.zero_grad()

            prob_pred = F.sigmoid(model(x))  # No sigmoid layer inside model
            train_loss = loss(prob_pred, y)

            # forward + backward + optimize
            train_loss.backward()
            opt.step()

            totalTrainLoss.append(train_loss)
            # calculate the number of correct predictions
            max_pred = prob_pred.argmax(1)
            trainCorrect += (max_pred == y).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the VALIDATION set
            for (x, y) in val_dl:
                x = x[:, None, :, :, :]
                x = x.type(torch.FloatTensor)
                # make the predictions and calculate the validation loss
                # pred_val = F.sigmoid(model(x))
                pred_val = model(x)  # sigmoid/softam inside model
                val_loss = F.cross_entropy(pred_val, y)
                totalValLoss.append(val_loss)
                # calculate the number of correct predictions
                pred_val_max = pred_val.argmax(1)
                valCorrect += (pred_val_max == y).type(torch.float).sum().item()

                # Params for confusion mat, ROC/AUC and precision-recall curves
                output = pred_val_max.data.cpu().numpy()
                max_prob_preds = pred_val.cpu().numpy()
                max_prob_preds = [item[1] for item in max_prob_preds]
                true_labels = y.data.cpu().numpy()
                y_true.extend(true_labels)
                y_pred.extend(output)
                y_pred_prob.extend(max_prob_preds)

        # calculate the average training and validation loss
        avgTrainLoss = torch.mean(stack(totalTrainLoss))
        avgValLoss = torch.mean(stack(totalValLoss))

        # calculate the training and validation accuracy
        trainlen = len(train_dl.dataset)
        valen = len(val_dl.dataset)
        trainAccuracy = trainCorrect / trainlen
        valAccuracy = valCorrect / valen

        # update our training history
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["train_acc"].append(trainAccuracy)
        history["val_loss"].append(avgValLoss.cpu().detach().numpy())
        history["val_acc"].append(valAccuracy)
        history["y_true"] = y_true
        history["y_pred"] = y_pred
        history["class1_probs"] = y_pred_prob

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainAccuracy))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valAccuracy))

        # CHECKPOINT: Save the model every 5 epochs:
        if model_num == (epoch+1):
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            }, path2saveCheckp)

            torch.save(model.state_dict(), path2saveModel)

            # Save training history:
            pickle.dump(history, open("TrainingHistoryVar_SGD_"+str(model_num)+".dat", "wb"))
            save_acc_loss_fig(history, model_num)
            # Build and plot confusion matrix for validation set (per epoch):
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

            model_num += checkpoints

    print("##########################################################################")

    return history
