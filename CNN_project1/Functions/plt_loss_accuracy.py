from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

optimizer = 'SGD'


def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])


cdict = {  'red':((0.0,inter_from_256(64),inter_from_256(64)),
           (1/5*1,inter_from_256(112),inter_from_256(112)),
           (1/5*2,inter_from_256(230),inter_from_256(230)),
           (1/5*3,inter_from_256(253),inter_from_256(253)),
           (1/5*4,inter_from_256(244),inter_from_256(244)),
           (1.0,inter_from_256(169),inter_from_256(169))),
    'green': ((0.0, inter_from_256(57), inter_from_256(57)),
            (1 / 5 * 1, inter_from_256(198), inter_from_256(198)),
            (1 / 5 * 2, inter_from_256(241), inter_from_256(241)),
            (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
            (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
            (1.0, inter_from_256(23), inter_from_256(23))),
    'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
              (1 / 5 * 1, inter_from_256(162), inter_from_256(162)),
              (1 / 5 * 2, inter_from_256(246), inter_from_256(146)),
              (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
              (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
              (1.0, inter_from_256(69), inter_from_256(69)))}

def plot_pred(history):
    today = date.today()
    d = today.strftime("%b-%d-%Y")

    # plot the training loss and accuracy
    plt.style.use("seaborn")
    f = plt.figure(figsize=(20,20))
    f.add_subplot(2,1,1)
    htrain_acc = history["train_acc"]
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.ylim([0.4, 1.01])
    plt.title("Training Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    f.add_subplot(2,1,2)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.ylim([0,1])
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("Accuracy_Loss_"+d+".jpg", bbox_inches="tight", pad_inches=0.3, transparent=True)
    plt.show()


def save_acc_loss_fig(history, model_num):
    today = date.today()
    d = today.strftime("%b-%d-%Y")

    # build and save the training loss and accuracy
    plt.style.use("seaborn")
    f = plt.figure(figsize=(20,20))
    f.add_subplot(2,1,1)
    htrain_acc = history["train_acc"]
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.ylim([0.4, 1.01])
    plt.title("Training Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    f.add_subplot(2,1,2)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.ylim([0,1])
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("Accuracy_Loss_"+d+'_epoch_'+str(model_num)+'_'+optimizer+".jpg", bbox_inches="tight", pad_inches=0.3, transparent=True)
    plt.close()


def plot_sample(image):
    """Plot all the 100x100 frames - show change in time"""
    s = list(image.shape)
    # frames = s[2]
    frames = s[0]
    f = plt.figure(figsize=(30, 30))
    # new_cmap = colors.LinearSegmentedColormap('new_cmap', segmentdata=cdict)

    for frame in range(frames):
        f.add_subplot(8, 8, frame + 1)
        # f.colorbar(cm.ScalarMappable(norm=colors.Normalize(), cmap=new_cmap))
        img = image[frame, :, :]
        # img = image[:, :, frame]
        plt.imshow(img)

        # plt.imshow(img, cmap='cubehelix')
        plt.axis('off')

    plt.show()


def plot_confusion_mat(conf_mat, name):
    plt.figure(figsize=(12,7))
    sn.heatmap(conf_mat, annot=True, cmap='YlGnBu')
    plt.savefig(name)

def plot_roc(fpr, tpr, p_fpr, p_tpr):
    plt.figure(figsize=(12,7))
    plt.style.use('seaborn')
    # plot roc curves
    plt.plot(fpr, tpr, linestyle='--',color='orange', label='CNN_ROC')
    # plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='Random_ROC')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC_model1.png', dpi=300)
    plt.show()


def build_confusion_mat(conf_mat, pathfile):
    plt.figure(figsize=(12,7))
    sn.heatmap(conf_mat, annot=True, cmap='YlGnBu')
    plt.savefig(pathfile)
    plt.close()

def build_roc(fpr, tpr, p_fpr, p_tpr, filepath):
    plt.figure(figsize=(14,7))
    plt.style.use('seaborn')
    # plot roc curves
    plt.plot(fpr, tpr, linestyle='--',color='orange', label='CNN_ROC')
    # plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(filepath, dpi=300)
    plt.close()
