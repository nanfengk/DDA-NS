from matplotlib import pyplot as plt
import numpy as np
fpr1=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.2/fpr_auc_0.9042658831440612.txt')
tpr1=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.2/tpr_auc_0.9042658831440612.txt')
fpr2=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.4/fpr_auc_0.9243440180694527.txt')
tpr2=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.4/tpr_auc_0.9243440180694527.txt')
fpr3=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.6/fpr_auc_0.9335502471465988.txt')
tpr3=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.6/tpr_auc_0.9335502471465988.txt')
fpr4=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.8/fpr_auc_0.9424936521173233.txt')
tpr4=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.8/tpr_auc_0.9424936521173233.txt')
fpr5=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_1.0/fpr_auc_0.9535826157218559.txt')
tpr5=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_1.0/tpr_auc_0.9535826157218559.txt')

recall1=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.2/recall_aupr_0.9096856024295152.txt')
precision1=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.2/precision_aupr_0.9096856024295152.txt')
recall2=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.4/recall_aupr_0.9273677893831087.txt')
precision2=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.4/precision_aupr_0.9273677893831087.txt')
recall3=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.6/recall_aupr_0.9359964889651877.txt')
precision3=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.6/precision_aupr_0.9359964889651877.txt')
recall4=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.8/recall_aupr_0.9444344123899115.txt')
precision4=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_0.8/precision_aupr_0.9444344123899115.txt')
recall5=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_1.0/recall_aupr_0.951937786087643.txt')
precision5=np.loadtxt('C:/Users/Administrator/Desktop/plot/plot_1.0/precision_aupr_0.951937786087643.txt')

plt.figure()
lw = 2
plt.plot(
    fpr1,
    tpr1,
    # color="r",
    lw=lw,
    label="  20%% Data size (AUC = %0.4f)" % 0.9043,
)
plt.plot(
    fpr2,
    tpr2,
    # color="darkorange",
    lw=lw,
    label="  40%% Data size (AUC = %0.4f)" % 0.9243,
)
plt.plot(
    fpr3,
    tpr3,
    # color="darkviolet",
    lw=lw,
    label="  60%% Data size (AUC = %0.4f)" % 0.9360,
)
plt.plot(
    fpr4,
    tpr4,
    # color="darkviolet",
    lw=lw,
    label="  80%% Data size (AUC = %0.4f)" % 0.9425,
)
plt.plot(
    fpr5,
    tpr5,
    # color="darkviolet",
    lw=lw,
    label="100%% Data size (AUC = %0.4f)" % 0.9536,
)
plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
# plt.title("Receiver operating characteristic curve")
plt.legend(loc="lower right")
plt.savefig("AUC.svg", dpi=300)
plt.show()

plt.figure()
lw = 2
plt.plot(
    recall1,
    precision1,
    # color="r",
    lw=lw,
    label="  20%% Data size (AUPR = %0.4f)" % 0.9097,
)
plt.plot(
    recall2,
    precision2,
    # color="darkorange",
    lw=lw,
    label="  40%% Data size (AUPR = %0.4f)" % 0.9274,
)
plt.plot(
    recall3,
    precision3,
    # color="darkviolet",
    lw=lw,
    label="  60%% Data size (AUPR = %0.4f)" % 0.9336,
)
plt.plot(
    recall4,
    precision4,
    # color="darkviolet",
    lw=lw,
    label="  80%% Data size (AUPR = %0.4f)" % 0.9444,
)
plt.plot(
    recall5,
    precision5,
    # color="darkviolet",
    lw=lw,
    label="100%% Data size (AUPR = %0.4f)" % 0.9519,
)
plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
# plt.title("receiver operating characteristic curve")
plt.legend(loc="lower right")
plt.savefig("AUPR.svg", dpi=300)
plt.show()
