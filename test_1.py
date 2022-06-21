#
# # for i in range(1,60):
# #     with open('C:\\Users\\Administrator\\Desktop\\SEAL-master\\Python\\data\\123.txt', 'a') as f:
# #         f.write(str(i)+","+str(i+100)+"\n")
#
#
# import math
# import random
# from torch_geometric.utils import to_undirected
# import torch
# from creat_dataset import MyOwnDataset
# from torch_geometric.utils import (negative_sampling, add_self_loops,
#                                    train_test_split_edges)
# import numpy as np
# from torch_geometric.data import Data
#
#
#
# def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
#     data = dataset[0]
#     random.seed(234)
#     torch.manual_seed(234)
#
#     if not fast_split:
#         num_nodes = data.num_nodes
#         row, col = data.edge_index
#         len_r = row[np.argmax(row)]
#         len_c= col[np.argmax(col)]
#         data.edge_index = None
#
#         # Return upper triangular portion.
#         mask = row < col
#         row, col = row[mask], col[mask]
#
#         n_v = int(math.floor(val_ratio * row.size(0)))
#         n_t = int(math.floor(test_ratio * row.size(0)))
#
#         # Positive edges.
#         perm = torch.randperm(row.size(0))
#         row, col = row[perm], col[perm]
#
#         r, c = row[:n_v], col[:n_v]
#         data.val_pos_edge_index = torch.stack([r, c], dim=0)
#
#
#         r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
#         data.test_pos_edge_index = torch.stack([r, c], dim=0)
#
#
#         r, c = row[n_v + n_t:], col[n_v + n_t:]
#         data.train_pos_edge_index = torch.stack([r, c], dim=0)
#         data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
#
#         # Negative edges.
#
#         val_size = data.val_pos_edge_index.size(1)
#         val_neg = [[], []]
#         while len(val_neg[0]) < val_size:
#             i, j = random.randint(0, len_r), random.randint(len_r + 1, len_c - 1)
#             val_neg[0].append(i)
#             val_neg[1].append(j)
#         data.val_neg_edge_index = torch.tensor(val_neg, dtype=torch.long)
#
#
#         test_size = data.test_pos_edge_index.size(1)
#         test_neg = [[], []]
#         while len(test_neg[0]) < test_size:
#             i, j = random.randint(0, len_r), random.randint(len_r + 1, len_c - 1)
#             test_neg[0].append(i)
#             test_neg[1].append(j)
#         data.test_neg_edge_index = torch.tensor(test_neg, dtype=torch.long)
#
#
#         train_size = data.train_pos_edge_index.size(1)
#         train_neg = [[], []]
#         while len(train_neg[0]) < train_size:
#             i, j = random.randint(0, len_r), random.randint(len_r + 1, len_c - 1)
#             train_neg[0].append(i)
#             train_neg[1].append(j)
#         data.train_neg_edge_index = torch.tensor(train_neg, dtype=torch.long)
#
#
#     else:
#         num_nodes = data.num_nodes
#         row, col = data.edge_index
#         # Return upper triangular portion.
#         mask = row < col
#         row, col = row[mask], col[mask]
#         n_v = int(math.floor(val_ratio * row.size(0)))
#         n_t = int(math.floor(test_ratio * row.size(0)))
#         # Positive edges.
#         perm = torch.randperm(row.size(0))
#         row, col = row[perm], col[perm]
#         r, c = row[:n_v], col[:n_v]
#         data.val_pos_edge_index = torch.stack([r, c], dim=0)
#         r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
#         data.test_pos_edge_index = torch.stack([r, c], dim=0)
#         r, c = row[n_v + n_t:], col[n_v + n_t:]
#         data.train_pos_edge_index = torch.stack([r, c], dim=0)
#         # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
#         neg_edge_index = negative_sampling(
#             data.edge_index, num_nodes=num_nodes,
#             num_neg_samples=row.size(0))
#         data.val_neg_edge_index = neg_edge_index[:, :n_v]
#         data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
#         data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]
#
#     split_edge = {'train': {}, 'valid': {}, 'test': {}}
#     split_edge['train']['edge'] = data.train_pos_edge_index.t()
#     split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
#     split_edge['valid']['edge'] = data.val_pos_edge_index.t()
#     split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
#     split_edge['test']['edge'] = data.test_pos_edge_index.t()
#     split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
#     return split_edge
#
#
#
# # arr = np.loadtxt('C:\\Users\\Administrator\\Desktop\\SEAL-master\\Python\\data\\123.txt', delimiter=",", dtype=np.int64)
# # net=np.transpose(arr)
# # edge_index = torch.tensor(net, dtype=torch.long)
# # data = Data(edge_index=edge_index)
#
#
# dataset = MyOwnDataset("dataset\\test_data")
# split_edge = do_edge_split(dataset, fast_split=False)
#
# print()
#
#
#


#
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn import svm, datasets
# from sklearn.metrics import auc
# from sklearn.metrics import RocCurveDisplay
# from sklearn.model_selection import StratifiedKFold
#
# # #############################################################################
# # Data IO and generation
#
# # Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X, y = X[y != 2], y[y != 2]
# n_samples, n_features = X.shape
#
# # Add noisy features
# random_state = np.random.RandomState(0)
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#
# # #############################################################################
# # Classification and ROC analysis
#
# # Run classifier with cross-validation and plot ROC curves
# cv = StratifiedKFold(n_splits=6)
# classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
#
# fig, ax = plt.subplots()
# for i, (train, test) in enumerate(cv.split(X, y)):
#     classifier.fit(X[train], y[train])
#     viz = RocCurveDisplay.from_estimator(
#         classifier,
#         X[test],
#         y[test],
#         name="ROC fold {}".format(i),
#         alpha=0.3,
#         lw=1,
#         ax=ax,
#     )
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)
#
# ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
# plt.show()
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#     lw=2,
#     alpha=0.8,
# )
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )
#
# ax.set(
#     xlim=[-0.05, 1.05],
#     ylim=[-0.05, 1.05],
#     title="Receiver operating characteristic example",
# )
# ax.legend(loc="lower right")
# plt.show()
#

from matplotlib import pyplot as plt
import numpy as np

auc_list=np.loadtxt('results/my_data_20220516_17_28_01/auc_list.txt')
fpr_list=np.loadtxt('results/my_data_20220516_17_28_01/fprs_list.txt')
tpr_list=np.loadtxt('results/my_data_20220516_17_28_01/tprs_list.txt')

aupr_list=np.loadtxt('results/my_data_20220516_17_28_01/aupr_list.txt')
recall_list=np.loadtxt('results/my_data_20220516_17_28_01/recall_list.txt')
precision_list=np.loadtxt('results/my_data_20220516_17_28_01/precision_list.txt')

plt.figure()
lw = 2
plt.plot(
    fpr_list[0],
    tpr_list[0],
    # color="r",
    lw=lw,
    label="flod1 AUC = %0.4f" % auc_list[0],
)
plt.plot(
    fpr_list[1],
    tpr_list[1],
    # color="r",
    lw=lw,
    label="flod2 AUC = %0.4f" % auc_list[1],
)
plt.plot(
    fpr_list[2],
    tpr_list[2],
    # color="r",
    lw=lw,
    label="flod3 AUC = %0.4f" % auc_list[2],
)
plt.plot(
    fpr_list[3],
    tpr_list[3],
    # color="r",
    lw=lw,
    label="flod4 AUC = %0.4f" % auc_list[3],
)
plt.plot(
    fpr_list[4],
    tpr_list[4],
    # color="r",
    lw=lw,
    label="flod5 AUC = %0.4f" % auc_list[4],
)
plt.plot(
    fpr_list[5],
    tpr_list[5],
    # color="r",
    lw=lw,
    label="flod6 AUC = %0.4f" % auc_list[5],
)
plt.plot(
    fpr_list[6],
    tpr_list[6],
    # color="r",
    lw=lw,
    label="flod7 AUC = %0.4f" % auc_list[6],
)
plt.plot(
    fpr_list[7],
    tpr_list[7],
    # color="r",
    lw=lw,
    label="flod8 AUC = %0.4f" % auc_list[7],
)
plt.plot(
    fpr_list[8],
    tpr_list[8],
    # color="r",
    lw=lw,
    label="flod9 AUC = %0.4f" % auc_list[8],
)
plt.plot(
    fpr_list[9],
    tpr_list[9],
    color="r",
    lw=lw,
    label="flod10 AUC = %0.4f" % auc_list[9],
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
    recall_list[0],
    precision_list[0],
    # color="r",
    lw=lw,
    label="flod1 AUPR = %0.4f" % aupr_list[0],
)
plt.plot(
    recall_list[1],
    precision_list[1],
    # color="r",
    lw=lw,
    label="flod2 AUPR = %0.4f" % aupr_list[1],
)
plt.plot(
    recall_list[2],
    precision_list[2],
    # color="r",
    lw=lw,
    label="flod3 AUPR = %0.4f" % aupr_list[2],
)
plt.plot(
    recall_list[3],
    precision_list[3],
    # color="r",
    lw=lw,
    label="flod4 AUPR = %0.4f" % aupr_list[3],
)
plt.plot(
    recall_list[4],
    precision_list[4],
    # color="r",
    lw=lw,
    label="flod5 AUPR = %0.4f" % aupr_list[4],
)
plt.plot(
    recall_list[5],
    precision_list[5],
    # color="r",
    lw=lw,
    label="flod6 AUPR = %0.4f" % aupr_list[5],
)
plt.plot(
    recall_list[6],
    precision_list[6],
    # color="r",
    lw=lw,
    label="flod7 AUPR = %0.4f" % aupr_list[6],
)
plt.plot(
    recall_list[7],
    precision_list[7],
    # color="r",
    lw=lw,
    label="flod8 AUPR = %0.4f" % aupr_list[7],
)
plt.plot(
    recall_list[8],
    precision_list[8],
    # color="r",
    lw=lw,
    label="flod9 AUPR = %0.4f" % aupr_list[8],
)
plt.plot(
    recall_list[9],
    precision_list[9],
    color="r",
    lw=lw,
    label="flod10 AUPR = %0.4f" % aupr_list[9],
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







