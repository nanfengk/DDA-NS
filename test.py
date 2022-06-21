# import math
# import random
# import scipy.io as sio
# import scipy.sparse as ssp
# import torch
# from creat_dataset import MyOwnDataset
# from torch_geometric.utils import (negative_sampling, add_self_loops,
#                                    train_test_split_edges)
# import numpy as np
# from torch_geometric.data import Data
#
# def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
#     data = dataset[0]
#     random.seed(234)
#     torch.manual_seed(234)
#
#     if not fast_split:
#         data = train_test_split_edges(data, val_ratio, test_ratio)
#         edge_index, _ = add_self_loops(data.train_pos_edge_index)
#         data.train_neg_edge_index = negative_sampling(
#             edge_index, num_nodes=data.num_nodes,
#             num_neg_samples=data.train_pos_edge_index.size(1))
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



# arr = np.loadtxt('C:\\Users\\Administrator\\Desktop\\SEAL-master\\Python\\data\\123.txt', delimiter=",", dtype=np.int64)
# net=np.transpose(arr)
# edge_index = torch.tensor(net, dtype=torch.long)
# data = Data(edge_index=edge_index)


# dataset = MyOwnDataset("dataset\\test_data")
# split_edge = do_edge_split(dataset, fast_split=False)
#
# print()



