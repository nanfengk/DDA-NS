import argparse
import os
import os.path as osp
from torch.nn import BCEWithLogitsLoss
from creat_dataset import MyOwnDataset
from utils import *
from models import *
import time
import datetime
from matplotlib import pyplot as plt

start_time = datetime.datetime.now()
time1 = datetime.datetime.strftime(start_time,'%Y-%m-%d %H:%M:%S')
print("Program start time："+str(time1))
splilt_line='------------------------------------------------------------------------------------------------------'
print(splilt_line)

def train():
    model.train()

    total_loss = 0
    train_loss = 0
    y_pred, y_true = [], []
    # pbar = tqdm(train_loader, ncols=70)
    for data in iter(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        train_loss=total_loss / len(train_dataset)
    train_pred, train_true = torch.cat(y_pred), torch.cat(y_true)
    train_metrics = evaluate_metrics(train_true.detach(), train_pred.detach())

    return train_loss, train_metrics


@torch.no_grad()
def test():
    model.eval()
    total_loss = 0
    valid_loss = 0
    y_pred, y_true = [], []
    for data in iter(valid_loader):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        valid_loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        y_pred.append(logits.view(-1).cpu().sigmoid())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
        total_loss += valid_loss.item() * data.num_graphs
        valid_loss=total_loss / len(valid_dataset)
    valid_pred, valid_true = torch.cat(y_pred), torch.cat(y_true)
    valid_metrics = evaluate_metrics(valid_true, valid_pred)
    return valid_loss, valid_metrics

# Data settings
parser = argparse.ArgumentParser(description='NSNP')
parser.add_argument('--dataset', type=str)
# GNN settings
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl',
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true',
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--dynamic_train', action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--num_workers', type=int, default=16,
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='',
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='',
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true',
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None,
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', action='store_true',
                    help="only test without training")
parser.add_argument('--test_multiple_models', action='store_true',
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")
args = parser.parse_args()

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d_%H_%M_%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'

args.res_dir = osp.join('../results/{}{}'.format(args.dataset, args.save_appendix))
# print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

log_file = osp.join(args.res_dir, 'log.txt')
metrics_file = osp.join(args.res_dir, 'metrics.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(log_file, 'a') as f:
    f.write(cmd_input)

path = osp.join('../dataset', args.dataset)
dataset = MyOwnDataset(path)
data = dataset[0]


train_auc=[]
valid_auc=[]


train_metrics_sum = {}
valid_metrics_sum = {}




x_list=[]
for i in range(1,args.epochs+1):
    x_list.append(i)



for flod_num in range(0,10):

    print('Training Fold ', flod_num + 1)
    split_edge = do_edge_split(dataset, flod_num)
    data.edge_index = split_edge['train']['edge'].t()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SEAL.
    path = dataset.root + '_seal{}'.format(args.data_appendix)
    use_coalesce = True if args.dataset == 'ogbl-collab' else False
    if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
        args.num_workers = 0

    train_dataset = DynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        percent=args.train_percent,
        split='train',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        flod_num=flod_num,
    )

    valid_dataset = DynamicDataset(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        percent=args.val_percent,
        split='valid',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        flod_num=flod_num,
    )

    max_z = 1000  # set a large max_z so that every z has embeddings to look up

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    if args.train_node_embedding:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad=False
    else:
        emb = None


    if args.model == 'DGCNN':
        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
                      train_dataset, args.dynamic_train, use_feature=args.use_feature,
                      node_embedding=emb).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args.hidden_channels, args.num_layers, max_z, train_dataset,
                     args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset,
                    args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'GIN':
        model = GIN(args.hidden_channels, args.num_layers, max_z, train_dataset,
                    args.use_feature, node_embedding=emb).to(device)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    # total_params = sum(p.numel() for param in parameters for p in param)
    # print(f'Total number of parameters is {total_params}')

    start_epoch = 1

    train_metrics_list = []
    valid_metrics_list = []

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_metrics = train()
        valid_loss, valid_metrics = test()
        # AUC, AUPR, F1 score, Accuracy, Recall, Specificity, Precision, MCC, fpr, tpr, recall_list, precision_list
        train_metrics_list.append(train_metrics)
        valid_metrics_list.append(valid_metrics)

        to_print =(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, '+
                   f'Train AUC: {train_metrics[0]:.4f}, Valid AUC: {valid_metrics[0]:.4f}, '+
                   f'AUPR: {valid_metrics[1]:.4f}, F1 score: {valid_metrics[2]:.4f}, ' +
                   f'Accuracy: {valid_metrics[3]:.4f}, Recall: {valid_metrics[4]:.4f}, ' +
                   f'Specificity: {valid_metrics[5]:.4f}, Precision: {valid_metrics[6]:.4f}, ' +
                   f'MCC: {valid_metrics[7]:.4f}')
        print_save(to_print, log_file)
    print_save(splilt_line, log_file)

    train_metrics_sum[str('metrics' + str(flod_num+1))] = train_metrics_list
    valid_metrics_sum[str('metrics' + str(flod_num+1))] = valid_metrics_list

auc_list=[]
aupr_list=[]
f1_list=[]
acc_list=[]
rec_list=[]
spe_list=[]
pre_list=[]
mcc_list=[]
fprs_list = []
tprs_list = []
recall_list = []
precision_list = []
for keys in valid_metrics_sum:
    auc = 0
    aupr = 0
    f1 = 0
    acc = 0
    rec = 0
    spe = 0
    pre = 0
    mcc = 0
    fprs = []
    tprs = []
    recall = []
    precision = []
    valid_result=valid_metrics_sum[keys]
    for metrics in valid_result:
        if metrics[0]>auc:
            auc=metrics[0]
            aupr=metrics[1]
            f1=metrics[2]
            acc=metrics[3]
            rec=metrics[4]
            spe=metrics[5]
            pre=metrics[6]
            mcc=metrics[7]
            fprs=metrics[8]
            tprs=metrics[9]
            recall=metrics[10]
            precision = metrics[11]

    auc_list.append(auc)
    aupr_list.append(aupr)
    f1_list.append(f1)
    acc_list.append(acc)
    rec_list.append(rec)
    spe_list.append(spe)
    pre_list.append(pre)
    mcc_list.append(mcc)
    fprs_list.append(fprs)
    tprs_list.append(tprs)
    recall_list.append(recall)
    precision_list.append(precision)

np.savetxt(osp.join(args.res_dir,"auc_list.txt"), auc_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"aupr_list.txt"), aupr_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"f1_list.txt"), f1_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"acc_list.txt"), acc_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"rec_list.txt"), rec_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"spe_list.txt"), spe_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"pre_list.txt"), pre_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"mcc_list.txt"), mcc_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"fprs_list.txt"), fprs_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"tprs_list.txt"), tprs_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"recall_list.txt"), recall_list, fmt='%.15f')
np.savetxt(osp.join(args.res_dir,"precision_list.txt"), precision_list, fmt='%.15f')

# test=np.loadtxt(osp.join(args.res_dir,"auc_list.txt"))

to_print = (f'AUC: {np.mean(auc_list):.4f}, AUPR: {np.mean(aupr_list):.4f}, '
            f'F1 score: {np.mean(f1_list):.4f}, Accuracy: {np.mean(acc_list):.4f}, '
            f'Recall: {np.mean(rec_list):.4f}, Specificity: {np.mean(spe_list):.4f}, '
            f'Precision: {np.mean(pre_list):.4f}, MCC: {np.mean(mcc_list):.4f}')
print_save(to_print, log_file)

end_time = datetime.datetime.now()
time2 = datetime.datetime.strftime(end_time,'%Y-%m-%d %H:%M:%S')
print("Program end time："+str(time2))
running_time=end_time - start_time
print('Program run time:'+str(running_time))
