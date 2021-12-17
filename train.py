import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from models.GCN_model import GCN
from models.MedianGCN_model import MedianGCN
from scipy import sparse

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g,features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def evaluate_attack(model, g, attack_g, features, attack_features, labels, mask, attack_nodes_index):
    model.eval()
    with torch.no_grad():
        logits = model(g,features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        test_acc = correct.item() * 1.0 / len(labels)#test_acc
        correat_attack_compare=torch.sum(indices[attack_nodes_index]==labels[attack_nodes_index])
        test_index_acc= correat_attack_compare.item() * 1.0 / len(attack_nodes_index)#test_index_acc


        attack_logits = model(attack_g,attack_features)
        attack_logits = attack_logits[mask]
        _, attack_indices = torch.max(attack_logits, dim=1)
        attack_correct = torch.sum(attack_indices == labels)
        attack_test_acc=attack_correct.item() * 1.0 / len(labels)#attack_test_acc

        correct_attack=torch.sum(attack_indices[attack_nodes_index]==labels[attack_nodes_index])
        attack_test_index_acc= correct_attack.item() * 1.0 / len(attack_nodes_index)#attack_test_index_acc

        return test_acc,test_index_acc,attack_test_acc,attack_test_index_acc


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu!=-1:
        torch.cuda.manual_seed(args.seed)


    
    if args.dataset == 'cora':
        data=np.load('./data_clean/cora_processed.npz',allow_pickle=True)
    elif args.dataset == 'citeseer':
        data=np.load('./data_clean/citeseer_processed.npz',allow_pickle=True)
    elif args.dataset == 'pubmed':
        data=np.load('./data_clean/pubmed_processed.npz',allow_pickle=True)
    elif args.dataset == 'cora_full':
        data=np.load('./data_clean/cora_full_processed.npz',allow_pickle=True)
    elif args.dataset == 'polblogs':
        data=np.load('./data_clean/polblogs_processed.npz',allow_pickle=True)
        
    attack_nodes_index=[]

    if args.attack == True:
        if args.dataset == 'cora':
            data_attack=np.load('./data_attack/my_attack_data/'+args.attack_data_name+'.npz',allow_pickle=True)
        elif args.dataset == 'citeseer':
            data_attack=np.load('./data_attack/my_attack_data/'+args.attack_data_name+'.npz',allow_pickle=True)
        elif args.dataset == 'pubmed':
            data_attack=np.load('./data_attack/my_attack_data/'+args.attack_data_name+'.npz',allow_pickle=True)
        elif args.dataset == 'cora_full':
            data_attack=np.load('./data_attack/my_attack_data/'+args.attack_data_name+'.npz',allow_pickle=True)
        elif args.dataset == 'polblogs':
            data_attack=np.load('./data_attack/my_attack_data/'+args.attack_data_name+'.npz',allow_pickle=True)
        
        attack_nodes=data_attack['perturbanced_node']
        attack_nodes_index=data_attack['perturbanced_node_index']

        np.savez('./attack_node_index/'+args.attack_data_name+'_index.npz',attack_nodes_index_for_compare=attack_nodes_index)
    
    

    _A_obs=sparse.csr_matrix(data['_A_obs'])
    g=dgl.from_scipy(_A_obs)
    if args.attack==True:
        attack_A_obs=sparse.csr_matrix(data_attack['_A_obs'])
        attack_g=dgl.from_scipy(attack_A_obs)


    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
        if args.attack==True:
            attack_g = attack_g.int().to(args.gpu)

    features=torch.tensor(data['_X_obs'])
    labels=torch.tensor(data['_z_obs']).long()
    train_mask=torch.tensor(data['train_mask'])
    val_mask=torch.tensor(data['val_mask'])
    test_mask=torch.tensor(data['test_mask'])
    in_feats = features.shape[1]
    n_classes = np.array(labels).max()+1
    n_edges = g.number_of_edges()
    if args.attack==True:
        attack_features=torch.tensor(data_attack['_X_obs'])


    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    if args.model == 'GCN':
        model = GCN(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    elif args.model == 'MedianGCN':
        model = MedianGCN(in_feats,
                        n_classes,
                        hids=[16],
                        acts=['relu'],
                        dropout=0.5,
                        bias=True)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g,features)

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.attack==False:
        acc = evaluate(model, g, features, labels, test_mask)
        print("Test accuracy {:.2%}".format(acc))
    elif args.attack==True:
        test_acc,test_index_acc,attack_test_acc,attack_test_index_acc=evaluate_attack(model, g, attack_g, features, attack_features, labels, test_mask, attack_nodes_index)
        print("Test accuracy {:.2%}".format(test_acc))
        print("Test index accuracy {:.2%}".format(test_index_acc))
        print("Attack Test accuracy {:.2%}".format(attack_test_acc))
        print("Attack Test index accuracy {:.2%}".format(attack_test_index_acc))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name.")
    parser.add_argument("--model", type=str, default="GCN",
                        help="Dataset name ('GCN', 'MedianGCN').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--seed", type=int,
                        help="random seed",default=125)
    parser.add_argument("--attack", type=bool,
                        help="use attack data or not",default=False)

    parser.add_argument("--attack_data_name", type=str,
                        help="use what way to attack data",default='')
    
    
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)