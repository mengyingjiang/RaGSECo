from layers import *
import torch
import torch.nn.functional as F

class RaSECo(torch.nn.Module):
    def __init__(self, initial_dim, event_num,args):
        super( RaSECo, self).__init__()
        self.initial_dim = initial_dim
        graph_dim = args.graph_dim
        n_hop = args.n_hop
        dim_EN = args.dim_EN
        drop_out_rating = args.drop_out_rating
        self.RGCN1 = RGCN(event_num+1, initial_dim//2, graph_dim)
        self.GCN1 = HomoGCN(graph_dim, graph_dim,n_hop)
        self.GCN2 = HomoGCN(graph_dim, graph_dim,n_hop)
        self.GCN3 = HomoGCN(graph_dim, graph_dim,n_hop)

        self.disc = Discriminator_dd(dim_EN)

        self.FNN1 = FNN(graph_dim*2, dim_EN, args)  # Joining together
        self.FNN2 = FNN(initial_dim, dim_EN, args)  # Joining together
        self.cnn_concat = CNN_concat(dim_EN, 'drug', **config)
        self.dr = torch.nn.Dropout(drop_out_rating)
        N_Fea = dim_EN*4
        self.l1 = torch.nn.Linear(N_Fea, (N_Fea + event_num))
        self.bn1 = torch.nn.BatchNorm1d((N_Fea + event_num))
        self.l2 = torch.nn.Linear((N_Fea + event_num), event_num)
        self.ac = gelu

    def forward(self, multi_graph, label_graph, drug_intera_fea, x_initial, ddi_edge, ddi_edge_mixup, lam, drug_coding ):

        x_known = F.relu(self.RGCN1(x_initial,label_graph))
        x_tar_all = F.relu(self.dr(self.GCN1(x_known, multi_graph[0, :, :])))
        x_enzy_all = F.relu(self.dr(self.GCN2(x_known, multi_graph[1, :, :])))
        x_sub_all = F.relu(self.dr(self.GCN3(x_known, multi_graph[2, :, :])))

        x_embed_all = x_tar_all+ x_enzy_all+x_sub_all

        node_id = ddi_edge.T
        node_id_mixup = ddi_edge_mixup.T
        X_smile = lam * torch.cat([drug_coding[node_id[0]], drug_coding[node_id[1]]], dim=2) \
                + (1 - lam) * torch.cat([drug_coding[node_id_mixup[0]], drug_coding[node_id_mixup[1]]], dim=2)

        x_embed_all = lam * torch.cat([x_embed_all[node_id[0]], x_embed_all[node_id[1]]], dim=1) \
                    + (1 - lam) * torch.cat([x_embed_all[node_id_mixup[0]], x_embed_all[node_id_mixup[1]]], dim=1)

        x_initial = lam * torch.cat([x_initial[node_id[0]], x_initial[node_id[1]]], dim=1) \
                  + (1 - lam) * torch.cat([x_initial[node_id_mixup[0]], x_initial[node_id_mixup[1]]], dim=1)

        x_inter_fea = drug_intera_fea[node_id[0]] + drug_intera_fea[node_id[1]]

        row, clomn, row_neg, clomn_neg, n_posi, n_neg = posi_neg_gene(x_inter_fea)

        X_smile = self.cnn_concat(X_smile)

        x_embed_all = self.FNN1(x_embed_all)

        x_initial = self.FNN2(x_initial)

        ss  = self.disc(x_initial, x_embed_all, row, clomn, row_neg, clomn_neg, n_posi, n_neg)

        X = torch.cat((x_initial, x_embed_all, X_smile, x_initial+x_embed_all+X_smile), 1)
        X = self.dr(F.relu(self.l1(X)))
        X = self.l2(X)

        return X,  ss, n_posi, n_neg

