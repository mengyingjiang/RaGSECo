from numpy.random import seed
import os
import gc
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from radam import RAdam
from model import *
from accuracy import *
from utils import *
from layers import *
import warnings
warnings.filterwarnings("ignore")


seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class DDIDataset(Dataset):  # [59622, 3432], [59622]
    def __init__(self, y, z):
        self.len = len(y)
        self.y_data = torch.from_numpy(y).to(device)
        self.z_data = torch.from_numpy(z).to(device)

    def __getitem__(self, index):
        return self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.len

def RaSECo_train(model, y_train,  y_test, event_num, X_vector, adj, drug_intera_fea, ddi_edge_train,ddi_edge_test, multi_graph ,drug_coding,args):  # model [29811,3432],[29811],[7453,3432],[7453],65

    model_optimizer = RAdam(model.parameters(), lr= args.learn_rating, weight_decay=args.weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    ddi_edge_train = np.vstack((ddi_edge_train,np.vstack((ddi_edge_train[:,1],ddi_edge_train[:,0])).T))
    y_train = np.hstack((y_train, y_train))

    N_edges = ddi_edge_train.shape
    index = np.arange(N_edges[0])
    np.random.seed(seed)
    np.random.shuffle(index)

    y_train = y_train[index]
    ddi_edge_train = ddi_edge_train[index]

    len_train = len(y_train)
    len_test = len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))

    train_dataset = DDIDataset(ddi_edge_train, np.array(y_train))  # [59622, 3432], [59622]
    test_dataset = DDIDataset(ddi_edge_test, np.array(y_test))  # [7453, 3432], [7453]

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    multi_graph = multi_graph.to(device)
    drug_intera_fea = drug_intera_fea.to(device)
    for epoch in range(args.epo_num):
        my_loss = my_loss1()
        running_loss = 0.0
        x_vector = X_vector.to(device)
        drug_coding = drug_coding.to(device)
        adj = adj.to(device)

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            train_edge, train_edge_labels = data
            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(train_edge.size()[0]).to(device)
            targets_a, targets_b = train_edge_labels, train_edge_labels[index]
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            train_edge = torch.tensor(train_edge, dtype=torch.long)
            train_edge = train_edge.to(device)
            train_edge_mixup = train_edge[index, :]
            model_optimizer.zero_grad()

            X,  ss, n_posi, n_neg = model(multi_graph,adj, drug_intera_fea, x_vector, train_edge, train_edge_mixup, lam, drug_coding )

            lbl = torch.cat((torch.ones(n_posi*2,1), torch.zeros(n_neg*2,1)), 0).to(device)
            loss = lam * my_loss(X, targets_a, ss , lbl) \
                   + (1 - lam) * my_loss(X, targets_b, lbl, lbl)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()
            del X,targets_a,targets_b,train_edge,loss
            gc.collect()
        # 循环完 该epoch的训练结束
        model.eval()
        testing_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader, 0):
                test_edge, test_edge_labels = data
                test_edge = torch.tensor(test_edge, dtype=torch.long)
                test_edge = test_edge.to(device)
                lam = 1
                test_edge_labels = test_edge_labels.type(torch.int64).to(device)
                X, _, _, _ = model(multi_graph, adj, drug_intera_fea, x_vector, test_edge,test_edge,lam, drug_coding )
                loss = torch.nn.functional.cross_entropy(X, test_edge_labels)
                testing_loss += loss.item()

            del test_edge
            gc.collect()

        print('epoch [%d] trn_los: %.6f tet_los: %.6f' % (
            epoch + 1, running_loss / len_train, testing_loss / len_test))

    pre_score = np.zeros((0, event_num), dtype=float)

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            test_edge, _ = data
            test_edge = torch.tensor(test_edge, dtype=torch.long)
            test_edge = test_edge.to(device)
            lam = 1
            X, _ , _, _= model(multi_graph, adj, drug_intera_fea, x_vector, test_edge, test_edge,lam, drug_coding )
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))


    del model
    del X
    del model_optimizer
    del train_loader
    del test_loader
    del train_dataset
    del test_dataset
    gc.collect()

    return pre_score
