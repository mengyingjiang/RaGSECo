import numpy as np
import pandas as pd
from pandas import DataFrame
import csv
import torch
import copy
import dgl
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y','@']

MAX_SEQ_DRUG = 100
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    d_label = {}
    d_feature = {}
    drug_index = {}
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    tensor_tempvec_multi = torch.tensor([])
    N_drugs = df_drug.shape[0]
    for i in feature_list:
        tempvec, df_feature = feature_vector(i, df_drug)
        tensor_tempvec = torch.tensor(tempvec)
        vector = np.hstack((vector, tempvec))
        adj1 = gcnnormalization(tensor_tempvec)
        adj = adj1.reshape(1, N_drugs, N_drugs)
        tensor_tempvec_multi = torch.cat((tensor_tempvec_multi, adj), 0)

    X_vector = copy.deepcopy(vector)

    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]
        drug_index[np.array(df_drug['name']).tolist()[i]] = i

    print('unique drugs: ' + str(len(df_drug['name'].unique())))

    new_feature = []
    new_label = []
    DDI_edge = np.zeros((len(d_event), 2))
    X_vector = torch.tensor(X_vector, dtype=torch.float32)

    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        DDI_edge[i][0] = drug_index[drugA[i]]
        DDI_edge[i][1] = drug_index[drugB[i]]
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_label = np.array(new_label)  # 323539
    DDI_edge = np.array(DDI_edge)

    seed = 0
    index = np.arange(DDI_edge.shape[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    new_label = new_label[index]
    DDI_edge = DDI_edge[index]
    A=(drugA[index]).values
    B=(drugB[index]).values
    drugA = pd.Series(A)
    drugB = pd.Series(B)

    f = torch.isnan(tensor_tempvec_multi)
    tensor_tempvec_multi[f] = 0
    tensor_tempvec_multi = tensor_tempvec_multi.to(torch.float32)
    return  new_label, drugA, drugB, event_num, X_vector, DDI_edge, tensor_tempvec_multi

def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix, df_feature

def save_result(filepath,result_type,result,task):
    if task == "task1":
        with open(filepath + result_type + 'task1' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in result:
                writer.writerow(i)
    if task == "task2":
        with open(filepath + result_type + 'task2' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in result:
                writer.writerow(i)
    if task == "task3":
        with open(filepath + result_type + 'task3' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in result:
                writer.writerow(i)
    return 0

def adj_Heter_gene(DDI_edge,X_vector,event_num,new_label):
    edge_labels = torch.tensor(new_label + 1, dtype=torch.int64)
    N = X_vector.size()[0]
    shape = torch.Size([N, N])
    # values = torch.ones([DDI_edge.size()[0]])
    edges = torch.tensor(DDI_edge.T, dtype=torch.float)
    edges = torch.cat((edges, torch.vstack((edges[1], edges[0]))), 1)
    edge_labels = torch.cat((edge_labels,edge_labels),0)
    adj = torch.sparse_coo_tensor(edges, edge_labels, shape)
    adj = adj.to_dense()
    adj_Heter = torch.zeros(event_num+1,N,N)
    num_drug_labels = torch.ones(N)
    drug_intera_fea = torch.zeros(adj.shape[0], event_num+1)
    adj[adj>event_num]=0

    for i in range(N):
        drug_labels= torch.unique(adj[i])
        num_drug_labels[i] = drug_labels.size()[0]+1
        drug_labels = torch.tensor(drug_labels, dtype=int)
        drug_intera_fea[i, drug_labels] = 1

    drug_intera_fea = drug_intera_fea[:,1:]
    drug_intera_sum = torch.sum(drug_intera_fea,dim=1)
    index = torch.arange(0, N)
    new_drug = index[drug_intera_sum == 0]
    drug_intera_fea_norm = F.normalize(drug_intera_fea)
    ddis = torch.mm(drug_intera_fea_norm, drug_intera_fea_norm.T)
    ddis[new_drug, :] = -1
    ddis[:, new_drug] = -1

    for i in range(N):
        for j in range(N):
            if i>=j:
                ddis[i,j]=-1
    row,clum = torch.where(ddis >= 0.9)
    row_neg, clum_neg = torch.where((ddis >= 0.5) & (ddis < 0.6))

    co_pair = torch.zeros(row.shape[0]+N,2)
    co_pair[:,0] = torch.cat((row,torch.range(0,N-1)),dim=0)
    co_pair[:,1] = torch.cat((clum,torch.range(0,N-1)),dim=0)
    co_pair = torch.tensor(co_pair,dtype = int)

    co_pair_neg = torch.zeros(co_pair.shape[0],2)
    co_pair_neg[:,0] = row_neg[:co_pair.shape[0]]
    co_pair_neg[:,1] = clum_neg[:co_pair.shape[0]]
    co_pair_neg = torch.tensor(co_pair_neg, dtype = int)

    for i in range(event_num):
        adj_i = torch.where(adj == i+1, 1.0, 0.0)
        adj_i = adj_i + adj_i.T
        adj_i = torch.where(adj_i >= 0.5, 1.0, 0.0)
        adj_i = gcnnormalization(adj_i)
        f = torch.isnan(adj_i)
        adj_i[f] = 0
        adj_Heter[i, :, :] = adj_i

    adj_Heter= torch.tensor(adj_Heter).to(torch.float32)
    adj_Heter = adj_Heter/num_drug_labels
    adj_Heter[event_num, :, :] = torch.eye(N)
    f = torch.isnan(adj_Heter)
    adj_Heter[f] = 0

    return adj_Heter, drug_intera_fea

def gcnnormalization(a):
    sum_a = torch.sum(a, dim=1)
    sqrt_sum_a = torch.pow(sum_a, -0.5)
    diag_sqrt_sum_a = torch.diag(sqrt_sum_a)
    b = torch.mm(diag_sqrt_sum_a, a)
    c = torch.mm(b, diag_sqrt_sum_a)
    return c

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def encode_drug(df_data, drug_encoding, column_name = 'SMILES', save_column_name = 'drug_encoding'):
	print('encoding drug...')

	if drug_encoding == 'CNN':
		unique = pd.Series(df_data[column_name]).apply(trans_drug)
		unique_dict = dict(zip(df_data[column_name], unique))
		df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch.
	elif drug_encoding == 'CNN_RNN':
		unique = pd.Series(df_data[column_name]).apply(trans_drug)
		unique_dict = dict(zip(df_data[column_name], unique))
		df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
	else:
		raise AttributeError("Please use the correct drug encoding available!")
	return df_data

def drug_smile_coding(drug_smile, drug_encoding):
    drug_smile = np.array(drug_smile)
    df_smile = pd.DataFrame(zip(drug_smile))
    df_smile.rename(columns={0: 'SMILES'},
                    inplace=True)
    df_data = encode_drug(df_smile, drug_encoding)
    df_sambol = np.array(df_data['drug_encoding'])
    drug_coding = torch.zeros([len(df_sambol), 64, 100])
    for i in range(len(df_sambol)):
        sambol = df_sambol[i]
        drug_coding[i, :, :] = torch.tensor(drug_2_embed(sambol), dtype=float)
    return drug_coding


def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T