import random
import torch
import networkx as nx
import numpy as np
import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import from_networkx, subgraph
import json
import csv
from datetime import datetime
from sklearn.model_selection import KFold
from collections import Counter
import os
import pandas as pd
import statistics
import time


# GCN 모델 정의 + fully connected layer
class GCNModelWithFC(nn.Module):
    def __init__(self, input_dim, epochs, hidden_dim_gcn, hidden_dim_fc, output_dim, num_gcn_layers, num_fc_layers,
                 learning_rate, device):
        super(GCNModelWithFC, self).__init__()
        # assert len(hidden_dim_gcn) == num_gcn_layers, "len(hidden_dim_gcn) != num_gcn_layers"
        # assert len(hidden_dim_fc)+1 == num_fc_layers, "len(hidden_dim_fc)+1 != num_fc_layers"

        # 파라미터들을 속성으로 저장
        self.input_dim = input_dim
        self.hidden_dim_gcn = hidden_dim_gcn
        self.hidden_dim_fc = hidden_dim_fc
        self.output_dim = output_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_fc_layers = num_fc_layers
        self.epochs = epochs
        self.learning_rate = learning_rate

        # GCN layer
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(self.input_dim, self.hidden_dim_gcn[0]))  # layer 1
        for i in range(1, self.num_gcn_layers):
            self.gcn_layers.append(GCNConv(self.hidden_dim_gcn[i - 1], self.hidden_dim_gcn[i]))  # 그 외 layers

        # FC layer
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.hidden_dim_gcn[-1], self.hidden_dim_fc[0]))  # layer 1
        for i in range(1, self.num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(self.hidden_dim_fc[i - 1], self.hidden_dim_fc[i]))
        self.fc_layers.append(nn.Linear(self.hidden_dim_fc[-1], self.output_dim))  # 마지막 layer (output)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.device = device

    def forward(self, data):
        # 노드 피처와 엣지 인덱스 추출
        x, edge_index = data.x, data.edge_index

        # GCN
        for conv in self.gcn_layers:
            x = conv(x, edge_index)
            x = F.relu(x)

        # FC
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:  # 마지막 layer는 출력이므로 별도의 활성화 함수 적용 안함
                x = F.relu(x)

        return x

    def fit(self, data, ewc=None, lambda_ewc=None):
        self.to(self.device)
        loss_dict = {}

        # Training loop
        self.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self(data)
            loss = self.criterion(outputs, data.y)

            # EWC 페널티 적용
            if ewc is not None:
                ewc_loss = ewc.penalty(self)
                if torch.isnan(ewc_loss) == False:
                    loss += lambda_ewc * ewc_loss
                else:
                    print("nan")

            loss.backward()
            self.optimizer.step()
            loss_dict[epoch] = loss.item()

        # testQ loop
        # self.eval()
        # with torch.no_grad():
        #     test_output = self(data)
        #     test_loss = self.criterion(test_output, data.y).item()

        if ewc is not None:
            new_fisher_information = ewc.calculate_fisher_information()  # 새로 계산된 FIM
            ewc.update_fisher_information(new_fisher_information)  # 기존 FIM과 결합하여 업데이트

        # return loss_dict, test_loss
        return loss_dict

class EWC:
    def __init__(self, model, data, lambda_ewc, device):
        self.model = model.to(device)
        self.data = data
        self.lambda_ewc = lambda_ewc
        self.fisher_information = self.calculate_fisher_information()

    def calculate_fisher_information(self):
        fisher_information = {}
        for name, param in self.model.named_parameters():
            fisher_information[name] = torch.zeros_like(param)

        self.model.eval()
        self.model.zero_grad()

        output = self.model(self.data)
        loss = torch.nn.functional.mse_loss(output, self.data.y)  # 타겟에 대한 MSE 손실 사용
        loss.backward()

        for name, param in self.model.named_parameters():
            fisher_information[name] += param.grad.data ** 2

        return fisher_information

    def penalty(self, model):
        loss = 0
        for name, param in model.named_parameters():
            fisher_term = self.fisher_information[name]
            loss += (fisher_term * (param - param.data) ** 2).sum()
        return self.lambda_ewc * loss

    def update_fisher_information(self, new_fisher_information):
        for name, fisher_old in self.fisher_information.items():
            fisher_new = new_fisher_information[name]
            self.fisher_information[name] = torch.sqrt(fisher_old * fisher_new)


def get_follower_counts(G, sample=[]):
    if len(sample) == 0: sample = G.nodes()
    original_coreness = nx.core_number(G)  # 원본 그래프의 coreness 계산
    follower_counts = {}
    followers = {}

    for node in sample:
        # 노드를 하나 제거한 새로운 그래프 생성
        G_copy = G.copy()
        G_copy.remove_node(node)

        # 노드가 제거된 그래프에서의 coreness 계산
        new_coreness = nx.core_number(G_copy)

        follower = []
        # coreness 차이를 계산하고 팔로워 수를 계산
        for n in G.nodes():
            if n == node: continue
            if original_coreness[n] != new_coreness[n]:
                follower.append(n)
        followers[node] = follower
        follower_counts[node] = sum(original_coreness.values()) - sum(new_coreness.values())
    max_node = max(follower_counts, key=follower_counts.get)

    return follower_counts, max_node, followers

def node_features(G):
    # 노드 피처 계산
    degree = torch.tensor([G.degree[node] for node in G.nodes()], dtype=torch.float).view(-1, 1)
    clustering = torch.tensor([nx.clustering(G, node) for node in G.nodes()], dtype=torch.float).view(-1, 1)
    coreness = torch.tensor([nx.core_number(G)[node] for node in G.nodes()], dtype=torch.float).view(-1, 1)
    pagerank = torch.tensor([nx.pagerank(G)[node] for node in G.nodes()], dtype=torch.float).view(-1, 1)

    # 노드 피처 병합
    node_features = torch.cat([
        degree,
        clustering,
        coreness,
        pagerank,
    ], dim=1)

    return node_features

def set_Xy(G, node_set, node_features, follower_counts, device, data=None):
    if data is None:
        # PyTorch Geometric의 Data 객체로 변환
        data = from_networkx(G)

    # 마스크 생성 (train_nodes는 트레이닝, test_nodes는 테스트)
    train_mask = torch.zeros(G.number_of_nodes(), dtype=torch.bool)
    test_mask = torch.ones(G.number_of_nodes(), dtype=torch.bool).to(device)
    for node in node_set:
        train_mask[node] = True

    # 데이터 객체에 마스크 할당
    data.train_mask = train_mask
    data.test_mask = test_mask


    # MinMaxScaler를 사용하여 node feature scaling
    scaler = MinMaxScaler()
    node_features_scaled = scaler.fit_transform(node_features)  # 0~1 사이 값으로 변환
    data.x = torch.tensor(node_features_scaled, dtype=torch.float).to(device)  # 데이터를 GPU로 이동

    follower_target = torch.tensor([follower_counts[node] for node in G.nodes()], dtype=torch.float).view(-1, 1)
    # MinMaxScaler를 사용하여 target scaling
    scaler_y = MinMaxScaler()
    follower_target_scaled = scaler_y.fit_transform(follower_target)  # 0~1 사이 값으로 변환
    data.y = torch.tensor(follower_target_scaled, dtype=torch.float).to(device)  # 데이터를 GPU로 이동

    # 엣지 인덱스 설정
    data.edge_index, _ = subgraph(node_set, data.edge_index, relabel_nodes=True, num_nodes=G.number_of_nodes())
    data.edge_index = data.edge_index.to(device)  # 데이터를 GPU로 이동

    return data, scaler_y

def model_train(data, device, best_params, ewc=None):
    input_dim = data.x.shape[1]
    hidden_dim_gcn = [best_params['hidden_dim_gcn']] * best_params['num_gcn_layers']
    hidden_dim_fc = [best_params['hidden_dim_fc']] * best_params['num_fc_layers']
    model = GCNModelWithFC(input_dim, best_params['epochs'], hidden_dim_gcn, hidden_dim_fc, 1,
                           best_params['num_gcn_layers'], best_params['num_fc_layers'], best_params['learning_rate'],
                           device)

    if ewc is not None:
        loss_dict = model.fit(data, ewc=ewc, lambda_ewc=best_params['lambda_ewc'])
    else:
        loss_dict = model.fit(data)  # ewc 없이 학습

    return model, loss_dict, ewc, best_params

def model_eval(node_set, model, data, scaler_y, iteration, st=None, predict={}):
    result = {}
    model.eval()
    with torch.no_grad():
        test_output = model(data)

        # Test loss 계산 (역변환된 값이 아닌 스케일된 상태에서 Loss 계산)
        test_loss = model.criterion(test_output, data.y).item()

        # 만약 타겟 값과 예측 값이 MinMaxScaler로 스케일링된 경우, 이를 원래 값으로 복원
        test_output_original = scaler_y.inverse_transform(test_output.cpu())  # 예측 값 역변환
        test_output_original = np.round(test_output_original).astype(int).flatten().tolist()
        actual_values_original = scaler_y.inverse_transform(data.y.cpu())  # 실제 값 역변환
        actual_values_original = np.round(actual_values_original).astype(int).flatten().tolist()
        for i, n in enumerate(node_set):
            predict[n] = test_output_original[i]
        max_key = max(predict, key=predict.get)
        end_time = time.time() - st

        result[iteration] = {}
        result[iteration]['loss'] = test_loss
        result[iteration]['predict'] = test_output_original
        result[iteration]['real'] = actual_values_original
        result[iteration]['num_samples'] = len(node_set)
        result[iteration]['predict_coreness_gain_sum'] = sum(predict.values())
        result[iteration]['real_coreness_gain_sum'] = sum(actual_values_original)
        result[iteration]['iter_time'] = end_time
        result[iteration]['max_key'] = max_key
    return max_key, result, predict


def remove_nodes(G, max_key, predict, followers):
    if isinstance(max_key, int):
        G.remove_node(max_key)
        del predict[max_key]
    else:
        G.remove_nodes_from(max_key[0])
        del predict[max_key[0]]

    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    new_graph = nx.relabel_nodes(G, mapping)
    changed_nodes = [mapping[old_id] for old_id in followers[max_key]]
    predict = {mapping[k]: v for k, v in predict.items()}

    return changed_nodes, new_graph, predict, mapping
