import networkx as nx
import torch
import argparse
import time

import functions

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='../dataset/karate/', help='network file path')
parser.add_argument('--gcn', type=int, default=64)
parser.add_argument('--gcn_num', type=int, default=2)
parser.add_argument('--fc', type=int, default=128)
parser.add_argument('--fc_num', type=int, default=2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--l_ewc', type=float, default=400, help='penalty')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--b', type=int, default=10)

args = parser.parse_args()

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print("GPU 사용 가능")
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")

    # GPU 0번에 할당
    device = torch.device("cuda:0")
    print(f"선택된 GPU: {torch.cuda.get_device_name(args.gpu)}")
else:
    print("GPU 사용 불가, CPU를 사용합니다.")
    device = torch.device("cpu")

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

our = []
loss = []

# 1. 그래프 데이터 로드
all_time = time.time()
G = nx.read_edgelist(args.network + 'network.dat', nodetype=int)
name = args.network.split('/')[2]
# self loop 제거
G.remove_edges_from(nx.selfloop_edges(G))
# 노드 라벨 변경
mapping = {node: idx for idx, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)
print(f"========== name: {name}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()} ==========")
original_number = G.number_of_nodes()
st = time.time()

# 노드 sampling
sample_nodes = list(G.nodes())

# 노드 피처 계산
node_features = functions.node_features(G)

# 팔로워 수 계산
follower_counts, max_max, max_follower = functions.get_follower_counts(G)

# scaling 후 PyTorch Geometric의 Data 객체로 변환
data, scaler_y = functions.set_Xy(G, sample_nodes, node_features, follower_counts, device)

# 모델 학습
learning_rate = 0.01
iteration = 1
best_params = {
    'num_gcn_layers': args.gcn_num,
    'hidden_dim_gcn': args.gcn,
    'num_fc_layers': args.fc_num,
    'hidden_dim_fc': args.fc,
    'epochs': args.epoch,
    'learning_rate': args.lr,
    'lambda_ewc': args.l_ewc}

name = args.network.split('/')[2]
model, loss_dict, ewc, best_params = functions.model_train(data, device, best_params, None)

# 모델 평가
max_keys, result, predict = functions.model_eval(sample_nodes, model, data, scaler_y, iteration, st)
first = result[iteration]

# budget 설정
lambda_ewc = best_params['lambda_ewc']
while iteration <= args.b:
    st = time.time()
    changed_nodes, G, predict, mapping_dict = functions.remove_nodes(G, max_keys, predict, max_follower)
    if G.number_of_nodes() == 0: break
    if len(changed_nodes) == 0:
        del max_follower[max_keys]
        for key in list(max_follower.keys()):
            # 1. 값 리스트에서 max_key에 해당하는 값 제거
            max_follower[key] = [node_id for node_id in max_follower[key] if node_id != max_keys]
            # 2. 값 리스트 안의 값들도 mapping_dict에 따라 리라벨링
            max_follower[key] = [mapping_dict.get(node_id, node_id) for node_id in max_follower[key]]
            # 3. 키를 mapping_dict에 따라 리라벨링
            if key in mapping_dict:
                new_key = mapping_dict[key]
                max_follower[new_key] = max_follower.pop(key)  # 기존 키 삭제하고 새로운 키로 변경
        max_keys = max(predict, key=predict.get)
        continue
    iteration += 1

    # 노드 피처 계산
    node_feature = functions.node_features(G)

    # 팔로워 수 계산
    follower_counts, max_max, max_follower = functions.get_follower_counts(G)

    # scaling 후 PyTorch Geometric의 Data 객체로 변환
    data, scaler_y = functions.set_Xy(G, changed_nodes, node_feature, follower_counts, device)
    # data_sample, scaler_y_sample = functions.set_Xy(G, sample_nodes_random, node_feature, follower_counts, device)

    if ewc is None:
        ewc = functions.EWC(model, data, lambda_ewc, device)
        # ewc_random = functions.EWC(model, data_sample, lambda_ewc, device)

    # 모델 학습
    model, loss_dict, ewc, best_params = functions.model_train(data, device, best_params, ewc)

    # 모델 평가
    max_keys, result, predict = functions.model_eval(changed_nodes, model, data, scaler_y, iteration, st, predict)


    result[iteration]['coreness_sum'] = sum(nx.core_number(G).values())

    our.append(result)
    loss.append(loss_dict)

all_end_time = time.time() - all_time

core = nx.core_number(G)
core_num = sum(core.values())
for o in our:
    print(o)
print('total sum:', core_num)