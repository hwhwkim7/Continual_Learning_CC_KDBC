# Enhancing User Engagement through Network Coreness via Continual Learning
- 연속학습을 통한 네트워크 최소 코어-중심성을 이용한 유저 참여도 강화

## How to use

### Input parameters
- network : Path of the graph data
- gcn : The hidden layer dimension of GCN (default: 64)
- gcn_num : The number of GCN layers (default: 2)
- fc : The hidden layer dimension of the fully connected layer (default: 128)
- fc_num : The number of fully connected layers (default: 2)
- epoch : The number of epochs for training (default: 100)
- gpu : The GPU ID to use (default: 2)
- l_ewc : The penalty coefficient for elastic weight consolidation (default: 400)
- lr : The learning rate (default: 0.01)
- b : The budget size (default: 10)



Example code
```
python3 main.py --network ../dataset/karate/ --gcn 64 --gcn_num 2 --fc 128 --fc_num 2 --epoch 100 --gpu 1 --l_ewc 400 --lr 0.01 --b 10
```




