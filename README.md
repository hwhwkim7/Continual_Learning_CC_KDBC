# Conintual learning with the Collapsed Coreness problem
This algorithm is described in the following papaer:
- Enhancing User Engagement through Network Coreness via Continual Learning
  (연속학습을 통한 네트워크 최소 코어-중심성을 이용한 유저 참여도 강화)

## How to use

### Input parameters
  - gcn : the hidden layer dimemsion of gcn 
  - gcn_num : the number of gcn layer
  - fc : the hidden layer dimemsion of fc 
  - fc_num : the number of fc layer
  - epoch: the number of epoch
  - Path of the graph data



Example code
```
python3 main.py --network ../dataset/karate/ --gcn 64 --gcn_num 2 --fc 128 --fc_num 2 --epoch 100 --gpu 1 --l_ewc 400 --lr 0.01 --b 10
```




