# thesis

To read experiment settings:
```
with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)
```

To initialize GPU and random engines
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

# TODO
- still not moving anything to(device), keep that in mind
- test dataset.py
- implement dataset visualization
- do I need to explicitly add self-loops?
- check https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network

# how can I use edge weights in SageConv?
- Adapt the implementation in https://github.com/khuangaf/Pytorch-Geometric-YooChoose
- Using StellarGraph https://stellargraph.readthedocs.io/en/stable/demos/index.html#find-a-demo-for-an-algorithm
- In DGL https://discuss.dgl.ai/t/graphsage-with-edge-features/1046
- Also in DGL https://discuss.dgl.ai/t/using-node-and-edge-features-in-message-passing/762
- In PyG https://github.com/rusty1s/pytorch_geometric/issues/1282
- Using Neighbor sampling https://github.com/rusty1s/pytorch_geometric/issues/64
- Using another type of layer