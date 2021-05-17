To read experiment settings:
```
with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)
```

# TODO
- test dataset.py
- implement dataset visualization
- do I need to explicitly add self-loops?
- https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network
- In model.py implement the forward function for SAGENet
- Continue working on train.py, parametrize experiment setting via exp_settings.json, implement metrics logging and model saving

# Should I do one-hot encoding?
- https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
- https://stats.stackexchange.com/questions/140061/how-to-set-up-neural-network-to-output-ordinal-data

# How can I use edge weights in SageConv?
- Adapt the implementation in https://github.com/khuangaf/Pytorch-Geometric-YooChoose
- Using StellarGraph https://stackoverflow.com/questions/57259920/is-there-a-way-to-allow-graphsage-take-into-account-weighted-edges
- Check GraphSAINT or NNConv or SGC

# How to access edge_attr from NeighborSampler?
Neighbor sampler works on a single graph. So we can:
- create a sampler for each graph and loop over them while training (research forgetfulness)
- or load all graphs into a single graph where nodes belonging to different graphs are not connected.

# To keep in mind
- moving things to GPU
- dataset.py and generate_disaster_dict.py have argv