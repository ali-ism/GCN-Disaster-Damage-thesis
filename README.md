# thesis

To read experiment settings:
```
with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)
```

# TODO
- test dataset.py
- implement dataset visualization
- do I need to explicitly add self-loops?
- check https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network
- In model.py implement the forward function for SAGENet
- Continue working on train.py, how to access edge_attr from NeighborSampler?, parametrize experiment setting via exp_settings.json, implement metrics logging and model saving
- Should I do one-hot encoding?

# How can I use edge weights in SageConv?
- Adapt the implementation in https://github.com/khuangaf/Pytorch-Geometric-YooChoose
- Using StellarGraph https://stackoverflow.com/questions/57259920/is-there-a-way-to-allow-graphsage-take-into-account-weighted-edges
- Check GraphSAINT or NNConv or SGC

# To keep in mind
- moving things to device
- dataset.py and generate_disaster_dict.py have argv