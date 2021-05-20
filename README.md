To read experiment settings:
```
with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)
```

# TODO
- test dataset.py
- implement dataset visualization
- In model.py implement the forward function for SAGENet, what is res_size?
- Continue working on train.py, parametrize experiment setting via exp_settings.json, implement metrics logging and model saving

# Should I do one-hot encoding?
- https://stats.stackexchange.com/questions/140061/how-to-set-up-neural-network-to-output-ordinal-data
- https://stats.stackexchange.com/questions/222073/classification-with-ordered-classes

# How can I use edge weights in SageConv?
- Check GraphSAINT or NNConv or SGC

# How to access edge_attr from NeighborSampler?
- add self loop because it seems that neighbor sampler assumes self loops for sampling edges? (https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network)

Neighbor sampler works on a single graph. So we can:
- create a sampler for each graph and loop over them while training (research forgetfulness)
- or load all graphs into a single graph where nodes belonging to different graphs are not connected (endorsed by developers).