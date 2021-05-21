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
(https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network)

Neighbor sampler works on a single graph. So we can:
- create a sampler for each graph and loop over them while training (research forgetfulness)
- or load all graphs into a single graph where nodes belonging to different graphs are not connected (endorsed by developers).