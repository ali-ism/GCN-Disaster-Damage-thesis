# TODO
- make dataset creation faster (vectorization, and disk-based)
- test dataset.py
- implement dataset visualization
- In model.py implement EdgeSAGEConv
- Test train.py

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

# Discrepancy in f1 score calculation?