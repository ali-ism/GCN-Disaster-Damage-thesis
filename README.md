# TODO
- implement model embedding visualization

# Should I do one-hot encoding?
- https://stats.stackexchange.com/questions/140061/how-to-set-up-neural-network-to-output-ordinal-data
- https://stats.stackexchange.com/questions/222073/classification-with-ordered-classes

# How to access edge_attr from NeighborSampler?
(https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network)

Neighbor sampler works on a single graph. So we can:
- create a sampler for each graph and loop over them while training (forgetfulness)
- or load all graphs into a single giant graph where nodes belonging to different graphs are not connected (endorsed by developers).

# Discrepancy in f1 score calculation?

# Pixel coordinate centroids are only valid within each tile (not cross-tile)
- Discard distance in edge features and still have large graphs spanning multiple tiles. Still we have no basis on which to group tiles.
- Have one graph per tile but end up with smaller individual graphs (some with only one node). For tiles that have very few polygons, perhaps they can be appended to other tiles but the euclidean similarity is set to zero.
- So far, one graph per tile discarding tiles with single polygon. "disaster_dirs.json" not really needed.

# Sampling Strategies other than Neighbor Sampling
- ClusterGCN: uses a graph clustering algorithm to sample dense subgraphs to use as mini-batches.
- GraphSAINT: samples subgraphs based on a node sampling probability to use as mini-batches. The parameters of the sampling probability are also used to normalize the loss function to reduce bias due to sampling.
- SIGN: does not rely on graph sampling but precomputes fixed neighborhood aggregations inspired by Inception. A regular MLP can subsequently be trained to make predictions.

# Experiments

| Train                                                                          | Test                  |
|--------------------------------------------------------------------------------|-----------------------|
| Mexico earthquake                                                              | Mexico earthquake     |
| Palu tsunami                                                                   | Mexico earthquake     |
| Palu tsunami + hurricane matthew + santa rosa wildfire                         | Mexico earthquake     |
| Palu tsunami + hurricane matthew + santa rosa wildfire + 10% mexico earthquake | 90% mexico earthquake |