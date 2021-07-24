# TODO
- Implement model embedding visualization.
- Prepare Beirut data.

# Should I do one-hot encoding?
- https://stats.stackexchange.com/questions/140061/how-to-set-up-neural-network-to-output-ordinal-data
- https://stats.stackexchange.com/questions/222073/classification-with-ordered-classes

# How to access edge_attr from NeighborSampler?
(https://stackoverflow.com/questions/10649673/how-to-generate-a-fully-connected-subgraph-from-node-list-using-pythons-network)

# Experiments

| Train                                                                          | Test                  |
|--------------------------------------------------------------------------------|-----------------------|
| Mexico earthquake                                                              | Mexico earthquake     |
| Palu tsunami                                                                   | Mexico earthquake     |
| Palu tsunami + hurricane matthew + santa rosa wildfire                         | Mexico earthquake     |
| Palu tsunami + hurricane matthew + santa rosa wildfire + 10% mexico earthquake | 90% mexico earthquake |