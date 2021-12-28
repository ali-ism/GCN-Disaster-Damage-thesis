import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

train_loss = np.load('results/Beirut Meta Comparison/beirut_gcn_loss_train.npy')
test_loss = np.load('results/Beirut Meta Comparison/beirut_gcn_loss_test.npy')
train_acc = np.load('results/Beirut Meta Comparison/beirut_gcn_acc_train.npy')
test_acc = np.load('results/Beirut Meta Comparison/beirut_gcn_acc_test.npy')
train_precision = np.load('results/Beirut Meta Comparison/beirut_gcn_prec_train.npy')
test_precision = np.load('results/Beirut Meta Comparison/beirut_gcn_prec_test.npy')
train_recall = np.load('results/Beirut Meta Comparison/beirut_gcn_rec_train.npy')
test_recall = np.load('results/Beirut Meta Comparison/beirut_gcn_rec_test.npy')
train_specificity = np.load('results/Beirut Meta Comparison/beirut_gcn_spec_train.npy')
test_specificity = np.load('results/Beirut Meta Comparison/beirut_gcn_spec_test.npy')
train_f1 = np.load('results/Beirut Meta Comparison/beirut_gcn_f1_train.npy')
test_f1 = np.load('results/Beirut Meta Comparison/beirut_gcn_f1_test.npy')

train_loss_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_loss_train.npy')
test_loss_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_loss_test.npy')
train_acc_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_acc_train.npy')
test_acc_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_acc_test.npy')
train_precision_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_prec_train.npy')
test_precision_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_prec_test.npy')
train_recall_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_rec_train.npy')
test_recall_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_rec_test.npy')
train_specificity_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_spec_train.npy')
test_specificity_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_spec_test.npy')
train_f1_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_f1_train.npy')
test_f1_meta = np.load('results/Beirut Meta Comparison/beirut_gcn_meta_f1_test.npy')

fig = make_subplots(rows=2, cols=3)

fig.add_trace(go.Scatter(y=train_loss, name='train', legendgroup='train', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(y=test_loss, name='test', legendgroup='test', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(y=train_loss_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(y=test_loss_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red')), row=1, col=1)

fig.add_trace(go.Scatter(y=train_acc, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(y=test_acc, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(y=train_acc_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(y=test_acc_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(y=train_precision, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(y=test_precision, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(y=train_precision_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(y=test_precision_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=1, col=3)

fig.add_trace(go.Scatter(y=train_recall, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(y=test_recall, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(y=train_recall_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(y=test_recall_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(y=train_specificity, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(y=test_specificity, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(y=train_specificity_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(y=test_specificity_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(y=train_f1, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=2, col=3)
fig.add_trace(go.Scatter(y=test_f1, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=2, col=3)
fig.add_trace(go.Scatter(y=train_f1_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=2, col=3)
fig.add_trace(go.Scatter(y=test_f1_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=2, col=3)

fig['layout']['xaxis']['title']='epochs'
fig['layout']['xaxis2']['title']='epochs'
fig['layout']['xaxis3']['title']='epochs'
fig['layout']['xaxis4']['title']='epochs'
fig['layout']['xaxis5']['title']='epochs'
fig['layout']['xaxis6']['title']='epochs'
fig['layout']['yaxis']['title']='loss'
fig['layout']['yaxis2']['title']='accuracy'
fig['layout']['yaxis3']['title']='precision'
fig['layout']['yaxis4']['title']='recall'
fig['layout']['yaxis5']['title']='specificity'
fig['layout']['yaxis6']['title']='f1'

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()