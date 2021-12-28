import numpy as np
import matplotlib.pyplot as plt

gcn_acc = np.load('results/Sensitivity GCN AE/gcn_acc_sens.npy')
gcn_prec = np.load('results/Sensitivity GCN AE/gcn_prec_sens.npy')
gcn_rec = np.load('results/Sensitivity GCN AE/gcn_rec_sens.npy')
gcn_spec = np.load('results/Sensitivity GCN AE/gcn_spec_sens.npy')
gcn_f1 = np.load('results/Sensitivity GCN AE/gcn_f1_sens.npy')

ae_acc = np.load('results/Sensitivity GCN AE/ae_acc_sens.npy')
ae_prec = np.load('results/Sensitivity GCN AE/ae_prec_sens.npy')
ae_rec = np.load('results/Sensitivity GCN AE/ae_rec_sens.npy')
ae_spec = np.load('results/Sensitivity GCN AE/ae_spec_sens.npy')
ae_f1 = np.load('results/Sensitivity GCN AE/ae_f1_sens.npy')

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
gcn_scores = [gcn_acc, gcn_prec, gcn_rec, gcn_spec, gcn_f1]
ae_scores = [ae_acc, ae_prec, ae_rec, ae_spec, ae_f1]
labeled_sizes = [0.1, 0.2, 0.3, 0.4]

fig = plt.figure(figsize=(10,7))
for i, (metric, gcn_score, ae_score) in enumerate(zip(metrics, gcn_scores, ae_scores)):
    fig.add_subplot(2, 3, i+1)
    plt.plot(labeled_sizes, gcn_score, marker='o', label='BLDNet')
    plt.plot(labeled_sizes, ae_score, marker='o', label='AE')
    #plt.legend()
    plt.xlabel('size of labeled set')
    plt.ylabel(metric)
    handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc=[0.08,0.9])
fig.tight_layout()
plt.show()