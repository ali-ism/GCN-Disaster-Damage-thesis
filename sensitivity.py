import numpy as np
import matplotlib.pyplot as plt

gcn_acc = np.load('results/gcn_acc_sens.npy')
gcn_prec = np.load('results/gcn_prec_sens.npy')
gcn_rec = np.load('results/gcn_rec_sens.npy')
gcn_spec = np.load('results/gcn_spec_sens.npy')
gcn_f1 = np.load('results/gcn_f1_sens.npy')

ae_acc = np.load('results/ae_acc_sens.npy')
ae_prec = np.load('results/ae_prec_sens.npy')
ae_rec = np.load('results/ae_rec_sens.npy')
ae_spec = np.load('results/ae_spec_sens.npy')
ae_f1 = np.load('results/ae_f1_sens.npy')

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
gcn_scores = [gcn_acc, gcn_prec, gcn_rec, gcn_spec, gcn_f1]
ae_scores = [ae_acc, ae_prec, ae_rec, ae_spec, ae_f1]

fig = plt.figure(figsize=(10,7))
columns = 3
rows = 2
for i, metric, gcn_score, ae_score in enumerate(zip(metrics, gcn_scores, ae_scores)):
    fig.add_subplot(2, 3, i)
    plt.plot(gcn_score, marker='o', label='gcn')
    plt.plot(ae_score, marker='o', label='ae')
    plt.legend()
    plt.xlabel('size of labeled set')
    plt.ylabel(metric)
plt.show()