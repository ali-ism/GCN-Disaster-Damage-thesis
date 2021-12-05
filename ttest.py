import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel, wilcoxon


gcn_acc = np.load('results/T-test GCN AE/gcn_acc_ttest.npy')
gcn_prec = np.load('results/T-test GCN AE/gcn_prec_ttest.npy')
gcn_rec = np.load('results/T-test GCN AE/gcn_rec_ttest.npy')
gcn_spec = np.load('results/T-test GCN AE/gcn_spec_ttest.npy')
gcn_f1 = np.load('results/T-test GCN AE/gcn_f1_ttest.npy')

ae_acc = np.load('results/T-test GCN AE/ae_acc_ttest.npy')
ae_prec = np.load('results/T-test GCN AE/ae_prec_ttest.npy')
ae_rec = np.load('results/T-test GCN AE/ae_rec_ttest.npy')
ae_spec = np.load('results/T-test GCN AE/ae_spec_ttest.npy')
ae_f1 = np.load('results/T-test GCN AE/ae_f1_ttest.npy')

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
gcn_scores = [gcn_acc, gcn_prec, gcn_rec, gcn_spec, gcn_f1]
ae_scores = [ae_acc, ae_prec, ae_rec, ae_spec, ae_f1]


plt.figure(figsize=(10,7))
bplot = plt.boxplot(gcn_scores, patch_artist=True, labels=metrics)
for patch in bplot['boxes']:
    patch.set_facecolor('orange')
    patch.set_alpha(0.7)
bplot = plt.boxplot(ae_scores, patch_artist=True, labels=metrics)
for patch in bplot['boxes']:
    patch.set_facecolor('blue')
    patch.set_alpha(0.7)
custom_lines = [Line2D([0],[0],color='orange',lw=4), Line2D([0],[0],color='blue',lw=4)]
leg = plt.legend(custom_lines, ['GCN', 'AE'])
for lh in leg.legendHandles: 
    lh.set_alpha(0.7)
plt.show()


fig = plt.figure(figsize=(10,7))
for i, (metric, gcn_score, ae_score) in enumerate(zip(metrics, gcn_scores, ae_scores)):
    fig.add_subplot(2, 3, i+1)
    plt.hist(gcn_score, alpha=0.7, label='gcn')
    plt.hist(ae_score, alpha=0.7, label='ae')
    plt.legend()
    plt.title(metric)
    t_stat, p_value = ttest_rel(gcn_score, ae_score)
    print(f'\n************{metric}************')
    print('Paired t-test:')
    print(f'p value: {p_value}')
    if p_value <= 0.05:
        print('Null hypothesis rejected')
    else:
        print('Failed to reject null hypothesis')
    print('\nWilcoxon signed rank test:')
    t_stat, p_value = wilcoxon(gcn_score, ae_score)
    print(f'p value: {p_value}')
    if p_value <= 0.05:
        print('Null hypothesis rejected')
    else:
        print('Failed to reject null hypothesis')
fig.tight_layout()
plt.show()