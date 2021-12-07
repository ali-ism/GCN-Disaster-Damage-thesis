import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel, wilcoxon


acc = np.load('results/T-test Beirut/beirut_acc_ttest.npy')
prec = np.load('results/T-test Beirut/beirut_prec_ttest.npy')
rec = np.load('results/T-test Beirut/beirut_rec_ttest.npy')
spec = np.load('results/T-test Beirut/beirut_spec_ttest.npy')
f1 = np.load('results/T-test Beirut/beirut_f1_ttest.npy')

meta_acc = np.load('results/T-test Beirut/beirut_meta_acc_ttest.npy')
meta_prec = np.load('results/T-test Beirut/beirut_meta_prec_ttest.npy')
meta_rec = np.load('results/T-test Beirut/beirut_meta_rec_ttest.npy')
meta_spec = np.load('results/T-test Beirut/beirut_meta_spec_ttest.npy')
meta_f1 = np.load('results/T-test Beirut/beirut_meta_f1_ttest.npy')

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
scores = [acc, prec, rec, spec, f1]
meta_scores = [meta_acc, meta_prec, meta_rec, meta_spec, meta_f1]


plt.figure(figsize=(10,7))
bplot = plt.boxplot(scores, patch_artist=True, labels=metrics)
for patch in bplot['boxes']:
    patch.set_facecolor('orange')
    patch.set_alpha(0.7)
bplot = plt.boxplot(meta_scores, patch_artist=True, labels=metrics)
for patch in bplot['boxes']:
    patch.set_facecolor('blue')
    patch.set_alpha(0.7)
custom_lines = [Line2D([0],[0],color='orange',lw=4), Line2D([0],[0],color='blue',lw=4)]
leg = plt.legend(custom_lines, ['No Meta', 'With Meta'])
for lh in leg.legendHandles: 
    lh.set_alpha(0.7)
plt.show()


fig = plt.figure(figsize=(10,7))
for i, (metric, gcn_score, ae_score) in enumerate(zip(metrics, scores, meta_scores)):
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