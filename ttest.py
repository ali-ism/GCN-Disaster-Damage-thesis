import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

#with open('exp_settings.json', 'r') as JSON:
#    settings_dict = json.load(JSON)

gcn_acc = np.load('results/gcn_acc_ttest.npy')
gcn_prec = np.load('results/gcn_prec_ttest.npy')
gcn_rec = np.load('results/gcn_rec_ttest.npy')
gcn_spec = np.load('results/gcn_spec_ttest.npy')
gcn_f1 = np.load('results/gcn_f1_ttest.npy')

ae_acc = np.load('results/ae_acc_ttest.npy')
ae_prec = np.load('results/ae_prec_ttest.npy')
ae_rec = np.load('results/ae_rec_ttest.npy')
ae_spec = np.load('results/ae_spec_ttest.npy')
ae_f1 = np.load('results/ae_f1_ttest.npy')

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
gcn_scores = [gcn_acc, gcn_prec, gcn_rec, gcn_spec, gcn_f1]
ae_scores = [ae_acc, ae_prec, ae_rec, ae_spec, ae_f1]

#n = settings_dict['data_ss']['reduced_size']
#n1 = round(settings_dict['data_ss']['reduced_size'] * settings_dict['data_ss']['labeled_size'])
#n2 = round(settings_dict['data_ss']['reduced_size'] * 0.5)
#n = n1 + n2

fig = plt.figure(figsize=(10,7))
for i, (metric, gcn_score, ae_score) in enumerate(zip(metrics, gcn_scores, ae_scores)):
    fig.add_subplot(2, 3, i+1)
    plt.hist(gcn_score, alpha=0.7, label='gcn')
    plt.hist(ae_score, alpha=0.7, label='ae')
    plt.legend()
    plt.title(metric)
    #d = gcn_score - ae_score
    #mean = np.mean(d)
    #var = np.var(d)
    #var_mod = var + (1/n + n2/n1)
    #t_stat =  mean / np.sqrt(var_mod)
    #p_value = ((1 - t.cdf(t_stat, n-1))*200)
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