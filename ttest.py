import json

import numpy as np
from scipy.stats import t

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

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

n = settings_dict['data_ss']['reduced_size']
n1 = round(n * settings_dict['data_ss']['labeled_size'])
n2 = round(n * 0.5)

for metric, gcn_score, ae_score in zip(metrics, gcn_scores, ae_scores):
    d = gcn_score - ae_score
    mean = np.mean(d)
    var = np.var(d)
    var_mod = var + (1/n + n2/n1)
    t_static =  mean / np.sqrt(var_mod)
    p_value = ((1 - t.cdf(t_static, n-1))*200)
    print(f'\n{metric}:')
    print(f'p value: {p_value}')
    if p_value < 0.005:
        print('Null hypothesis rejected')
    else:
        print('Failed to reject null hypothesis')