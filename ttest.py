import numpy as np
from scipy.stats import t

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

scores = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
gcn_scores = [gcn_acc, gcn_prec, gcn_rec, gcn_spec, gcn_f1]
ae_scores = [ae_acc, ae_prec, ae_rec, ae_spec, ae_f1]