"""
Demonstration of how to use SQUID with example Kipoi model (DeepSTARR)

For using kipoi models, the following packages must be installed:
    >>> pip install kipoi
    >>> pip install kipoiseq
"""

import os, sys
sys.dont_write_bytecode = True
import mavenn
import squid
import kipoi
import matplotlib.pyplot as plt


# =============================================================================
# Computational settings
# =============================================================================
gpu = False
save = True


# =============================================================================
# Main pipeline
# =============================================================================
if save:
    py_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(py_dir, 'squid_testing_outputs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
else:
    save_dir = None


task_idx = 0 #deepstarr task index (either 0 Dev or 1 HK)
alphabet = ['A','C','G','T']

# define sequence of interest
seq = 'GGCTCTGTCTCAGTTTCTGATTCAGTTTCGGATCCACTTCGAGAGGCAGAAGTCGGGGTCCGAGAGGCATTAGCTTGTTAGTTCTACAACCTGCTGGCAAATGTGCCAATATGTTTGCACGCTGATAAGGCCTACATGGCACCGAATTGAAAACCGCTTACATAATGAAGTGAATAGTCAGCGAATCGGCAGAGCAACCGCAATGCATTGCATTCACCATCGCGAATAATCAGATTCAAGGCAACGATC'

# convert to one-hot
x = squid.utils.seq2oh(seq, alphabet)

# instantiate kipoi model for deepstarr
model = kipoi.get_model('DeepSTARR')

# define how to go from kipoi prediction to scalar
kipoi_predictor = squid.predictor.ScalarPredictor(model.predict_on_batch, task_idx=task_idx, batch_size=512)

# set up mutagenizer class for in silico MAVE
mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=0.1, uniform=False)

# generate in silico MAVE
seq_length = len(x)
mut_window = [0, seq_length] #interval in sequence to mutagenize
num_sim = 20000 #number of sequence to simulate
mave = squid.mave.InSilicoMAVE(mut_generator, mut_predictor=kipoi_predictor, seq_length=seq_length, mut_window=mut_window)
x_mut, y_mut = mave.generate(x, num_sim=num_sim)

# choose surrogate model type
gpmap = 'additive'

# MAVE-NN model with GE nonlinearity
surrogate_model = squid.surrogate_zoo.SurrogateMAVENN(x_mut.shape, num_tasks=y_mut.shape[1],
                                                gpmap=gpmap, regression_type='GE',
                                                linearity='nonlinear', noise='SkewedT',
                                                noise_order=2, reg_strength=0.1,
                                                alphabet=alphabet, deduplicate=True,
                                                gpu=gpu)

# train surrogate model
surrogate, mave_df = surrogate_model.train(x_mut, y_mut, learning_rate=5e-4, epochs=500, batch_size=100,
                                           early_stopping=True, patience=25, restore_best_weights=True,
                                           save_dir=None, verbose=1)

# retrieve model parameters
params = surrogate_model.get_params(gauge='empirical')

# generate sequence logo
logo = surrogate_model.get_logo(mut_window=mut_window, full_length=seq_length)

# fix gauge for variant effect prediction
variant_effect = squid.utils.fix_gauge(logo, gauge='wildtype', wt=x_mut[0])

# save variant effects to pandas
variant_effect_df = squid.utils.arr2pd(variant_effect, alphabet)
print(variant_effect_df)
variant_effect_df.to_csv(os.path.join(save_dir, 'variant_effects.csv'))

# plot additive logo in wildtype gauge
fig = squid.impress.plot_additive_logo(variant_effect, center=False, view_window=mut_window, alphabet=alphabet, fig_size=[20,2.5], save_dir=save_dir)