"""
Demonstration of how to perform SQUID global (pairwise) analysis with example Kipoi model (BPNet)

Due to BPNet requiring incompatible libraries with MAVE-NN, the current script is separated
into two parts (with switches 'STEP 1' and 'STEP 2'). STEP 1 requires an activated BPNet environment,
and can be run via: 'python example_global_pairwise.py 1 {inter_dist}' where '{inter_dist}' is an int.
Once outputs are saved to file, deactivate the environment and activate the MAVE-NN environment.
Finally, rerun this script via: 'python example_global_pairwise.py 2 {inter_dist}'

For using the BPNet Kipoi model, the following packages must be installed in the BPNet environment:
    >>> pip install kipoi --upgrade
    >>> pip install kipoiseq --upgrade

For instruction on installing the BPNet environment, see:
https://github.com/evanseitz/squid-manuscript/blob/main/examples/README_environments.md
"""

import os, sys
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random


def op(py_dir, step, inter_dist):
    """Function to enable batch computations for different values of inter-site distance.
    Hyperparameters are defined via command line: e.g., python example_global_pairwise.py 1 10
    representing an 'inter_dist' of 10 for 'step' 1. All other hyperparameters should be adjusted
    inside this function before running. It is also possible to run a batch of 'inter_dist' values
    sequentially (i.e., 'bash example_global_pairwise_batch.sh') or in parralel on GPUs via:
    e.g.,   for i in {0..30}; do echo "python example_global_pairwise.py 1 $i 
            device=\$CUDA_VISIBLE_DEVICES && sleep 3"; done | simple_gpu_scheduler --gpus 0,1,2

    Parameters
    ----------
    step : int
        either 1 or 2 for STEP 1 (BPNet) or STEP 2 (MAVE-NN), respectively
    inter_dist : int
        inter-site distance between end of pattern_1 and start of pattern_2 (0 <= inter_dist)
    """
    gpu = True
    parent_dir = os.path.dirname(py_dir)

    # define global patterns (i.e., conserved sequences of interest)
    task_idx = 'Nanog' # bpnet task index ('Oct4', 'Sox2', 'Klf4' or 'Nanog')
    pattern_1 = 'AGCCATCAA' # Nanog binding site
    pattern_2 = 'GAACAATAG' # Sox2 binding site

    alphabet = ['A','C','G','T']
    seq_length = 1000 # full sequence length of bpnet inputs

    # create initial sequence with global patterns surrounded by random background
    start_pos = int(seq_length//2) # position of inserted pattern_1 in background DNA
    bg = ''.join(random.choices(str(''.join(alphabet)), k=seq_length)) # random DNA background
    seq = bg[:start_pos] + pattern_1 + bg[start_pos:start_pos+inter_dist] + pattern_2 + bg[start_pos+inter_dist:]
    seq = seq[:seq_length]
    mut_window = [start_pos, start_pos+len(pattern_1)+inter_dist+len(pattern_2)] # interval in sequence to mutagenize (locally)
    inter_window = [start_pos+len(pattern_1), start_pos+len(pattern_1)+inter_dist] # inter-pattern window to mutagenize (globally)

    save_dir = os.path.join(py_dir, 'outputs_global_pairwise/dist_%s' % inter_dist)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if step == 1: # STEP 1 (requires BPNet environment)
        import kipoi
        sys.path.append(os.path.join(parent_dir, 'squid'))
        import utils, predictor, mutagenizer, mave # import squid modules manually

        # convert to one-hot
        x = utils.seq2oh(seq, alphabet)

        # instantiate kipoi model for bpnet
        model = kipoi.get_model('BPNet-OSKN')

        # define how to go from kipoi prediction to scalar
        bpnet_predictor = predictor.BPNetPredictor(model.predict_on_batch, task_idx=task_idx, batch_size=512,
                                                   #reduce_fun='wn', # default BPNet contribution scores
                                                   reduce_fun=predictor.profile_pca, save_dir=save_dir)

        # set up mutagenizer class for in silico MAVE
        mut_generator = mutagenizer.RandomMutagenesis(mut_rate=0.1, uniform=False)

        # generate in silico MAVE
        num_sim = 100000 # number of sequence to simulate
        mave = mave.InSilicoMAVE(mut_generator, mut_predictor=bpnet_predictor, seq_length=seq_length, mut_window=mut_window,
                                alphabet=alphabet, inter_window=inter_window, context_agnostic=True) # this line required for global analysis 
        x_mut, y_mut = mave.generate(x, num_sim=num_sim)

        # save in silico MAVE dataset for STEP 2
        np.save(os.path.join(save_dir, 'x_mut.npy'), x_mut)
        np.save(os.path.join(save_dir, 'y_mut.npy'), y_mut)
        print('In silico MAVE dataset saved.')


    elif step == 2: # STEP 2 (requires MAVE-NN environment)
        import mavenn
        import squid
        import matplotlib.pyplot as plt

        # choose surrogate model type
        gpmap = 'pairwise'

        # load in in silico MAVE dataset from STEP 1
        x_mut = np.load(os.path.join(save_dir, 'x_mut.npy'))
        y_mut = np.load(os.path.join(save_dir, 'y_mut.npy'))

        # delimit sequence to region of interest (required for pairwise computational constraints)
        x_mut_trim = x_mut[:,490:550,:]

        # MAVE-NN model with GE nonlinearity
        surrogate_model = squid.surrogate_zoo.SurrogateMAVENN(x_mut_trim.shape, num_tasks=y_mut.shape[1],
                                                        gpmap=gpmap, regression_type='GE',
                                                        linearity='nonlinear', noise='SkewedT',
                                                        noise_order=2, reg_strength=0.1,
                                                        alphabet=alphabet, deduplicate=True,
                                                        gpu=gpu)

        # train surrogate model
        surrogate, mave_df = surrogate_model.train(x_mut_trim, y_mut, learning_rate=5e-4, epochs=500, batch_size=100,
                                                early_stopping=True, patience=25, restore_best_weights=True,
                                                save_dir=None, verbose=1)
        
        # save mavenn model
        surrogate.save(os.path.join(save_dir, 'surrogate_model'))

        # save mave dataframe
        mave_df.to_csv(os.path.join(save_dir, 'mave_dataframe.csv'))

        # retrieve model parameters
        params = surrogate_model.get_params(gauge='empirical')

        # evaluate model performance
        trainval_df, test_df = mavenn.split_dataset(mave_df)
        info = surrogate_model.get_info()

        # plot mavenn model performance
        fig = squid.impress.plot_performance(surrogate, info=info, save_dir=save_dir)

        # plot mavenn y versus yhat
        fig = squid.impress.plot_y_vs_yhat(surrogate, mave_df=mave_df, save_dir=save_dir)

        # plot mavenn y versus phi
        fig = squid.impress.plot_y_vs_phi(surrogate, mave_df=mave_df, save_dir=save_dir)

        # plot additive logo in hierarchical gauge
        fig = squid.impress.plot_additive_logo(params[1], center=True, view_window=None, alphabet=alphabet, fig_size=[20,2.5], save_dir=save_dir)

        # plot pairwise matrix
        fig = squid.impress.plot_pairwise_matrix(params[2], view_window=None, alphabet=alphabet, threshold=None, save_dir=save_dir)



if __name__ == '__main__':
    py_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 2:
        step = int(sys.argv[1])
        inter_dist = int(sys.argv[2])
        if step not in [1, 2]:
            print('Argument for the current step must be 1 or 2.')
            sys.exit(0)
        if inter_dist < 0:
            print('Argument for the inter-site distance must be greater than 0.')
            sys.exit(0)
    else:
        print('Script must be run with correct arguments.')
        sys.exit(0)
    op(py_dir, step, inter_dist)