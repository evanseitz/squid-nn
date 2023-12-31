"""
Demonstration of how to perform SQUID global (additive) analysis with example Kipoi model (BPNet)

Due to BPNet requiring incompatible libraries with MAVE-NN, the current script is separated
into two parts (with switches 'STEP 1' and 'STEP 2'). STEP 1 requires an activated BPNet environment.
Once outputs are saved to file, deactivate the environment and activate the MAVE-NN environment.
Finally, turn off the STEP 1 switch below (i.e., 'if 0:') and rerun this script.

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


def op(py_dir, step):
    """Function to switch between STEP 1 and STEP 2 using command line arguments.
    All hyperparameters should be adjusted inside this function before running.

    Parameters
    ----------
    step : int
        either 1 or 2 for STEP 1 (BPNet) or STEP 2 (MAVE-NN), respectively
    """
    gpu = False
    parent_dir = os.path.dirname(py_dir)

    # define a global pattern (i.e., conserved sequence of interest)
    if 0:
        task_idx = 'Oct4' # bpnet task index ('Oct4', 'Sox2', 'Klf4' or 'Nanog')
        pattern = 'TTTGCAT' # Oct4 binding site
    elif 0:
        task_idx = 'Sox2' # bpnet task index ('Oct4', 'Sox2', 'Klf4' or 'Nanog')
        pattern = 'GAACAATAG' # Sox2 binding site
    elif 0:
        task_idx = 'Klf4' # bpnet task index ('Oct4', 'Sox2', 'Klf4' or 'Nanog')
        pattern = 'GGGTGTGGC' # Klf4 binding site
    elif 1:
        task_idx = 'Nanog' # bpnet task index ('Oct4', 'Sox2', 'Klf4' or 'Nanog')
        pattern = 'AGCCATCAA' # Nanog binding site

    alphabet = ['A','C','G','T']
    seq_length = 1000 # full sequence length of bpnet inputs

    # create initial sequence with global pattern surrounded by random background
    start_pos = int(seq_length//2) # position of inserted pattern in background DNA
    bg_left = random.choices(str(''.join(alphabet)), k=start_pos) # random DNA background on LHS
    bg_right = random.choices(str(''.join(alphabet)), k=start_pos-len(pattern)) # random DNA background on RHS
    seq = ''.join(bg_left) + pattern + ''.join(bg_right)
    mut_window = [start_pos, start_pos+len(pattern)] # interval in sequence to mutagenize (locally)

    save_dir = os.path.join(py_dir, 'outputs_global_additive/%s' % task_idx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if step == 1: # STEP 1 (BPNet)
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
                                alphabet=alphabet, context_agnostic=True) # this line required for global analysis 
        x_mut, y_mut = mave.generate(x, num_sim=num_sim)

        # save in silico MAVE dataset for STEP 2
        np.save(os.path.join(save_dir, 'x_mut.npy'), x_mut)
        np.save(os.path.join(save_dir, 'y_mut.npy'), y_mut)
        print('In silico MAVE dataset saved.')


    elif step == 2: # STEP 2 (MAVE-NN)
        import mavenn
        import squid
        import matplotlib.pyplot as plt

        # choose surrogate model type
        gpmap = 'additive'

        # load in in silico MAVE dataset from STEP 1
        x_mut = np.load(os.path.join(save_dir, 'x_mut.npy'))
        y_mut = np.load(os.path.join(save_dir, 'y_mut.npy'))

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
        view_window = [start_pos-15, start_pos+len(pattern)+15]
        fig = squid.impress.plot_additive_logo(params[1], center=True, view_window=view_window, alphabet=alphabet, fig_size=[20,2.5], save_dir=save_dir)



if __name__ == '__main__':
    py_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        step = int(sys.argv[1])
        if step not in [1, 2]:
            print('Argument for the current step must be 1 or 2.')
            sys.exit(0)
    else:
        print('Script must be run with correct arguments.')
        sys.exit(0)
    op(py_dir, step)