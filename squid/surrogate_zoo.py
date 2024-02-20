import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.regularizers import l1_l2
try:
    import mavenn
except ImportError:
    pass
try:
    from sklearn.linear_model import Lasso, LassoLarsCV, RidgeCV
except ImportError:
    pass



class SurrogateBase():
    """
    Base class for surrogate model.
    """
    def __init__(self):
        pass

    def train(self, x):
        raise NotImplementedError()

    def get_params(self, x):
        raise NotImplementedError()



class SurrogateLinear(SurrogateBase):
    """Module for linear surrogate model (no GE or noise models).

    Parameters
    ----------
    l1 : float, optional
        L1 regularization penalty
    l2 : float, optional
        L2 regularization penalty

    Returns
    -------
    keras.Model  
    """
    def __init__(self, input_shape, num_tasks, l1=1e-8, l2=1e-4,
                 alphabet=['A','C','G','T']):

        self.model = self.build(input_shape, num_tasks, l1, l2)
        self.alphabet = alphabet


    def build(self, input_shape, num_tasks, l1, l2):
        """Build linear surrogate model."""
        
        self.N, self.L, self.A = input_shape

        # input layer
        inputs = keras.layers.Input(shape=(self.L,self.A))
        flatten = keras.layers.Flatten()(inputs)
        outputs = keras.layers.Dense(num_tasks,
                                     activation='linear',
                                     kernel_regularizer=l1_l2(l1=l1, l2=l2),
                                     use_bias=True)(flatten)

        # compile model
        return keras.Model(inputs=inputs, outputs=outputs)


    def train(self, x, y, learning_rate=1e-3, epochs=500, batch_size=100, early_stopping=True,
              patience=25, restore_best_weights=True, rnd_seed=None, save_dir=None, verbose=1):
        """Train linear surrogate model."""

        # generate data splits
        train_index, valid_index, test_index = data_splits(x.shape[0], test_split=0.1, valid_split=0.1, rnd_seed=rnd_seed)
        x_train = x[train_index]
        y_train = y[train_index]
        x_valid = x[valid_index]
        y_valid = y[valid_index]
        x_test = x[test_index]
        y_test = y[test_index]

        # set up optimizer and metrics
        self.model.compile(keras.optimizers.Adam(learning_rate), loss='mse')

        # early stopping callback
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    verbose=1,
                                                    mode='min',
                                                    restore_best_weights=restore_best_weights)

        # reduce learning rate callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.2,
                                                      patience=3,
                                                      min_lr=1e-7,
                                                      mode='min',
                                                      verbose=verbose)

        # fit model to data
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 validation_data=(x_valid, y_valid),
                                 callbacks=[es_callback, reduce_lr],
                                 verbose=verbose)
        
        if save_dir is not None:
            self.model.save(os.path.join(save_dir, 'linear_model'))

        return (self.model, None)


    def get_params(self, gauge=None, save_dir=None):
        """Function to return trained parameters from the linear model.

        Parameters
        ----------
        gauge : None
            None, to match output of MAVE-NN get_params()
        save_dir : str
            Directory for saving figures to file.
        
        Returns
        -------
        tuple 
            theta_0     :   None
                None, to match output of MAVE-NN get_params()
            theta_lc    :   numpy.ndarray
                Additive terms in trained parameters (shape : (L,C)).
            theta_lclc  :   None
                None, to match output of MAVE-NN get_params()
        """

        for layer in self.model.layers:
            weights = layer.get_weights()

        theta_lc = self.model.layers[2].get_weights()[0]
        theta_lc = theta_lc.reshape((self.L, self.A))

        if save_dir is not None:
            np.save(os.path.join(save_dir, 'theta_lc.npy'), theta_lc)

        return (None, theta_lc, None)
    

    def get_logo(self, full_length=None, view_window=None):

        # insert the (potentially-delimited) additive logo back into the max-length sequence
        if full_length is None:
            full_length = self.L
        additive_logo = self.get_params(self.model)[1]
        if view_window is not None:
            additive_logo_zeros = np.zeros(shape=(full_length, self.A))
            additive_logo_zeros[view_window[0]:view_window[1], :] = additive_logo
            additive_logo = additive_logo_zeros

        return additive_logo
    

class SurrogateLasso(SurrogateBase):
    """Module for linear surrogate model (no GE or noise models) using sklearn Lasso.
    For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization strength;
        alpha must be a non-negative float i.e. in [0, inf). When alpha = 0, 
        the objective is equivalent to ordinary least squares, solved by the
        LinearRegression object. For numerical reasons, using alpha = 0 with the 
        Lasso object is not advised. Instead, you should use the LinearRegression object.

    Returns
    -------
    sklearn.Model
    """
    def __init__(self, input_shape, num_tasks, alpha=1,
                 deduplicate=True, alphabet=['A','C','G','T'], gpu=True):

        self.N, self.L, self.A = input_shape
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.alphabet = alphabet
        self.deduplicate = deduplicate
        self.gpu = gpu


    def dataframe(self, x, y, alphabet, gpu):
        
        N = x.shape[0]
        mave_df = pd.DataFrame(columns = ['y', 'x'], index=range(N))
        mave_df['y'] = y
        
        if gpu is False:
            alphabet_dict = {}
            idx = 0
            for i in range(len(alphabet)):
                alphabet_dict[i] = alphabet[i]
                idx += 1
            for i in range(N): #standard approach
                seq_index = np.argmax(x[i,:,:], axis=1)
                seq = []
                for s in seq_index:
                    seq.append(alphabet_dict[s])
                seq = ''.join(seq)
                mave_df.at[i, 'x'] = seq

        elif gpu is True: # convert entire matrix at once (~twice as fast as standard approach if running on GPUs)
            seq_index_all = np.argmax(x, axis=-1)
            num2alpha = dict(zip(range(0, len(alphabet)), alphabet))
            seq_vector_all = np.vectorize(num2alpha.get)(seq_index_all)
            seq_list_all = seq_vector_all.tolist()
            dictionary_list = []
            for i in range(0, N, 1):
                dictionary_data = {0: ''.join(seq_list_all[i])}
                dictionary_list.append(dictionary_data)
            mave_df['x'] = pd.DataFrame.from_dict(dictionary_list)

        return mave_df
    

    def train(self, x, y, learning_rate=None, epochs=None, batch_size=None, early_stopping=None,
              patience=None, restore_best_weights=None, rnd_seed=None, save_dir=None, verbose=1):

        # convert matrix of one-hots into sequence dataframe
        if verbose:
            print('  Creating sequence dataframe...')
            print('')
        mave_df = self.dataframe(x, y, alphabet=self.alphabet, gpu=self.gpu)
        if verbose:
            print(mave_df)

        if self.deduplicate:
            mave_df.drop_duplicates(['y', 'x'], inplace=True, keep='first')

        x = x.reshape(x.shape[0], -1)
        self.model = Lasso(alpha=self.alpha).fit(x, y)

        return (self.model, mave_df)
    

    def get_params(self, gauge=None, save_dir=None):
        """Function to return trained parameters from the Lasso model.

        Parameters
        ----------
        gauge : None
            None, to match output of MAVE-NN get_params()
        save_dir : str
            Directory for saving figures to file.
        
        Returns
        -------
        tuple 
            theta_0     :   None
                None, to match output of MAVE-NN get_params()
            theta_lc    :   numpy.ndarray
                Additive terms in trained parameters (shape : (L,C)).
            theta_lclc  :   None
                None, to match output of MAVE-NN get_params()
        """
        coef = self.model.coef_
        #yhat = self.model.predict(x) # run inference on dataset

        theta_lc = coef.reshape((self.L, self.A))

        if save_dir is not None:
            np.save(os.path.join(save_dir, 'theta_lc.npy'), theta_lc)

        return (None, theta_lc, None)
    

    def get_logo(self, full_length=None, view_window=None):

        # insert the (potentially-delimited) additive logo back into the max-length sequence
        if full_length is None:
            full_length = self.L
        additive_logo = self.get_params(self.model)[1]
        if view_window is not None:
            additive_logo_zeros = np.zeros(shape=(full_length, self.A))
            additive_logo_zeros[view_window[0]:view_window[1], :] = additive_logo
            additive_logo = additive_logo_zeros

        return additive_logo
    


class SurrogateRidgeCV(SurrogateBase):
    """Module for linear surrogate model (no GE or noise models) using sklearn RidgeCV.
    For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try. Regularization strength; must be a positive float.
        Regularization improves the conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization. Alpha corresponds to 1 / (2C) in other linear models
        such as LogisticRegression or LinearSVC. If using Leave-One-Out cross-validation, alphas must be positive.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the efficient Leave-One-Out cross-validation
            - integer, to specify the number of folds
            - CV splitter
            - An iterable yielding (train, test) splits as arrays of indices.

    Returns
    -------
    sklearn.Model
    """
    def __init__(self, input_shape, num_tasks, alphas=[1e-3, 1e-2, 1e-1, 1], cv=5,
                 deduplicate=True, alphabet=['A','C','G','T'], gpu=True):

        self.N, self.L, self.A = input_shape
        self.num_tasks = num_tasks
        self.alphas = alphas
        self.cv = cv
        self.alphabet = alphabet
        self.deduplicate = deduplicate
        self.gpu = gpu


    def dataframe(self, x, y, alphabet, gpu):
        
        N = x.shape[0]
        mave_df = pd.DataFrame(columns = ['y', 'x'], index=range(N))
        mave_df['y'] = y
        
        if gpu is False:
            alphabet_dict = {}
            idx = 0
            for i in range(len(alphabet)):
                alphabet_dict[i] = alphabet[i]
                idx += 1
            for i in range(N): #standard approach
                seq_index = np.argmax(x[i,:,:], axis=1)
                seq = []
                for s in seq_index:
                    seq.append(alphabet_dict[s])
                seq = ''.join(seq)
                mave_df.at[i, 'x'] = seq

        elif gpu is True: # convert entire matrix at once (~twice as fast as standard approach if running on GPUs)
            seq_index_all = np.argmax(x, axis=-1)
            num2alpha = dict(zip(range(0, len(alphabet)), alphabet))
            seq_vector_all = np.vectorize(num2alpha.get)(seq_index_all)
            seq_list_all = seq_vector_all.tolist()
            dictionary_list = []
            for i in range(0, N, 1):
                dictionary_data = {0: ''.join(seq_list_all[i])}
                dictionary_list.append(dictionary_data)
            mave_df['x'] = pd.DataFrame.from_dict(dictionary_list)

        return mave_df
    

    def train(self, x, y, learning_rate=None, epochs=None, batch_size=None, early_stopping=None,
              patience=None, restore_best_weights=None, rnd_seed=None, save_dir=None, verbose=1):

        # convert matrix of one-hots into sequence dataframe
        if verbose:
            print('  Creating sequence dataframe...')
            print('')
        mave_df = self.dataframe(x, y, alphabet=self.alphabet, gpu=self.gpu)
        if verbose:
            print(mave_df)

        if self.deduplicate:
            mave_df.drop_duplicates(['y', 'x'], inplace=True, keep='first')

        x = x.reshape(x.shape[0], -1)
        self.model = RidgeCV(alphas=self.alphas, cv=self.cv).fit(x, y)

        return (self.model, mave_df)
    

    def get_params(self, gauge=None, save_dir=None):
        """Function to return trained parameters from the RidgeCV model.

        Parameters
        ----------
        gauge : None
            None, to match output of MAVE-NN get_params()
        save_dir : str
            Directory for saving figures to file.
        
        Returns
        -------
        tuple 
            theta_0     :   None
                None, to match output of MAVE-NN get_params()
            theta_lc    :   numpy.ndarray
                Additive terms in trained parameters (shape : (L,C)).
            theta_lclc  :   None
                None, to match output of MAVE-NN get_params()
        """
        coef = self.model.coef_
        #yhat = self.model.predict(x) # run inference on dataset

        theta_lc = coef.reshape((self.L, self.A))

        if save_dir is not None:
            np.save(os.path.join(save_dir, 'theta_lc.npy'), theta_lc)

        return (None, theta_lc, None)
    

    def get_logo(self, full_length=None, view_window=None):

        # insert the (potentially-delimited) additive logo back into the max-length sequence
        if full_length is None:
            full_length = self.L
        additive_logo = self.get_params(self.model)[1]
        if view_window is not None:
            additive_logo_zeros = np.zeros(shape=(full_length, self.A))
            additive_logo_zeros[view_window[0]:view_window[1], :] = additive_logo
            additive_logo = additive_logo_zeros

        return additive_logo



class SurrogateLIME(SurrogateBase):
    """Module for linear surrogate model (no GE or noise models) using LIME.
    For more information, see https://arxiv.org/pdf/1602.04938.pdf and
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html

    Parameters
    ----------
    k : int
        The desired number of nonzero weights

    Returns
    -------
    sklearn.Model
    """
    def __init__(self, input_shape, num_tasks, k=20,
                 deduplicate=True, alphabet=['A','C','G','T'], gpu=True):

        self.N, self.L, self.A = input_shape
        self.num_tasks = num_tasks
        self.k = k
        self.alphabet = alphabet
        self.deduplicate = deduplicate
        self.gpu = gpu


    def dataframe(self, x, y, alphabet, gpu):
        
        N = x.shape[0]
        mave_df = pd.DataFrame(columns = ['y', 'x'], index=range(N))
        mave_df['y'] = y
        
        if gpu is False:
            alphabet_dict = {}
            idx = 0
            for i in range(len(alphabet)):
                alphabet_dict[i] = alphabet[i]
                idx += 1
            for i in range(N): #standard approach
                seq_index = np.argmax(x[i,:,:], axis=1)
                seq = []
                for s in seq_index:
                    seq.append(alphabet_dict[s])
                seq = ''.join(seq)
                mave_df.at[i, 'x'] = seq

        elif gpu is True: # convert entire matrix at once (~twice as fast as standard approach if running on GPUs)
            seq_index_all = np.argmax(x, axis=-1)
            num2alpha = dict(zip(range(0, len(alphabet)), alphabet))
            seq_vector_all = np.vectorize(num2alpha.get)(seq_index_all)
            seq_list_all = seq_vector_all.tolist()
            dictionary_list = []
            for i in range(0, N, 1):
                dictionary_data = {0: ''.join(seq_list_all[i])}
                dictionary_list.append(dictionary_data)
            mave_df['x'] = pd.DataFrame.from_dict(dictionary_list)

        return mave_df
    

    def train(self, x, y, learning_rate=None, epochs=None, batch_size=None, early_stopping=None,
              patience=None, restore_best_weights=None, rnd_seed=None, save_dir=None, verbose=1):

        # convert matrix of one-hots into sequence dataframe
        if verbose:
            print('  Creating sequence dataframe...')
            print('')
        mave_df = self.dataframe(x, y, alphabet=self.alphabet, gpu=self.gpu)
        if verbose:
            print(mave_df)

        if self.deduplicate:
            mave_df.drop_duplicates(['y', 'x'], inplace=True, keep='first')

        x = x.reshape(x.shape[0], -1)

        models = LassoLarsCV(cv=None).fit(x, y)

        alphas = models.alphas_
        nonzero_weights = np.apply_along_axis(np.count_nonzero, 0, models.coef_path_)
        selected_alpha_idx = np.where(nonzero_weights == self.k)[0][0]
        selected_alpha = alphas[selected_alpha_idx]

        # get the coefficients of the model with k nonzero features
        coef = models.coef_path_[:,selected_alpha_idx]

        # mask out all zeroed features
        zeros_index = np.where(coef==0)
        x_masked = x.copy()
        x_masked[:, zeros_index] = 0

        # refit the data on the masked set using least squares linear regression
        self.model = Lasso(alpha=0.).fit(x_masked, y)

        return (self.model, mave_df)
    

    def get_params(self, gauge=None, save_dir=None):
        """Function to return trained parameters from the RidgeCV model.

        Parameters
        ----------
        gauge : None
            None, to match output of MAVE-NN get_params()
        save_dir : str
            Directory for saving figures to file.
        
        Returns
        -------
        tuple 
            theta_0     :   None
                None, to match output of MAVE-NN get_params()
            theta_lc    :   numpy.ndarray
                Additive terms in trained parameters (shape : (L,C)).
            theta_lclc  :   None
                None, to match output of MAVE-NN get_params()
        """
        coef = self.model.coef_
        #yhat = self.model.predict(x) # run inference on dataset

        theta_lc = coef.reshape((self.L, self.A))

        if save_dir is not None:
            np.save(os.path.join(save_dir, 'theta_lc.npy'), theta_lc)

        return (None, theta_lc, None)
    

    def get_logo(self, full_length=None, view_window=None):

        # insert the (potentially-delimited) additive logo back into the max-length sequence
        if full_length is None:
            full_length = self.L
        additive_logo = self.get_params(self.model)[1]
        if view_window is not None:
            additive_logo_zeros = np.zeros(shape=(full_length, self.A))
            additive_logo_zeros[view_window[0]:view_window[1], :] = additive_logo
            additive_logo = additive_logo_zeros

        return additive_logo



class SurrogateMAVENN(SurrogateBase):
    """Module for MAVE-NN surrogate models (optional GE and noise models).

    Parameters
    ----------
    gpmap : string {'additive' or 'pairwise'}
        Define MAVE-NN surrogate model used to interpret deep learning model.
            'additive'  :   Assume that each position contributes independently to the latent phenotype.
            'pairwise'  :   Assume that every pair of positions contribute to the latent phenotype.
    regression_type : string
        Type of regression used for measurement process.
            'MPA'   :   measurement process agnostic (categorical y-values).
            'GE'    :   global epistasis (continuous y-values).
    linearity : string
        Define use of additional nonlinearity for fitting data.
            'nonlinear' :   Additionally fit data using GE nonlinear function.
            'linear'    :   Do not apply GE nonlinearity for fitting data.
    noise : string
        Noise model to use for when defining a GE model (no effect on MPA models).
        See https://mavenn.readthedocs.io/en/latest/math.html for more info.
            'Gaussian'  :   Gaussian-based noise model.
            'Cauchy'    :   Cauchy-based noise model.
            'SkewedT'   :   SkewedT-based noise model.
    noise_order : int
        In the GE context, the order of the polynomial(s) used to define noise model parameters.
        In the linear context, the order is zero by default.
    reg_strength : float
        L2 regularization strength for G-P map parameters.
    hidden_nodes : int
        Number of hidden nodes (i.e. sigmoidal contributions) to use when defining the nonlinearity component of a GE model.
        Has no effect on MPA models.
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.
    deduplicate : boole
        Remove duplicate sequence-function pairs in dataset (default: True).
    gpu : boolean
        Enable GPUs (default: True).

    Returns
    -------
    keras.Model
    """
    def __init__(self, input_shape, num_tasks, gpmap='additive', regression_type='GE',
                 linearity='nonlinear', noise='SkewedT', noise_order=2, reg_strength=0.1,
                 hidden_nodes=50, alphabet=['A','C','G','T'], deduplicate=True, gpu=True):
        
        self.N, self.L, self.A = input_shape
        self.num_tasks = num_tasks
        self.gpmap = gpmap
        self.regression_type = regression_type
        self.linearity = linearity
        self.noise = noise
        if self.linearity == 'linear':
            self.noise_order = 0
        elif self.linearity == 'nonlinear':
            self.noise_order = noise_order
        self.reg_strength = reg_strength
        self.hidden_nodes = hidden_nodes
        self.alphabet = alphabet
        self.deduplicate = deduplicate
        self.gpu = gpu
    

    def dataframe(self, x, y, alphabet, gpu):
        
        N = x.shape[0]
        mave_df = pd.DataFrame(columns = ['y', 'x'], index=range(N))
        mave_df['y'] = y
        
        if gpu is False:
            alphabet_dict = {}
            idx = 0
            for i in range(len(alphabet)):
                alphabet_dict[i] = alphabet[i]
                idx += 1
            for i in range(N): #standard approach
                seq_index = np.argmax(x[i,:,:], axis=1)
                seq = []
                for s in seq_index:
                    seq.append(alphabet_dict[s])
                seq = ''.join(seq)
                mave_df.at[i, 'x'] = seq

        elif gpu is True: # convert entire matrix at once (~twice as fast as standard approach if running on GPUs)
            seq_index_all = np.argmax(x, axis=-1)
            num2alpha = dict(zip(range(0, len(alphabet)), alphabet))
            seq_vector_all = np.vectorize(num2alpha.get)(seq_index_all)
            seq_list_all = seq_vector_all.tolist()
            dictionary_list = []
            for i in range(0, N, 1):
                dictionary_data = {0: ''.join(seq_list_all[i])}
                dictionary_list.append(dictionary_data)
            mave_df['x'] = pd.DataFrame.from_dict(dictionary_list)

        return mave_df

    
    def train(self, x, y, learning_rate=5e-4, epochs=500, batch_size=100,
              early_stopping=True, patience=25, restore_best_weights=True,
              save_dir=None, verbose=1):

        # convert matrix of one-hots into sequence dataframe
        if verbose:
            print('  Creating sequence dataframe...')
            print('')
        mave_df = self.dataframe(x, y, alphabet=self.alphabet, gpu=self.gpu)
        if verbose:
            print(mave_df)

        if self.deduplicate:
            mave_df.drop_duplicates(['y', 'x'], inplace=True, keep='first')

        # ensure proper format
        mave_df['y'] = mave_df['y'].apply(lambda x: np.asarray(x).astype('float32'))
 
        # split dataset for MAVE-NN
        mave_df['set'] = np.random.choice(a=['training','test','validation'],
                                        p=[.6,.2,.2],
                                        size=len(mave_df))
        new_cols = ['set'] + list(mave_df.columns[0:-2]) + ['x']
        mave_df = mave_df[new_cols]
        if save_dir is not None:
            mave_df.to_csv(os.path.join(save_dir, 'mave_df.csv'),  index=False) #.csv.gz', compression='gzip')

        trainval_df, self.test_df = mavenn.split_dataset(mave_df)
        self.y_cols = trainval_df.columns[1:-1] #get the column index for the counts
        
        # show dataset sizes
        if verbose:
            print(f'Train + val set size : {len(trainval_df):6,d} observations')
            print(f'Test set size        : {len(self.test_df):6,d} observations')
            print('')

        gpmap_kwargs = {'L': self.L,
                        'C': self.A,
                        'theta_regularization': self.reg_strength}
        
        if verbose:
            if self.gpmap == 'additive':
                print('Initializing additive model... %s parameters' % int(self.L*4))
            elif self.gpmap == 'neighbor':
                print('Initializing neighbor model... %s parameters' % int((self.L-1)*16))
            elif self.gpmap == 'pairwise':
                print('Initializing pairwise model... %s parameters' % int((self.L*(self.L-1)*16)/2))
            print('')
        
        # create the model
        if self.regression_type == 'MPA':
            self.model = mavenn.Model(L=self.L,
                                Y=len(self.y_cols),
                                alphabet=self.alphabet,
                                ge_nonlinearity_type=self.linearity,
                                regression_type='MPA',
                                gpmap_type=self.gpmap, 
                                gpmap_kwargs=gpmap_kwargs)
            y_data = trainval_df[self.y_cols]
            
        elif self.regression_type == 'GE':
            self.model = mavenn.Model(L=self.L,
                                Y=len(self.y_cols),
                                alphabet=self.alphabet,
                                ge_nonlinearity_type=self.linearity,
                                regression_type='GE',
                                ge_noise_model_type=self.noise,
                                ge_heteroskedasticity_order=self.noise_order, 
                                gpmap_type=self.gpmap,
                                ge_nonlinearity_hidden_nodes=self.hidden_nodes,
                                gpmap_kwargs=gpmap_kwargs)
            y_data = trainval_df['y']
        
        # set training data
        self.model.set_data(x=trainval_df['x'], y=y_data,
                        validation_flags=trainval_df['validation'],
                        shuffle=True)
        
        # fit model to data
        self.model.fit(learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                early_stopping_patience=patience,
                linear_initialization=False,
                #restore_best_weights=self.restore_best_weights,
                verbose=False)
        
        if save_dir is not None:
            self.model.save(os.path.join(save_dir, 'mavenn_model_%s' % self.gpmap))

        return (self.model, mave_df)


    def get_info(self, save_dir=None, verbose=1):
        """Function to return estimated variational information from MAVE-NN model.

        Parameters
        ----------
        save_dir : str
            Directory for saving figures to file.
        verbose : bool
            print info
        Returns
        -------
        I_pred : float
            MAVE-NN estimated variational information (I_pred), in bits.
        """
        # compute predictive information on test data
        if self.regression_type == 'MPA':
            I_pred, dI_pred = self.model.I_predictive(x=self.test_df['x'], y=self.test_df[self.y_cols])
        elif self.regression_type == 'GE':
            I_pred, dI_pred = self.model.I_predictive(x=self.test_df['x'], y=self.test_df['y'])

        if verbose:
            print('')
            print('Model performance:')
            print(f'  test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits')
            print('  max I_var:', np.amax(self.model.history['I_var']))
            print('  max val_I_var:', np.amax(self.model.history['val_I_var']))
            print('')

        if save_dir is not None:
            # save information content to text file
            if os.path.exists(os.path.join(save_dir, 'model_info.txt')):
                os.remove(os.path.join(save_dir,'model_info.txt'))
            f_out = open(os.path.join(save_dir, 'model_info.txt'),'w')
            print(f'test_I_pred: {I_pred:.3f} bits', file=f_out)
            print('max I_var:',np.amax(self.model.history['I_var']), file=f_out)
            print('max val_I_var:',np.amax(self.model.history['val_I_var']), file=f_out)
            print('', file=f_out)
            f_out.close()
        
        return I_pred


    def get_params(self, gauge='empirical', save_dir=None):
        """Function to return trained parameters from MAVE-NN model.

        Parameters
        ----------
        gauge : gauge mode used to fix model parameters.
                See https://mavenn.readthedocs.io/en/latest/math.html for more info.
            'uniform'   :   hierarchical gauge using a uniform sequence distribution over
                            the characters at each position observed in the training set
                            (unobserved characters are assigned probability 0).
            'empirical' :   uses an empirical distribution computed from the training data.
            'consensus' :   wild-type gauge using the training data consensus sequence.
        save_dir : str
            Directory for saving figures to file.
        
        Returns
        -------
        tuple 
            theta_0     :   float
                Constant term in trained parameters.
            theta_lc    :   numpy.ndarray
                Additive terms in trained parameters (shape : (L,C)).
            theta_lclc  :   numpy.ndarray
                Pairwise terms in trained parameters (shape : (L,C,L,C)), if gpmap is 'pairwise'.
        """
        
        # fix gauge mode for model representation
        self.theta_dict = self.model.get_theta(gauge=gauge) #for usage: theta_dict.keys()
        
        theta_0 = self.theta_dict['theta_0']
        theta_lc = self.theta_dict['theta_lc']
        theta_lc[np.isnan(theta_lc)] = 0
        if save_dir is not None:
            np.save(os.path.join(save_dir, 'theta_0.npy'), theta_0)
            np.save(os.path.join(save_dir, 'theta_lc.npy'), theta_lc)

        if self.gpmap == 'pairwise':
            theta_lclc = self.theta_dict['theta_lclc']
            theta_lclc[np.isnan(theta_lclc)] = 0
            if save_dir is not None:
                np.save(os.path.join(save_dir, 'theta_lclc.npy'), theta_lclc)
        else:
            theta_lclc = None

        return (theta_0, theta_lc, theta_lclc)


    def get_logo(self, full_length=None, view_window=None):
        """Function to place trained additive parameters into surrounding 
        nonmutated sequence (zeros).

        Parameters
        ----------
        full_length : int
            Full length of sequence.
        view_window : [int, int]
            Index of start and stop position along sequence to probe;
            i.e., [start, stop], where start < stop and both entries
            satisfy 0 <= int <= L.

        Returns
        -------
        additive_logo : numpy.ndarray
            Additive logo parameters (shape : ('full_length',C)).
        """
        # insert the (potentially-delimited) additive logo back into the max-length sequence
        if full_length is None:
            full_length = self.L
        additive_logo = self.theta_dict['logomaker_df']
        additive_logo.fillna(0, inplace=True) #if necessary, set NaN parameters to zero
        if view_window is not None:
            additive_logo_zeros = np.zeros(shape=(full_length, self.A))
            additive_logo_zeros[view_window[0]:view_window[1], :] = additive_logo
            additive_logo = additive_logo_zeros

        return additive_logo


def data_splits(N, test_split, valid_split, rnd_seed=None):
    """Function to determine which sequences randomly split into train, validation, test set.

    Parameters
    ----------
    N : int
        number of data
    test_splitc : float (between 0 and 1)
        percent to split into test set
    valid_split : float (between 0 and 1)
        percent to split into validation set
    rnd_seed : int
        random number seed

    Returns
    -------
    train_index
        array of indices to be included in training set
    valid_index
        array of indices to be included in validation set
    test_index
        array of indices to be included in test set
    """
    if rnd_seed:
        np.random.seed(rnd_seed)
    train_split = 1 - test_split - valid_split
    shuffle = np.random.permutation(range(N))
    num_valid = int(valid_split*N)
    num_test = int(test_split*N)
    test_index = shuffle[:num_test]
    valid_index = shuffle[num_test:num_test+num_valid]
    train_index = shuffle[num_test+num_valid:]
    return train_index, valid_index, test_index


