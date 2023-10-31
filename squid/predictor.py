import numpy as np
from tqdm import tqdm


class BasePredictor():
    """
    Base class for running inference on in silico mutated sequences.
    """

    def __init__(self):
        raise NotImplementedError()

    def __call__(self, x):
        raise NotImplementedError()



class ScalarPredictor(BasePredictor):
    """Module for handling scalar-based model predictions.

    Parameters
    ----------
    pred_fun : built-in function
        Function for returning model predictions.
    task_idx : int
        Task index corresponding to a specific output head.
    batch_size : int
        The number of predictions per batch.

    Returns
    -------
    torch.Tensor
        Batch of scalar predictions corresponding to inputs.
    """

    def __init__(self, pred_fun, task_idx=0, batch_size=64, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.kwargs = kwargs
        self.batch_size = batch_size

    def __call__(self, x):
        pred = predict_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)
        return pred[self.task_idx]



class ProfilePredictor(BasePredictor):
    """Module for handling profile-based model predictions.

    Parameters
    ----------
    pred_fun : function
        Function for returning model predictions.
    task_idx : int
        Task index corresponding to a specific output head.
    batch_size : int
        The number of predictions per batch.
    reduce_fun : function
        Function for reducing profile prediction to scalar.

    Returns
    -------
    torch.Tensor
        Batch of scalar predictions corresponding to inputs.
    """

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun=np.sum, save_dir=None, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        #self.axis = axis
        self.save_dir = save_dir
        self.kwargs = kwargs

    def __call__(self, x):
        # get model predictions (all tasks)
        pred = predict_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)

        # reduce profile to scalar across axis for a given task_idx
        pred = self.reduce_fun(pred[:,:,self.task_idx], save_dir=self.save_dir)
        return pred[:,np.newaxis]



'''
class BPNetPredictor(BasePredictor):

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun=np.sum, axis=1, strand='pos', **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        self.axis = axis
        self.kwargs = kwargs
        if strand == 'pos':
            self.strand = 0
        else:
            self.strand = 1

    def __call__(self, x):

        # get model predictions (all tasks)
        pred = predict_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)

        # reduce bpnet profile prediction to scalar across axis for a given task_idx
        pred = pred[self.task_idx][0][:,self.strand]
        pred = self.reduce_fun(pred, axis=self.axis)
        return pred[:,np.newaxis]
'''



'''
class CustomPredictor():

    def __init__(self, pred_fun, reduce_fun, task_idx, batch_size):
        self.pred_fun = pred_fun
        self.reduce_fun = reduce_fun
        self.task_idx = task_idx
        self.batch_size = batch_size

    def __call__(self, x):
        # code goes here: process predictions into scalar
        return predictions

'''

################################################################################
# useful functions
################################################################################


def predict_in_batches(x, model_pred_fun, batch_size=None, **kwargs):
    """Function to compute model predictions in batch mode.

    Parameters
    ----------
    x : torch.Tensor
        One-hot sequences (shape: (N, L, A)).
    model_pred_fun : function
        Built-in function for accessing model inference on inputs.
    batch_size : int
        The number of predictions per batch of model inference.

    Returns
    -------
    numpy.ndarray
        Model predictions.
    """

    N, L, A = x.shape
    num_batches = np.floor(N/batch_size).astype(int)
    pred = []
    for i in tqdm(range(num_batches), desc="Inference"):
        pred.append(model_pred_fun(x[i*batch_size:(i+1)*batch_size], **kwargs))
    if num_batches*batch_size < N:
        pred.append(model_pred_fun(x[num_batches*batch_size:], **kwargs))

    try:
        preds = np.concatenate(pred, axis=1)
    except ValueError as ve:
        preds = np.vstack(pred)
    return preds



def profile_sum(pred, save_dir=None):
    """Function to transform predictions to scalars using summation.

    Parameters
    ----------
    pred : np.ndarray
        Batch of profile-based predictions.

    Returns
    -------
    numpy.ndarray
        Batch of scalar predictions.
    """

    sum = np.sum(pred, axis=1)
    return sum


def profile_pca(pred, save_dir=None):
    """Function to transform predictions to scalars using principal component analysis (PCA).

    Parameters
    ----------
    pred : np.ndarray
        Batch of profile-based predictions.

    Returns
    -------
    numpy.ndarray
        Batch of scalar predictions formed from projection of embedded
        profiles onto the first principal component.
    """

    N, B = pred.shape #B : number of bins in profile
    Y = pred.copy()
    sum = np.sum(pred, axis=1) #needed for sense correction

    # normalization: mean of all distributions is subtracted from each distribution
    mean_all = np.mean(Y, axis=0)
    for i in range(N):
        Y[i,:] -= mean_all

    u,s,v = np.linalg.svd(Y.T, full_matrices=False)
    vals = s**2 #eigenvalues
    vecs = u #eigenvectors
    
    U = Y.dot(vecs)
    v1, v2 = 0, 1
    
    corr = np.corrcoef(sum, U[:,v1])
    if corr[0,1] < 0: #correct for eigenvector "sense"
        U[:,v1] = -1.*U[:,v1]

    #impress.plot_eig_vals(vals, save_dir=save_dir)
    #impress.plot_eig_vecs(U, v1=v1, v2=v2, save_dir=save_dir)

    return U[:,v1]


"""
def custom_reduce(pred):
    # code to reduce predictions to (N,1)
    return pred_reduce
"""


