import os
#os.environ["TQDM_DISABLE"] = "1"
from tqdm import tqdm
import numpy as np
try:
    import impress
except ImportError:
    pass


class BasePredictor():
    """
    Base class for running inference on in silico mutated sequences.
    """
    save_dir = None
    
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, x, x_ref):
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

    def __init__(self, pred_fun, task_idx=0, batch_size=64, save_window=None, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.kwargs = kwargs
        self.batch_size = batch_size

    def __call__(self, x, x_ref, save_window):
        pred = predict_in_batches(x, x_ref, self.pred_fun, batch_size=self.batch_size, save_window=save_window, **self.kwargs)
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

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun=np.sum, save_dir=None, save_window=None, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        BasePredictor.save_dir = save_dir
        self.kwargs = kwargs

    def __call__(self, x, x_ref, save_window):
        # get model predictions (all tasks)
        pred = predict_in_batches(x, x_ref, self.pred_fun, batch_size=self.batch_size, save_window=save_window, **self.kwargs)

        # reduce profile to scalar across axis for a given task_idx
        pred = self.reduce_fun(pred[:,:,self.task_idx], save_dir=self.save_dir)
        return pred[:,np.newaxis]




class BPNetPredictor(BasePredictor):
    """Module for handling BPNet (Kipoi) model predictions.

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

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun='wn', axis=1, save_dir=None, save_window=None, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        self.axis = axis
        BasePredictor.save_dir = save_dir
        self.kwargs = kwargs

        if self.reduce_fun == 'wn': # transformation used in the original BPNet paper
            def contribution_score(preds):
                pred_scalars = np.zeros(preds.shape[0])
                import os
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
                import tensorflow as tf
                for pred_idx in tqdm(range(preds.shape[0]), desc='Compression'):
                    graph = tf.Graph()
                    pred = preds[pred_idx]
                    with graph.as_default():
                        wn = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(pred)) * pred, axis=-2), axis=-1)
                    with tf.Session(graph=graph).as_default() as sess:
                        pred_scalars[pred_idx] = float(wn.eval())              
                return pred_scalars
            self.reduce_fun = contribution_score

    def __call__(self, x, x_ref, save_window):
        # get model predictions (all tasks)
        pred = predict_in_batches(x, x_ref, self.pred_fun, batch_size=self.batch_size, task_idx=self.task_idx, save_window=save_window, **self.kwargs)

        # reduce profile prediction to scalar across axis for a given task_idx
        pred = self.reduce_fun(pred)

        return pred[:,np.newaxis]



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


def predict_in_batches(x, x_ref, model_pred_fun, batch_size=None, task_idx=None, save_window=None, **kwargs):
    """Function to compute model predictions in batch mode.

    Parameters
    ----------
    x : numpy.ndarray
        One-hot sequences (shape: (N, L, A)).
    x_ref : numpy.ndarray
        One-hot reference sequence (shape: (L, A)).
    model_pred_fun : function
        Built-in function for accessing model inference on inputs.
    batch_size : int
        The number of predictions per batch of model inference.
    save_window : [int, int]
        Window used for delimiting sequences that are exported in 'x_mut' array

    Returns
    -------
    numpy.ndarray
        Model predictions.
    """

    if save_window is not None:
        x_ref = x_ref[np.newaxis,:].astype('uint8')

    N, L, A = x.shape
    num_batches = np.floor(N/batch_size).astype(int)
    pred = []
    for i in tqdm(range(num_batches), desc="Inference"):
        x_batch = x[i*batch_size:(i+1)*batch_size]

        if save_window is not None:
            x_ref_start = np.broadcast_to(x_ref[:,:save_window[0],:], (x_batch.shape[0],save_window[0],x_ref.shape[2]))
            x_ref_stop = np.broadcast_to(x_ref[:,save_window[1]:,:], (x_batch.shape[0],x_ref.shape[1]-save_window[1],x_ref.shape[2]))
            x_batch = np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)

        p = model_pred_fun(x_batch.astype(float))
        if task_idx is not None:
            p = p[task_idx]
        pred.append(p, **kwargs)

    if num_batches*batch_size < N:
        x_batch = x[num_batches*batch_size:]

        if save_window is not None:
            x_ref_start = np.broadcast_to(x_ref[:,:save_window[0],:], (x_batch.shape[0],save_window[0],x_ref.shape[2]))
            x_ref_stop = np.broadcast_to(x_ref[:,save_window[1]:,:], (x_batch.shape[0],x_ref.shape[1]-save_window[1],x_ref.shape[2]))
            x_batch = np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)

        p = model_pred_fun(x_batch.astype(float))
        if task_idx is not None:
            p = p[task_idx]
        pred.append(p, **kwargs)

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
    if pred.ndim > 2:
        if 1: # flatten profile predictions across all profile dimensions
            dim_multiply = 1
            for dim in pred.shape[1:]:
                dim_multiply *= dim
            pred = pred.reshape(pred.shape[0], dim_multiply)
        else: # take only jth profile prediction
            pred = pred[:,:,0] # e.g., j=0 for positive strand profile (BPNet)

    N, B = pred.shape # number of bins (B) in profile
    Y = pred.copy()
    sum = np.sum(pred, axis=1) # needed for eigenvector sense correction

    # normalization: mean of all distributions is subtracted from each distribution
    mean_all = np.mean(Y, axis=0)
    for i in range(N):
        Y[i,:] -= mean_all

    u,s,v = np.linalg.svd(Y.T, full_matrices=False)
    vals = s**2 # eigenvalues
    vecs = u # eigenvectors
    
    U = Y.dot(vecs)
    v1, v2 = 0, 1
    
    corr = np.corrcoef(sum, U[:,v1])
    if corr[0,1] < 0: # correct sense of eigenvector
        U[:,v1] = -1.*U[:,v1]

    if BasePredictor.save_dir is not None:
        impress.plot_eig_vals(vals, save_dir=BasePredictor.save_dir)
        impress.plot_eig_vecs(U, v1=v1, v2=v2, save_dir=BasePredictor.save_dir)

    return U[:,v1]



"""
def custom_reduce(pred):
    # code to reduce predictions to (N,1)
    return pred_reduce
"""


