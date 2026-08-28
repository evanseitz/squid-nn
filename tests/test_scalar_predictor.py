"""Tests for ScalarPredictor task indexing.

Reproduces the DeepZebraFish-style bug: a single Keras head of shape
(batch, n_tasks) was indexed on axis 0 (a sequence) instead of the task axis.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from squid.mave import InSilicoMAVE
from squid.mutagenizer import RandomMutagenesis
from squid.predictor import BPNetPredictor, ScalarPredictor, predict_in_batches


N_TASKS = 639
TASK_IDX = 527
SEQ_LEN = 32
N_SEQS = 3000
BATCH_SIZE = 700  # 4 full batches + remainder 200, matching the report


def _onehot_seqs(n=N_SEQS, l=SEQ_LEN, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, 4, size=(n, l))
    x = np.eye(4, dtype=np.float32)[idx]
    return x


def deepzebrafish_like(x):
    """Single Keras head: (batch, 639), like DeepZebraFish peak regression."""
    scores = x.reshape(x.shape[0], -1).sum(axis=1)
    out = np.zeros((x.shape[0], N_TASKS), dtype=np.float32)
    out[:] = scores[:, None] + np.arange(N_TASKS)[None, :]
    return out


def deepstarr_multioutput(x):
    """Two Keras output heads, as in the kipoi-free SEAM/SQUID DeepSTARR examples."""
    scores = x.reshape(x.shape[0], -1).sum(axis=1)
    return [scores[:, None] + 0.0, scores[:, None] + 1.0]


def kipoi_deepstarr_array(x):
    """Single array of shape (batch, 2), Kipoi DeepSTARR schema (2,)."""
    scores = x.reshape(x.shape[0], -1).sum(axis=1)
    return np.stack([scores, scores + 1.0], axis=1)


def kipoi_bpnet_dict(x):
    """Kipoi BPNet-OSKN: dict of profiles, each (batch, bins, 2 strands)."""
    scores = x.reshape(x.shape[0], -1).sum(axis=1)
    n = x.shape[0]
    bins, strands = 100, 2
    def profile(offset):
        return np.broadcast_to(
            (scores + offset)[:, None, None], (n, bins, strands)
        ).copy()
    return {
        'Oct4': profile(0),
        'Sox2': profile(1),
        'Klf4': profile(2),
        'Nanog': profile(3),
    }


class TestScalarPredictor(unittest.TestCase):
    def test_single_head_selects_task_not_sequence(self):
        """Julian's case: (N, 639) model, task_idx=527, remainder batch."""
        x = _onehot_seqs()
        expected = deepzebrafish_like(x)[:, TASK_IDX]

        pred = ScalarPredictor(
            pred_fun=deepzebrafish_like,
            task_idx=TASK_IDX,
            batch_size=BATCH_SIZE,
        )
        y = pred(x)

        self.assertEqual(y.shape, (N_SEQS, 1), msg="y should be one scalar per sequence")
        np.testing.assert_allclose(y[:, 0], expected)
        # The old bug returned predictions[527] with shape (639,)
        self.assertNotEqual(y.shape, (N_TASKS,))

    def test_single_head_when_n_divides_batch_size(self):
        """Same model, but N is a multiple of batch_size (no remainder)."""
        n = 2800
        x = _onehot_seqs(n=n)
        expected = deepzebrafish_like(x)[:, TASK_IDX]

        y = ScalarPredictor(
            pred_fun=deepzebrafish_like,
            task_idx=TASK_IDX,
            batch_size=BATCH_SIZE,
        )(x)

        self.assertEqual(y.shape, (n, 1))
        np.testing.assert_allclose(y[:, 0], expected)

    def test_multioutput_heads_still_select_task(self):
        """DeepSTARR-style list of heads must keep working."""
        x = _onehot_seqs(n=1000)
        expected = deepstarr_multioutput(x)[1][:, 0]

        y = ScalarPredictor(
            pred_fun=deepstarr_multioutput,
            task_idx=1,
            batch_size=512,
        )(x)

        self.assertEqual(y.shape, (1000, 1))
        np.testing.assert_allclose(y[:, 0], expected)

    def test_predict_in_batches_task_idx_on_single_head(self):
        """Direct predict_in_batches(..., task_idx=527) used to IndexError on the remainder."""
        x = _onehot_seqs()
        expected = deepzebrafish_like(x)[:, TASK_IDX]

        y = predict_in_batches(
            x,
            model_pred_fun=deepzebrafish_like,
            batch_size=BATCH_SIZE,
            task_idx=TASK_IDX,
        )
        np.testing.assert_allclose(np.squeeze(y), expected)

    def test_kipoi_style_single_array_two_tasks(self):
        """Kipoi DeepSTARR-style (N, 2) array, not a list of heads."""
        x = _onehot_seqs(n=1000)
        expected = kipoi_deepstarr_array(x)[:, 0]

        y = ScalarPredictor(
            pred_fun=kipoi_deepstarr_array,
            task_idx=0,
            batch_size=512,
        )(x)

        self.assertEqual(y.shape, (1000, 1))
        np.testing.assert_allclose(y[:, 0], expected)

    def test_multioutput_when_n_equals_n_tasks(self):
        """List-of-heads stacked as (2, 2, 1) must not be treated as sample-first."""
        x = _onehot_seqs(n=2)
        expected = deepstarr_multioutput(x)[1][:, 0]
        y = ScalarPredictor(
            pred_fun=deepstarr_multioutput,
            task_idx=1,
            batch_size=64,
        )(x)
        self.assertEqual(y.shape, (2, 1))
        np.testing.assert_allclose(y[:, 0], expected)

    def test_tuple_of_heads(self):
        """Some Keras builds return a tuple of output tensors, not a list."""
        def pred_fun(x):
            return tuple(deepstarr_multioutput(x))

        x = _onehot_seqs(n=200)
        expected = deepstarr_multioutput(x)[1][:, 0]
        y = ScalarPredictor(pred_fun=pred_fun, task_idx=1, batch_size=64)(x)
        self.assertEqual(y.shape, (200, 1))
        np.testing.assert_allclose(y[:, 0], expected)

    def test_bpnet_dict_task_name(self):
        """Kipoi BPNet-OSKN dict keyed by TF name, via BPNetPredictor."""
        x = _onehot_seqs(n=300)
        expected = kipoi_bpnet_dict(x)['Nanog'].sum(axis=(1, 2))

        y = BPNetPredictor(
            pred_fun=kipoi_bpnet_dict,
            task_idx='Nanog',
            batch_size=64,
            reduce_fun=lambda p: p.sum(axis=(1, 2)),
        )(x)

        self.assertEqual(y.shape, (300, 1))
        np.testing.assert_allclose(y[:, 0], expected)

    def test_plain_numpy_callable(self):
        """pred_fun is just a function; no Keras/Kipoi required."""
        def oracle(x):
            return x.reshape(x.shape[0], -1).sum(axis=1, keepdims=True)

        x = _onehot_seqs(n=128)
        y = ScalarPredictor(pred_fun=oracle, task_idx=0, batch_size=32)(x)
        self.assertEqual(y.shape, (128, 1))
        np.testing.assert_allclose(y[:, 0], oracle(x)[:, 0])

    def test_mave_generate_y_shape(self):
        """InSilicoMAVE.generate should return y_mut of shape (num_sim, 1)."""
        rng = np.random.RandomState(1)
        x_ref = np.eye(4, dtype=np.float32)[rng.randint(0, 4, size=SEQ_LEN)]
        num_sim = 300

        pred_generator = ScalarPredictor(
            pred_fun=deepzebrafish_like,
            task_idx=TASK_IDX,
            batch_size=70,
        )
        mave = InSilicoMAVE(
            RandomMutagenesis(mut_rate=0.1, seed=0),
            pred_generator,
            SEQ_LEN,
        )
        x_mut, y_mut = mave.generate(x_ref, num_sim=num_sim)

        self.assertEqual(x_mut.shape, (num_sim, SEQ_LEN, 4))
        self.assertEqual(y_mut.shape, (num_sim, 1))
        expected = deepzebrafish_like(x_mut)[:, TASK_IDX]
        np.testing.assert_allclose(y_mut[:, 0], expected)


if __name__ == '__main__':
    unittest.main()
