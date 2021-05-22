import logging
import os

import numpy as np
import tensorflow as tf
from typing import List, Optional, Callable, Tuple

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')

import more_itertools as mit
from functools import partial
from pathlib import Path
from scipy import sparse as sp

from src.model.dae import DAEGraph
from src.data.generators import batch_iterator, TENSOR_OUTPUT_TYPE
from src.metrics.metrics import recall_at_k
from src.data.preprocessing import load_and_parse_msd_dataset

from tensorflow.keras.callbacks import Callback, ModelCheckpoint


TRAIN_DATA_PATH = Path('data/splits/train-data.npz')

VALID_DATA_TEST_PATH = Path('data/splits/valid-data-test.npz')
VALID_DATA_TRUE_PATH = Path('data/splits/valid-data-true.npz')

TEST_DATA_TEST_PATH = Path('data/splits/test-data-test.npz')
TEST_DATA_TRUE_PATH = Path('data/splits/test-data-true.npz')

ITEM_MAPPINGS_PATH = Path('data/splits/item2id.json')
USER_MAPPINGS_PATH = Path('data/splits/user2id.json')


def ll_loss(y_true, y_pred):
    log_softmax_var = tf.nn.log_softmax(y_pred)
    return -tf.reduce_mean(tf.reduce_sum(log_softmax_var * y_true, axis=1))


class MetricsCallback(Callback):

    def __init__(
        self,
        batch_size: int,
        data: Tuple
    ):
        super(Callback, self).__init__()

        self.metrics = {
            # 'ndcg@100': partial(ndcg_at_k, k=100),
            'recall@100': partial(recall_at_k, k=100)
        }
        self.x, self.y = data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):

        tracker = {name: [] for name, _ in self.metrics.items()}
        indices = np.arange(0, self.x.shape[0])
        for idx in mit.chunked(indices, self.batch_size):
            logits = self.model.predict(self.x[idx].todense())

            for name, metric in self.metrics.items():
                value = metric(self.y[idx], logits)
                tracker[name].append(value)

        results = {name: round(np.mean(values).item(), 3) for name, values in tracker.items()}
        for name, result in results.items():
            print(f"{name}: {result}")


class DAE:
    """Manages partially regularized Denoising Autoencoder,
        as proposed in:

        Variational Autoencoders for Collaborative Filtering, Liang et al.
        https://arxiv.org/pdf/1802.05814.pdf
        """

    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dims: Optional[List[int]] = None,
        batch_size_train: int = 500,
        batch_size_valid: int = 2000,
        epochs: int = 32,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        activation: Optional[Callable] = None,
        checkpoint_dir: Optional[Path] = None,
        keep_prob: float = 0.5,
        top_results: int = 100
    ):

        self.encoder_dims = encoder_dims

        if not decoder_dims:
            self.decoder_dims = [*reversed(self.encoder_dims)]
        else:
            if not self.encoder_dims != [*reversed(self.decoder_dims)]:
                raise Exception("encoder/decoder dims mismatch")
            self.decoder_dims = encoder_dims

        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_valid

        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.activation = activation if activation else tf.nn.tanh
        self.epochs = epochs

        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else Path('data/checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.top_results = top_results
        self.keep_prob = keep_prob

        self._build_train_dataset()
        self._build_test_dataset()

        self._load_valid_data()
        self._load_test_data()

    def train(self):
        self._setup_model()
        metrics = MetricsCallback(
            batch_size=self.batch_size_valid,
            data=(self.valid_data_input, self.valid_data_true)
        )

        checkpoints = ModelCheckpoint(
            filepath=self.checkpoint_dir,
            verbose=True,
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch'
        )

        logging.info('Training started')
        records, _ = self.train_data_input.shape
        self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            shuffle=False,
            steps_per_epoch=(records // self.batch_size_train) + 1,
            callbacks=[metrics, checkpoints]
        )

    def test(self):
        y_pred = self.model.predict(self.test_dataset)
        metric_value = recall_at_k(self.test_data_input, y_pred)
        return round(metric_value.item(), 3)

    def _setup_model(self):
        self.model = DAEGraph(
            self.encoder_dims,
            self.decoder_dims,
            self.activation,
            self.keep_prob
        )

        self.model.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate=self.lr,
                beta_1=self.beta1,
                beta_2=self.beta2,
            ),
            loss=ll_loss
        )

    def _build_train_dataset(self):
        self.train_data_input = sp.load_npz(TRAIN_DATA_PATH)
        self.nitems = self.train_data_input.shape[-1]
        self.train_dataset = tf.data.Dataset.from_generator(
            batch_iterator(
                batch_size=self.batch_size_train,
                data_input=self.train_data_input
            ),
            output_types=(TENSOR_OUTPUT_TYPE, TENSOR_OUTPUT_TYPE),
            output_shapes=(
                (self.batch_size_train, self.nitems), (self.batch_size_train, self.nitems)
            )
        )

    def _build_valid_dataset(self):
        """Data is used explicitly in MetricsCallback"""
        pass

    def _build_test_dataset(self):
        self._load_test_data()
        self.test_dataset = tf.data.Dataset.from_generator(
            batch_iterator(
                batch_size=self.batch_size_test,
                data_input=self.test_data_input,
                data_true=self.test_data_true
            ),
            output_types=(TENSOR_OUTPUT_TYPE, TENSOR_OUTPUT_TYPE),
            output_shapes=(
                (self.batch_size_test, self.nitems), (self.batch_size_test, self.nitems)
            )
        )

    def _load_valid_data(self):
        self.valid_data_input = sp.load_npz(VALID_DATA_TEST_PATH)
        self.valid_data_true = sp.load_npz(VALID_DATA_TRUE_PATH)

    def _load_test_data(self):
        self.test_data_input = sp.load_npz(TEST_DATA_TEST_PATH)
        self.test_data_true = sp.load_npz(TEST_DATA_TRUE_PATH)


if __name__ == '__main__':
    NITEMS = sp.load_npz(TRAIN_DATA_PATH).shape[-1]

    if not os.listdir('data/splits'):
        load_and_parse_msd_dataset()

    dae = DAE(encoder_dims=[NITEMS, 200, 64], batch_size_train=500)
    dae.train()
    recall_test = dae.test()
    print(f"Recall@100 is: {recall_test}")
