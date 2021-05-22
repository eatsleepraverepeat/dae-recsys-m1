import numpy as np
import tensorflow as tf
import more_itertools as mit
from scipy.sparse import csr_matrix
from typing import Optional, Generator


TENSOR_OUTPUT_TYPE = tf.float64


def to_tensor(inp: csr_matrix) -> tf.Tensor:
    return tf.convert_to_tensor(inp.todense(), dtype=TENSOR_OUTPUT_TYPE)


def batch_iterator(
    batch_size: int,
    data_input: csr_matrix,
    data_true: Optional[csr_matrix] = None
) -> Generator:
    """See https://www.tensorflow.org/guide/data?hl=en#consuming_python_generators"""

    training = data_true is None

    def generator():
        indices = np.arange(0, data_input.shape[0])
        indices = np.random.permutation(indices)

        for idx in mit.chunked(indices, batch_size):
            if training:
                yield data_input[idx].todense(), data_input[idx].todense()
            else:
                yield data_input[idx].todense(), data_true[idx].todense()

    return generator
