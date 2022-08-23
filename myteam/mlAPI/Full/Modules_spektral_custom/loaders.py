import numpy as np
import tensorflow as tf

from spektral.data.utils import (
    batch_generator,
    collate_labels_disjoint,
    prepend_none,
    sp_matrices_to_sp_tensors,
)
from ML.Full.Modules_spektral_custom.data_utils import to_disjoint, to_tf_signature

version = tf.__version__.split(".")
major, minor = int(version[0]), int(version[1])
tf_loader_available = major >= 2 and minor >= 4


class Loader:
    """
    Parent class for data loaders. The role of a Loader is to iterate over a
    Dataset and yield batches of graphs to feed your Keras Models.

    This is achieved by having a generator object that produces lists of Graphs,
    which are then collated together and returned as Tensors.

    The core of a Loader is the `collate(batch)` method.
    This takes as input a list of `Graph` objects and returns a list of Tensors,
    np.arrays, or SparseTensors.

    For instance, if all graphs have the same number of nodes and size of the
    attributes, a simple collation function can be:

    ```python
    def collate(self, batch):
        x = np.array([g.x for g in batch])
        a = np.array([g.a for g in batch)]
        return x, a
    ```

    The `load()` method of a Loader returns an object that can be passed to a Keras
    model when using the `fit`, `predict` and `evaluate` functions.
    You can use it as follows:

    ```python
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch)
    ```

    The `steps_per_epoch` property represents the number of batches that are in
    an epoch, and is a required keyword when calling `fit`, `predict` or `evaluate`
    with a Loader.

    If you are using a custom training function, you can specify the input signature
    of your batches with the tf.TypeSpec system to avoid unnecessary re-tracings.
    The signature is computed automatically by calling `loader.tf_signature()`.

    For example, a simple training step can be written as:

    ```python
    @tf.function(input_signature=loader.tf_signature())  # Specify signature here
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ```

    We can then train our model in a loop as follows:

    ```python
    for batch in loader:
        train_step(*batch)
    ```

    **Arguments**

    - `dataset`: a `spektral.data.Dataset` object;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the dataset at the start of each epoch.
    """

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self):
        """
        Returns lists (batches) of `Graph` objects.
        """
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )

    def collate(self, batch):
        """
        Converts a list of graph objects to Tensors or np.arrays representing the batch.
        :param batch: a list of `Graph` objects.
        """
        raise NotImplementedError

    def load(self):
        """
        Returns an object that can be passed to a Keras model when using the `fit`,
        `predict` and `evaluate` functions.
        By default, returns the Loader itself, which is a generator.
        """
        return self

    def tf_signature(self):
        """
        Returns the signature of the collated batches using the tf.TypeSpec system.
        By default, the signature is that of the dataset (`dataset.signature`):

            - Adjacency matrix has shape `[n_nodes, n_nodes]`
            - Node features have shape `[n_nodes, n_node_features]`
            - Edge features have shape `[n_edges, n_node_features]`
            - Targets have shape `[..., n_labels]`
        """
        signature = self.dataset.signature
        return to_tf_signature(signature)

    def pack(self, batch):
        """
        Given a batch of graphs, groups their attributes into separate lists and packs
        them in a dictionary.

        For instance, if a batch has three graphs g1, g2 and g3 with node
        features (x1, x2, x3) and adjacency matrices (a1, a2, a3), this method
        will return a dictionary:

        ```python
        >>> {'a_list': [a1, a2, a3], 'x_list': [x1, x2, x3]}
        ```

        :param batch: a list of `Graph` objects.
        """
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.dataset.signature.keys()]
        return dict(zip(keys, output))

    @property
    def steps_per_epoch(self):
        """
        :return: the number of batches of size `self.batch_size` in the dataset (i.e.,
        how many batches are in an epoch).
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))



class DisjointLoader(Loader):
    """
    A Loader for [disjoint mode](https://graphneural.network/data-modes/#disjoint-mode).

    This loader represents a batch of graphs via their disjoint union.

    The loader automatically computes a batch index tensor, containing integer
    indices that map each node to its corresponding graph in the batch.

    The adjacency matrix os returned as a SparseTensor, regardless of the input.

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are stacked along an additional dimension.
    If `node_level=True`, then the labels are stacked vertically.

    **Note:** TensorFlow 2.4 or above is required to use this Loader's `load()`
    method in a Keras training loop.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[n_nodes, n_node_features]`;
    - `a`: adjacency matrices of shape `[n_nodes, n_nodes]`;
    - `e`: edge attributes of shape `[n_edges, n_edge_features]`;
    - `i`: batch index of shape `[n_nodes]`.

    `labels` have shape `[batch, n_labels]` if `node_level=False` or
    `[n_nodes, n_labels]` otherwise.

    """

    def __init__(
        self, dataset, node_level=False, batch_size=1, epochs=None, shuffle=True
    ):
        self.node_level = node_level
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def load(self):
        if not tf_loader_available:
            raise RuntimeError(
                "Calling DisjointLoader.load() requires " "TensorFlow 2.4 or greater."
            )
        #print("self :", self)
        #print("self.tf_signature() :", self.tf_signature())
        return tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.tf_signature()
        )

    def tf_signature(self):
        """
        Adjacency matrix has shape [n_nodes, n_nodes]
        Node features have shape [n_nodes, n_node_features]
        Edge features have shape [n_edges, n_edge_features]
        Targets have shape [..., n_labels]
        """
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])
        if "a" in signature:
            signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)

        return to_tf_signature(signature)
