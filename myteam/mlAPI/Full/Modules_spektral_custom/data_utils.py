import numpy as np
from scipy import sparse as sp

def to_disjoint(x_list=None, a_list=None, e_list=None, f_list=None):
    """
    Converts lists of node features, adjacency matrices and edge features to
    [disjoint mode](https://graphneural.network/spektral/data-modes/#disjoint-mode).
    Either the node features or the adjacency matrices must be provided as input.
    The i-th element of each list must be associated with the i-th graph.
    The method also computes the batch index to retrieve individual graphs
    from the disjoint union.
    Edge attributes can be represented as:
    - a dense array of shape `(n_nodes, n_nodes, n_edge_features)`;
    - a sparse edge list of shape `(n_edges, n_edge_features)`;
    and they will always be returned as a stacked edge list.
    :param x_list: a list of np.arrays of shape `(n_nodes, n_node_features)`
    -- note that `n_nodes` can change between graphs;
    :param a_list: a list of np.arrays or scipy.sparse matrices of shape
    `(n_nodes, n_nodes)`;
    :param e_list: a list of np.arrays of shape
    `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`;
    :return: only if the corresponding list is given as input:
        -  `x`: np.array of shape `(n_nodes, n_node_features)`;
        -  `a`: scipy.sparse matrix of shape `(n_nodes, n_nodes)`;
        -  `e`: np.array of shape `(n_edges, n_edge_features)`;
        -  `i`: np.array of shape `(n_nodes, )`;
    """
    if a_list is None and x_list is None:
        raise ValueError("Need at least x_list or a_list.")

    # Node features
    x_out = None
    if x_list is not None:
        x_out = np.vstack(x_list)

    # Adjacency matrix
    a_out = None
    if a_list is not None:
        a_out = sp.block_diag(a_list)

    # Batch index
    n_nodes = np.array([x.shape[0] for x in (x_list if x_list is not None else a_list)])
    i_out = np.repeat(np.arange(len(n_nodes)), n_nodes)

    # Edge attributes
    e_out = None
    if e_list is not None:
        if e_list[0].ndim == 3:  # Convert dense to sparse
            e_list = [e[sp.find(a)[:-1]] for e, a in zip(e_list, a_list)]
        e_out = np.vstack(e_list)

    # Graph features
    f_out = None
    if f_list is not None:
        f_out = np.vstack(f_list)

    return tuple(out for out in [x_out, a_out, e_out, i_out, f_out] if out is not None)


def to_tf_signature(signature):
    """
    Converts a Dataset signature to a TensorFlow signature.
    :param signature: a Dataset signature.
    :return: a TensorFlow signature.
    """
    output = []
    keys = ["x", "a", "e", "i", "f"]
    for k in keys:
        if k in signature:
            shape = signature[k]["shape"]
            dtype = signature[k]["dtype"]
            spec = signature[k]["spec"]
            output.append(spec(shape, dtype))
    output = tuple(output)
    if "y" in signature:
        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]
        output = (output, spec(shape, dtype))

    return output