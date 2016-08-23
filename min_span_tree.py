import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._validation import validate_graph

"""
A modified version of the minimum spanning tree algorithm implemented in SciPy
"""


def min_spanning_tree_K(data,
                        col_indices,
                        indptr,
                        i_sort,
                        row_indices,
                        predecessors,
                        rank, k):
    # Work-horse routine for computing minimum spanning tree using
    #  Kruskal's algorithm.  By separating this code here, we get more
    #  efficient indexing.

    n_verts = predecessors.shape[0]
    n_data = i_sort.shape[0]

    # Arrange `row_indices` to contain the row index of each value in `data`.
    # Note that the array `col_indices` already contains the column index.
    for i in range(n_verts):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i

    # step through the edges from smallest to largest.
    #  V1 and V2 are connected vertices.
    n_edges_in_mst = 0
    i = 0
    copy = np.copy(i_sort)

    while i < n_data and n_edges_in_mst < n_verts - 1:
        # print(str(n_edges_in_mst)+' ',end="")
        if np.size(copy) > k:
            r = np.random.randint(0, k)
        else:
            r = np.random.randint(0, np.size(copy))

        j = copy[r]
        # print(str(i_sort[i]) + '  '+str(j))
        # j=i_sort[i]
        V1 = row_indices[j]
        V2 = col_indices[j]

        # progress upward to the head node of each subtree
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # Compress both paths.
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2

        # if the subtrees are different, then we connect them and keep the
        # edge.  Otherwise, we remove the edge: it duplicates one already
        # in the spanning tree.
        if R1 != R2:
            n_edges_in_mst += 1

            # Use approximate (because of path-compression) rank to try
            # to keep balanced trees.
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
            elif rank[R1] < rank[R2]:
                predecessors[R1] = R2
            else:
                predecessors[R2] = R1
                rank[R1] += 1
        else:
            data[j] = 0

        copy = np.delete(copy, r)

        i += 1

    # We may have stopped early if we found a full-sized MST so zero out the rest
    if i < n_data:
        for i in range(np.size(copy)):
            data[copy[i]] = 0
    """while i < n_data:
        j = i_sort[i]
        data[j] = 0
        i += 1"""


def minimum_spanning_tree_K(csgraph, k=1, overwrite=False):
    csgraph = validate_graph(csgraph, True, np.float64, dense_output=False,
                             copy_if_sparse=not overwrite)
    N = csgraph.shape[0]

    data = csgraph.data
    indices = csgraph.indices
    indptr = csgraph.indptr

    rank = np.zeros(N, dtype=np.int32)
    predecessors = np.arange(N, dtype=np.int32)

    i_sort = np.argsort(data).astype(np.int32)
    row_indices = np.zeros(len(data), dtype=np.int32)

    min_spanning_tree_K(data, indices, indptr, i_sort,
                        row_indices, predecessors, rank, k)

    sp_tree = csr_matrix((data, indices, indptr), (N, N))
    sp_tree.eliminate_zeros()

    return sp_tree


"""X = csr_matrix([[0, 8, 0, 3],
                [0, 0, 2, 5],
                [0, 0, 0, 6],
                [0, 0, 0, 0]])
for i in range(10000):
    Tcsr = minimum_spanning_traa(X)
    mst=Tcsr.toarray().astype(int)
    #print(mst)
    if np.count_nonzero(mst) !=3:
        print('fail')"""
"""
dfs_tree = depth_first_order(mst, directed=False, i_start=0)
tree = np.zeros(4, dtype=np.int)
tree[0] = -1
for p in range(1,4):
    tree[p] = dfs_tree[1][p]

print(tree)"""
