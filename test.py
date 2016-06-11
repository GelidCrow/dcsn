import numpy as np

X =np.array([[0, 8, 0, 3],
                [0, 0, 2, 5],
                [0, 0, 0, 6],
                [0, 0, 0, 0]])
edges=np.zeros(shape=[6,3])
index=0
a=1
for i in range(4):
    for c in range(a,4):
        edges[index]=[X[i][c],i,c]
        index+=1
    a+=1

graph_edges=edges[edges[:,0].argsort(kind='heapsort')[::-1]]
print(graph_edges)

n_tree_edges=4-1
n_edges_in_mst=0 # Number of edges in minimum spanning tree

graph_edges_copy=np.copy(graph_edges)
k=2
"""final_edges=np.zeros(shape=[n_tree_edges,3])
while n_edges_in_mst!=n_tree_edges:
    random_edge=np.random.randint(0,k)
    candidate_edge=graph_edges_copy[random_edge,0]

    valid= True
    for i in range(n_edges_in_mst):
        pass"""


def p(a,b=None):
    return 1+1

print(p(3))

