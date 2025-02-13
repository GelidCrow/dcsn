"""
Chow-Liu Trees

Chow, C. K. and Liu, C. N. (1968), Approximating discrete probability distributions with dependence trees, 
IEEE Transactions on Information Theory IT-14 (3): 462-467. 

"""

import random

import numba
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import minimum_spanning_tree

from logr import logr
from min_span_tree import minimum_spanning_tree_K
from utils import check_is_fitted


###############################################################################

@numba.njit
def cMI_numba(n_features,
              log_probs,
              log_j_probs,
              MI):
    for i in range(n_features):
        for j in range(i + 1, n_features):
            for v0 in range(2):
                for v1 in range(2):
                    MI[i, j] = MI[i, j] + np.exp(log_j_probs[i, j, v0, v1]) * (
                        log_j_probs[i, j, v0, v1] - log_probs[i, v0] - log_probs[j, v1])
                    MI[j, i] = MI[i, j]
    return MI


@numba.njit
def log_probs_numba(n_features,
                    scope,
                    n_samples,
                    alpha,
                    mpriors,
                    priors,
                    log_probs,
                    log_j_probs,
                    log_c_probs,
                    cond,
                    p):
    for i in range(n_features):
        id_i = scope[i]
        prob = (p[i] + alpha * mpriors[id_i, 1]) / (n_samples + alpha)
        log_probs[i, 0] = logr(1 - prob)
        log_probs[i, 1] = logr(prob)

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                id_i = scope[i]
                id_j = scope[j]
                log_j_probs[i, j, 1, 1] = cond[i, j]
                log_j_probs[i, j, 0, 1] = cond[j, j] - cond[i, j]
                log_j_probs[i, j, 1, 0] = cond[i, i] - cond[i, j]
                log_j_probs[i, j, 0, 0] = n_samples - log_j_probs[i, j, 1, 1] - log_j_probs[i, j, 0, 1] - log_j_probs[
                    i, j, 1, 0]

                log_j_probs[i, j, 1, 1] = logr(
                    (log_j_probs[i, j, 1, 1] + alpha * priors[id_i, id_j, 1, 1]) / (n_samples + alpha))
                log_j_probs[i, j, 0, 1] = logr(
                    (log_j_probs[i, j, 0, 1] + alpha * priors[id_i, id_j, 0, 1]) / (n_samples + alpha))
                log_j_probs[i, j, 1, 0] = logr(
                    (log_j_probs[i, j, 1, 0] + alpha * priors[id_i, id_j, 1, 0]) / (n_samples + alpha))
                log_j_probs[i, j, 0, 0] = logr(
                    (log_j_probs[i, j, 0, 0] + alpha * priors[id_i, id_j, 0, 0]) / (n_samples + alpha))

                log_c_probs[i, j, 1, 1] = log_j_probs[i, j, 1, 1] - log_probs[j, 1]
                log_c_probs[i, j, 0, 1] = log_j_probs[i, j, 0, 1] - log_probs[j, 1]
                log_c_probs[i, j, 1, 0] = log_j_probs[i, j, 1, 0] - log_probs[j, 0]
                log_c_probs[i, j, 0, 0] = log_j_probs[i, j, 0, 0] - log_probs[j, 0]

    return (log_probs, log_j_probs, log_c_probs)


@numba.njit
def compute_log_factors(tree,
                        n_features,
                        log_probs,
                        log_c_probs,
                        log_factors):
    for feature in range(0, n_features):
        if tree[feature] == -1:
            log_factors[feature, 0, 0] = log_probs[feature, 0]
            log_factors[feature, 0, 1] = log_probs[feature, 0]
            log_factors[feature, 1, 0] = log_probs[feature, 1]
            log_factors[feature, 1, 1] = log_probs[feature, 1]
        else:
            parent = int(tree[feature])
            for feature_val in range(2):
                for parent_val in range(2):
                    log_factors[feature, feature_val, parent_val] = log_c_probs[
                        feature, parent, feature_val, parent_val]

    return log_factors


###############################################################################


class Cltree:
    def __init__(self):
        self.num_trees = 1
        self.num_edges = 0
        self._forest = False

    def is_forest(self):
        return self._forest

    def fit(self, X,m_priors, j_priors, alpha=1.0, sample_weight=None, scope=None, and_leaves=False,
            noise=None):
        """Fit the model to the data.

        Parameters
        ----------
        X : ndarray, shape=(n, m)
        The data array.

        m_priors: 
        the marginal priors for each feature
        
        j_priors: 
        the joint priors for each couple of features

        alpha: float, default=1.0
        the constant for the smoothing

        sample_weight: ndarray, shape=(n,)
        The weight of each sample.

        scope: 
        unique identifiers for the features

        and_leaves: boolean, default=False


        """

        self.alpha = alpha
        self.and_leaves = and_leaves
        self.n_features = X.shape[1]

        if scope is None:
            self.scope = np.array([i for i in range(self.n_features)])
        else:
            self.scope = scope

        if sample_weight is None:
            self.n_samples = X.shape[0]
        else:
            self.n_samples = np.sum(sample_weight)

        (log_probs, log_j_probs, log_c_probs) = self.compute_log_probs(X, sample_weight, m_priors, j_priors)
        self.log_probs = log_probs
        self.log_j_probs = log_j_probs
        self.log_c_probs = log_c_probs

        self.MI = self.cMI(log_probs, log_j_probs)
        if noise is not None:
            self.MI = self.__AddNoise(noise)

        self.tree = None
        self._Minimum_SPTree_log_probs( log_probs, log_c_probs)
        self.num_edges = self.n_features - self.num_trees

    def _Minimum_SPTree_log_probs(self, log_probs, log_c_probs):
        """ the tree is represented as a sequence of parents"""
        mst = minimum_spanning_tree(-(self.MI))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)
        self.df_order = dfs_tree[0]
        self.tree = self.create_tree(dfs_tree)

        # computing the factored representation
        self.log_factors = np.zeros((self.n_features, 2, 2))
        self.log_factors = compute_log_factors(self.tree, self.n_features, log_probs, log_c_probs, self.log_factors)

    def create_tree(self, dfs_tree):
        tree = np.zeros(self.n_features, dtype=np.int)
        tree[0] = -1
        for p in range(1, self.n_features):
            tree[p] = dfs_tree[1][p]
        return tree

    def compute_log_probs(self, X, sample_weight, m_priors, j_priors):
        """ WRITEME """
        log_probs = np.zeros((self.n_features, 2))
        log_c_probs = np.zeros((self.n_features, self.n_features, 2, 2))
        log_j_probs = np.zeros((self.n_features, self.n_features, 2, 2))

        sparse_cooccurences = sparse.csr_matrix(X)

        if sample_weight is None:
            cooccurences_ = sparse_cooccurences.T.dot(sparse_cooccurences)
            cooccurences = np.array(cooccurences_.todense())
        else:
            weighted_X = np.einsum('ij,i->ij', X, sample_weight)
            cooccurences = sparse_cooccurences.T.dot(weighted_X)
        p = cooccurences.diagonal()

        return log_probs_numba(self.n_features,
                               self.scope,
                               self.n_samples,
                               self.alpha,
                               m_priors,
                               j_priors,
                               log_probs,
                               log_j_probs,
                               log_c_probs,
                               cooccurences,
                               p)

    def cMI(self, log_probs, log_j_probs):
        """ WRITEME """
        MI = np.zeros((self.n_features, self.n_features))
        return cMI_numba(self.n_features, log_probs, log_j_probs, MI)

    def score_samples_log_proba(self, X, sample_weight=None):
        """ WRITEME """
        check_is_fitted(self, "tree")

        Prob = X[:, 0] * 0.0
        for feature in range(0, self.n_features):
            parent = self.tree[feature]
            if parent <= -1:
                Prob = Prob + self.log_factors[feature, X[:, feature], 0]
            else:
                Prob = Prob + self.log_factors[feature, X[:, feature], X[:, parent]]

        if sample_weight is None:
            m = Prob.mean()
        else:
            Prob = sample_weight * Prob
            m = np.sum(Prob) / np.sum(sample_weight)
        return m

    def score_sample_log_proba(self, x):
        """ WRITEME """
        prob = 0.0
        for feature in range(0, self.n_features):
            parent = self.tree[feature]
            if parent <= -1:
                prob = prob + self.log_factors[feature, x[feature], 0]
            else:
                prob = prob + self.log_factors[feature, x[feature], x[parent]]
        return prob

    def score_samples_scope_log_proba(self, X, features, sample_weight=None):
        """
        In case of a forest, this procedure compute the ll of a single tree of the forest.
        The features parameter is the list of the features of the corresponding tree.
        """
        Prob = X[:, 0] * 0.0
        for feature in features:
            parent = self.tree[feature]
            if parent <= -1:
                Prob = Prob + self.log_factors[feature, X[:, feature], 0]
            else:
                Prob = Prob + self.log_factors[feature, X[:, feature], X[:, parent]]

        if sample_weight is None:
            m = Prob.mean()
        else:
            Prob = sample_weight * Prob
            m = np.sum(Prob) / np.sum(sample_weight)

        return m

    def score_sample_scope_log_proba(self, x, features):
        """ WRITEME """
        prob = 0.0
        for feature in features:
            parent = self.tree[feature]
            if parent == -1:
                prob = prob + self.log_factors[feature, x[feature], 0]
            else:
                prob = prob + self.log_factors[feature, x[feature], x[parent]]
        return prob

    def score_samples_log_proba_v(self, X, tree):
        """ WRITEME """
        prob = X[:, 0] * 0.0
        log_factors = np.zeros((self.n_features, 2, 2))
        log_factors = compute_log_factors(tree, self.n_features, self.log_probs, self.log_c_probs, log_factors)
        for feature in range(0, self.n_features):
            parent = tree[feature]
            if parent <= -1:
                prob = prob + log_factors[feature, X[:, feature], 0]
            else:
                prob = prob + log_factors[feature, X[:, feature], X[:, parent]]

        return prob.mean()

    def makeForest(self, vdata, forest_approach):
        self.current_best_validationll = self.score_samples_log_proba(vdata)
        if forest_approach[0] == 'grasp':
            self.__GRASP(forest_approach, vdata)

        elif forest_approach[0] == 'ii':
            self.__iterative_improvement(vdata)

        elif forest_approach[0] == 'rii':

            p = 0.7
            t = 10
            if len(forest_approach) > 1:
                p = float(forest_approach[1])
                if len(forest_approach) > 2:
                    t = int(forest_approach[2])
            self.__Randomised_Iterative_Improvement(vdata, probability=p, times=t)

        if self.num_trees > 1:
            self._forest = True
        self.num_edges = self.n_features - self.num_trees

    def __GRASP(self, forest_approach, vdata):

        times = 3
        k = 3  # Best k edges

        if len(forest_approach) > 1:
            times = int(forest_approach[1])
            if len(forest_approach) > 2:
                k = int(forest_approach[2])

        """GRASP"""
        t = 0
        while t < times:

            """CONSTRUCT"""
            initial_tree = None
            mst = minimum_spanning_tree_K(-(self.MI), k)  # Using modified version of kruskal algorithm

            dfs_tree = depth_first_order(mst, directed=False, i_start=0)
            initial_tree = self.create_tree(dfs_tree)
            """End Construct"""

            """ Local Search"""
            initial_valid_ll = self.score_samples_log_proba_v(vdata, initial_tree)
            initial_num_tree = 1
            improved = True
            while improved:
                improved = False
                best_ll = -np.inf
                best_edge = None
                valid_edges = np.where(initial_tree != -1)
                if np.size(valid_edges) > 0:
                    for i in np.nditer(valid_edges):
                        new = np.copy(initial_tree)
                        new[i] = -1
                        valid_ll = self.score_samples_log_proba_v(vdata, new)
                        if valid_ll > best_ll:
                            best_edge = i
                            best_ll = valid_ll
                    if best_ll > initial_valid_ll:
                        initial_valid_ll = best_ll
                        initial_num_tree += 1
                        initial_tree[best_edge] = -1
                        improved = True

            """End local search"""

            if initial_valid_ll > self.current_best_validationll:
                self.current_best_validationll = initial_valid_ll
                self.num_trees = initial_num_tree
                self.tree = initial_tree
                # Now i can compute the log factors
                self.log_factors = np.zeros((self.n_features, 2, 2))
                self.log_factors = compute_log_factors(self.tree, self.n_features, self.log_probs, self.log_c_probs,
                                                       self.log_factors)

            t += 1

    def __AddNoise(self, variance):

        new_MI = np.copy(self.MI) + variance * np.abs(np.random.randn(self.n_features, self.n_features))
        triangular_lower_ids = np.tril_indices(new_MI.shape[0])
        new_MI[triangular_lower_ids] = np.triu(new_MI).T[triangular_lower_ids]

        return new_MI

    def __iterative_improvement(self, vdata):

        improved = True
        while improved:
            improved = False
            best_ll = -np.inf
            best_edge = None
            valid_edges = np.where(self.tree != -1)
            if np.size(valid_edges) > 0:
                for i in np.nditer(valid_edges):
                    new = np.copy(self.tree)
                    new[i] = -1
                    valid_ll = self.score_samples_log_proba_v(vdata, new)
                    if valid_ll > best_ll:
                        best_edge = i
                        best_ll = valid_ll

                if best_ll > self.current_best_validationll:
                    self.current_best_validationll = best_ll
                    self.num_trees += 1
                    self.tree[best_edge] = -1

                    self.log_factors = np.zeros((self.n_features, 2, 2))
                    self.log_factors = compute_log_factors(self.tree, self.n_features, self.log_probs, self.log_c_probs,
                                                           self.log_factors)
                    improved = True

    def __Randomised_Iterative_Improvement(self, vdata, probability, times):
        t = 0
        valid_edges = np.where(self.tree != -1)
        while t < times and np.size(valid_edges) > 0:

            n_ll = -np.inf
            r = random.uniform(0, 1)
            edge_to_cut = None

            if r > probability:  # random cut
                r = random.randint(0, np.size(valid_edges) - 1)
                new = np.copy(self.tree)
                new[r] = -1
                edge_to_cut = r
                n_ll = self.score_samples_log_proba_v(vdata, new)
            else:  # best cut , if any
                for i in np.nditer(valid_edges):
                    n = np.copy(self.tree)
                    n[i] = -1
                    valid_ll = self.score_samples_log_proba_v(vdata, n)
                    if valid_ll > n_ll:
                        n_ll = valid_ll
                        edge_to_cut = i

            if n_ll > self.current_best_validationll:
                self.current_best_validationll = n_ll
                self.tree[edge_to_cut] = -1
                self.num_trees += 1
                self.log_factors = np.zeros((self.n_features, 2, 2))
                self.log_factors = compute_log_factors(self.tree, self.n_features, self.log_probs, self.log_c_probs,
                                                       self.log_factors)
            t += 1
            valid_edges = np.where(self.tree != -1)
