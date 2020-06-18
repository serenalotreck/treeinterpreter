# -*- coding: utf-8 -*-
import numpy as np
import sklearn

from sklearn.ensemble.forest import ForestClassifier, ForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("treeinterpreter requires scikit-learn 0.17 or later")


def _get_tree_paths(tree, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if node_id == _tree.TREE_LEAF: # unclear what this is
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]

    return paths


def _predict_tree(model, X, it, joint_contribution=False):
    """
    For a given DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor, or ExtraTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.

    it is the iteration number: this determines whether explanatory statements
    are printed or not.
    """
    print(f'Interpreting tree {it+1}')

    # First, we will use sklearn's apply() function on our
    # test instances X. This function returns the leaf that each instance
    # ended up in after being run through its decision path.
    leaves = model.apply(X)

    # Now that we know what leaves each instance belongs to, we can
    # use the treeinterpreter _get_tree_paths to get the decision paths for
    # each instance.
    # _get_tree_paths() is a recursive function that starts with the root node
    # and returns every decision path in the tree
    paths = _get_tree_paths(model.tree_, 0, it)

    # Since _get_tree_paths returns the paths in order from root to leaf,
    # we have to reverse the paths.
    for path in paths:
        path.reverse()

    # Now that we've gotten all the paths, we associate each path
    # with its leaf node by making a dictionary, where keys are
    # the leaf nodes, and values are the paths that lead to those nodes.
    leaf_to_path = {}
    #map leaves to paths
    for path in paths:
        leaf_to_path[path[-1]] = path

    # Next, we'll get the constant prediction values at each node.
    # These are the means of all instances that arrive at each node.

    # get predictions and remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze(axis=1)
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])

    if isinstance(model, DecisionTreeRegressor):
        # Now we want to get the biases. Since the bias is the same for
        # every decision path in the same tree, we just give the same bias
        # (the mean of all instance labels at the root node) to every path.
        biases = np.full(X.shape[0], values[paths[0][0]]) # shape = number of samples
        line_shape = X.shape[1] # shape = number of features

    elif isinstance(model, DecisionTreeClassifier):
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
        line_shape = (X.shape[1], model.n_classes_)

    # Next, we get the prediction made by each leaf
    direct_prediction = values[leaves]


    # make into python lists, accessing values will be faster
    values_list = list(values)
    feature_index = list(model.tree_.feature)
    # get feature names, to use later
    feature_names = list(X.columns) ## TODO make sure this works

    contributions = []
    if joint_contribution:
        if it == 0:
            print('==> Since we\'ve chosen to look at join contributions, we\'ll '
            'look at each path, starting at the top. We get the contribution of '
            'this first feature, then go to the next node, where we calculate '
            'the contribution of the second feature. Instead of reporting these '
            'back separately, we add this new contribution to the previous '
            'one, and say that this is the joint contribution of these two '
            'features, so on and so forth down the path.')
        for row, leaf in enumerate(leaves):
            # get path, which is a list of node IDs
            path = leaf_to_path[leaf]
            # initialize empty set for the features along the path.
            # this is filled iteratively with all the features used to split
            # the nodes on the path. Since this is a set, if we reuse a feature
            # along a path, that doesn't show up as an additional joint
            # contribution - we just overwrite the previous value
            path_features = set()
            # add an empty dict to contributions
            contributions.append({})

            for i in range(len(path) - 1):
                # path[i] is a node ID, which is then an index value for
                # feature_index. feature_index[path[i]] is the ID of the feature
                # used to split the node path[i]
                path_features.add(feature_index[path[i]])

                # now, we calculate the contribution of this feature, by
                # subtracting the value at the next node from the value
                # at this node
                contrib = values_list[path[i+1]] - \
                         values_list[path[i]]

                #path_features.sort() # commented out in original code

                # all we do to get the joint contributions is add "down the line"
                # we add the contribution at this node to the sum of all the
                # previous contributions.
                # for example, if my path has three nodes plus one leaf (terminal
                # node), split on features A, B, and C respectively, all we're
                # doing is making a dict of the form:
                # {('A'):contrA,('A','B'):contrA+contrB,('A','B','C'):contrA+contrB+contrC}
                contributions[row][tuple(sorted(path_features))] = \
                    contributions[row].get(tuple(sorted(path_features)), 0) + contrib

        return direct_prediction, biases, contributions

    else:
        if it == 0:
            print('==> Since we\'ve chosen to look at independent contributions, '
            'we go down each decision path, '
            'calculate the difference between the current node and the next node, '
            'and assign that contribution to the feature that split the current '
            'node. For each decision path, we create an array with a length equal '
            'to the number of features, and fill in the entries for the features '
            'that split nodes along this given path.')
        unique_leaves = np.unique(leaves)
        unique_contributions = {} # making a dict to store the contribs of
        # each feature to each decision path

        for row, leaf in enumerate(unique_leaves):
            for path in paths:
                if leaf == path[-1]:
                    break
            # the variable path keeps its last value after the loop ends;
            # so the above for loop serves to match the  leaf with its path
            # however, still not sure why you would do it this way, esp when you
            # have the dictionary leaf_to_path and that's how it's done above

            contribs = np.zeros(line_shape) # initializes array w len num_features
            # go down the decision path, getting the contribution for each
            # feature that splits a node on the path. We make a contribs array
            # for every decision path
            for i in range(len(path) - 1): # for all nodes but terminal node
                contrib = values_list[path[i+1]] - \
                         values_list[path[i]] # contrib is the amt that the mean
                         # is changed between this node and the next
                contribs[feature_index[path[i]]] += contrib # assign this
                # contribution to the index in the contribs array that corresponds
                # to the feature this node was split on
            unique_contributions[leaf] = contribs # store the contributions for
            # this leaf in the dict

        for row, leaf in enumerate(leaves):
            contributions.append(unique_contributions[leaf])

        return direct_prediction, biases, np.array(contributions)


def _iterative_mean(iter, current_mean, x):
    """
    Iteratively calculates mean using
    http://www.heikohoffmann.de/htmlthesis/node134.html
    :param iter: non-negative integer, iteration
    :param current_mean: numpy array, current value of mean
    :param x: numpy array, new value to be added to mean
    :return: numpy array, updated mean
    """
    return current_mean + ((x - current_mean) / (iter + 1))


def _predict_forest(model, X, joint_contribution=False):
    """
    For a given RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, or ExtraTreesClassifier returns a triple of
    [prediction, bias and feature_contributions], such that prediction ≈ bias +
    feature_contributions.
    """

    if joint_contribution:
        biases = []
        contributions = []
        predictions = []

        for i, tree in enumerate(model.estimators_):
            print('doing the thing')
            pred, bias, contribution = _predict_tree(tree, X, i, joint_contribution=joint_contribution)

            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)


        total_contributions = []

        for i in range(len(X)):
            contr = {}
            for j, dct in enumerate(contributions):
                for k in set(dct[i]).union(set(contr.keys())):
                    contr[k] = (contr.get(k, 0)*j + dct[i].get(k,0) ) / (j+1)

            total_contributions.append(contr)

        for i, item in enumerate(contribution):
            total_contributions[i]
            sm = sum([v for v in contribution[i].values()])



        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
            total_contributions)
    else:
        mean_pred = None
        mean_bias = None
        mean_contribution = None

        print('==> For every tree, we call _predict_tree():')
        for i, tree in enumerate(model.estimators_):
            pred, bias, contribution = _predict_tree(tree, X, i)

            if i < 1: # first iteration
                mean_bias = bias
                mean_contribution = contribution
                mean_pred = pred
            else:
                mean_bias = _iterative_mean(i, mean_bias, bias)
                mean_contribution = _iterative_mean(i, mean_contribution, contribution)
                mean_pred = _iterative_mean(i, mean_pred, pred)

        print('=== Back in _predict_forest() ===')
        print('==> Now that we have the contributions for each instance in '
        'each tree, we average the results across the forest.')

        return mean_pred, mean_bias, mean_contribution


def predict(model, X, joint_contribution=False):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.
    Parameters
    ----------
    model : DecisionTreeRegressor, DecisionTreeClassifier,
        ExtraTreeRegressor, ExtraTreeClassifier,
        RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier
    Scikit-learn model on which the prediction should be decomposed.

    X : array-like, shape = (n_samples, n_features)
    Test samples.

    joint_contribution : boolean
    Specifies if contributions are given individually from each feature,
    or jointly over them

    Returns
    -------
    decomposed prediction : triple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, If joint_contribution is False then returns and  array of
        shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification, denoting
        contribution from each feature.
        If joint_contribution is True, then shape is array of size n_samples,
        where each array element is a dict from a tuple of feature indices to
        to a value denoting the contribution from that feature tuple.
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if (isinstance(model, DecisionTreeClassifier) or
        isinstance(model, DecisionTreeRegressor)):
        return _predict_tree(model, X, joint_contribution=joint_contribution)
    elif (isinstance(model, ForestClassifier) or
          isinstance(model, ForestRegressor)):
        return _predict_forest(model, X, joint_contribution=joint_contribution)
    else:
        raise ValueError("Wrong model type. Base learner needs to be a "
                         "DecisionTreeClassifier or DecisionTreeRegressor.")
