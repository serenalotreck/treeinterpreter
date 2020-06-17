from sklearn.datasets import load_iris
from sklearn import tree
from treeinterpreter import treeinterpreter
import numpy as np


def makeTree(tree_type):
    """
    Make a simple decision tree classifier with iris dataset.

    tree_type, str: 'classification' or 'regression'

    returns: the tree fit with the data
    """
    if tree_type == 'classification':
        X, y = load_iris(return_X_y=True)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X,y)

        return X, y, clf

    if tree_type == 'regression':
        X = [[0, 0], [2, 2]]
        y = [0.5, 2.5]
        reg = tree.DecisionTreeRegressor()
        reg = reg.fit(X, y)

        return X, y, reg


if __name__ == "__main__":
    classX, classy, classTree = makeTree('classification')
    regX, regy, regTree = makeTree('regression')
    regX = np.array(regX)
    print("this is classX: {}".format(classX))
    print("this is regX: {}".format(regX))

    #print("=============== testing for classification ===============")
    #treeinterpreter.predict(classTree,classX,joint_contribution=True)

    print("=============== testing for regression ===============")
    treeinterpreter._predict_tree(regTree,regX,0,joint_contribution=False)
