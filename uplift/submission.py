import numpy as np
import pandas as pd
from typing import List, Tuple, Any


def calc_uplift(treatment, target):
    """
    Function takes in treatment flags and target variable values and calculates uplift.

            Parameters:
                    treatment: ndarray - treatment flags
                    target: ndarray - target variable values

            Returns:
                    uplift: float - difference between avg target value of treatment and avg target value of control
    """
    control_idx = np.where(treatment == 0)[0]
    treated_idx = np.where(treatment == 1)[0]
    uplift = np.mean(np.take(target, treated_idx)) - np.mean(np.take(target, control_idx))
    return uplift


def pos_threshold(node, feature):
    """
    Function takes a node and a feature (int corresponding to column) and outputs possible threshold values.

            Parameters:
                node: Node of Uplift Tree
                feature: int - column idx of feature data

            Returns:
                threshold_options: ndarray - possible threshold values to split data into left/right
    """
    column_values = node.data[:, feature]
    unique_values = np.unique(column_values)
    if len(unique_values) > 10:
        percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
    else:
        percentiles = np.percentile(unique_values, [10, 50, 90])
    threshold_options = np.unique(percentiles)
    return threshold_options


def make_split(node, feature, threshold):
    """
    Function takes a node, feature (int corresponding to column), and threshold value to split data into left
    (<= threshold) and right (> threshold) groups. Also, keeps track of and outputs treatment flags and target variable
    values in new split data.

            Parameters:
                    node: Node of Uplift Tree
                    feature: int - column idx of feature data
                    threshold: float - possible threshold value as found by function pos_threshold

            Returns:
                    data_left: ndarray - data points with feature <= threshold
                    data_right: ndarray - data points with feature > threshold
                    treatment_left: ndarray - treatment flags for data_left
                    treatment_right: ndarray - treatment flags for data_right
                    target_left: ndarray - target variable values for data_left
                    target_right: ndarray - target variable values for data_right
    """
    treatment = node.treatment
    target = node.target
    column_values = node.data[:, feature]

    # Find indices of treatment and control groups:
    left_idx = np.where(column_values <= threshold)[0]
    right_idx = np.where(column_values > threshold)[0]

    # Keep track of left/right data corresponding treatment flags and target variable variables after split.
    data_left = np.take(node.data, left_idx, axis=0)
    data_right = np.take(node.data, right_idx, axis=0)
    treatment_left = np.take(treatment, left_idx)
    treatment_right = np.take(treatment, right_idx)
    target_left = np.take(target, left_idx)
    target_right = np.take(target, right_idx)
    return data_left, data_right, treatment_left, treatment_right, target_left, target_right


def calc_ddp(treatment_left, treatment_right, target_left, target_right):
    """
    Function takes left and right treatment flags and target variable values to calculate DeltaDeltaP.

            Parameters:
                    treatment_left: ndarray - treatment flags for left data of split
                    treatment_right: ndarray - treatment flags for right data of split
                    target_left: ndarray - target variable values for left data of split
                    target_right: ndarray - target variable values for right data of split

            Returns:
                    difference: float - DeltaDeltaP value (difference between uplifts of right and left data groups)
    """
    uplift_left = calc_uplift(treatment_left, target_left)
    uplift_right = calc_uplift(treatment_right, target_right)
    difference = uplift_right - uplift_left
    return difference


class Node:
    """
    A class to represent a node of an Uplift Tree.

    ...

    Attributes
    ----------
    left: Node
        left child
    right: Node
        right child
    data: ndarray
        feature data of node
    depth: int
        depth of node in tree
    treatment: ndarray
        treatment flags of data points
    target: ndarray
        target variable values of data points
    max_ddp: float
        maximum DeltaDeltaP value over possible splits
    feature: float
        feature that yields max_ddp
    threshold: float
        threshold that yields max_ddp
    title: str
        type of node (e.g. root, leaf) in Uplift Tree
    ATE: float
        average treatment effect

    Methods
    -------
    print_node():
        Print nodes in form of tree following format of example_tree.txt.
    """

    def __init__(self, data, treatment, target):
        """
        Initializes all necessary Node attributes.

                Parameters:
                        data: ndarray - feature data
                        treatment: ndarray - treatment flags
                        target: ndarray - target variable values

                Returns:
                        None
        """
        self.left = None
        self.right = None
        self.data = data
        self.depth = 0                          # Keep track of node depth
        self.treatment = treatment              # Keep track of how treatment and target variable change with splits
        self.target = target
        self.max_ddp = 0.0
        self.feature = 0.0                      # Keep track of best feature and threshold for each split
        self.threshold = 0.0
        self.title = "Root"                     # Keep track node title purely for printing the tree

        control = np.where(self.treatment == 0)[0]      # locate indices of control and treatment group
        treated = np.where(self.treatment == 1)[0]

        self.ATE = np.mean(np.take(self.target, treated, axis=0)) - np.mean(np.take(self.target, control, axis=0))

    def print_node(self):
        """
        Print nodes in form of tree following format of example_tree.txt.

                Parameters:
                        N/A
                Returns:
                        None
        """
        indent = "\t" * self.depth
        n_items = "n_items: " + str(len(self.data))
        ate = "ATE: " + str(self.ATE)
        if self.feature is None:
            feat = "split_feat: None"
        else:
            feat = "split_feat: feat" + str(self.feature)
        split_threshold = "split_threshold: " + str(self.threshold)
        if self.feature is None:
            self.title += " <leaf>"
        print(indent + self.title + "\n" + indent + n_items + "\n" + indent + ate + "\n" + indent + feat + "\n" +
              indent + split_threshold + "\n")


class UpliftTree:
    """
    A class to represent an Uplift Tree.

    Attributes
    ----------
    root: Node
        root of tree
    max_depth: int
        maximum depth of tree
    min_samples_leaf: int
        minimum # data points in a (child) node for a split to be considered
    min_samples_leaf_treated: int
        minimum # data points in a (child) node's treatment group for a split to be considered
    min_samples_leaf_control: int
        minimum # data points in a (child) node's control group for a split to be considered

    Methods
    -------
    build(node):
        Recursively builds the Uplift Tree. Uses check of whether a node's best feature is None.
    """

    def __init__(self, max_depth, min_samples_leaf, min_samples_leaf_treated, min_samples_leaf_control, data,
                 treatment, target):
        """
        Initializes all necessary UpliftTree attributes.

                Parameters:
                        max_depth: int - maximum depth of tree
                        min_samples_leaf: int - minimum # data points in a (child) node for a split to be considered
                        min_samples_leaf_treated: int - minimum # data points in a (child) node's treatment group for a
                            split to be considered
                        min_samples_leaf_control: int - minimum # data points in a (child) node's control group for a
                            split to be considered
                        data: ndarray - feature data of root node
                        treatment: ndarray - treatment flags of root node
                        target: ndarray - target variable values of root node

                Returns:
                        None
        """
        self.root = Node(data, treatment, target)
        # self.depth = 0
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def build(self, node):
        """
        Recursively builds the Uplift Tree. Uses check of whether a node's best feature is None.

                Parameters:
                        node: Node of Uplift Tree

                Returns:
                        None
        """
        best_feature = None
        best_threshold = None
        # Check if reached max depth. Each node keeps track of its depth in the tree.
        if node.depth == self.max_depth:
            node.feature = None
            node.threshold = None
            node.print_node()
        else:
            # Check if node exists, i.e. if parent node had a valid split.
            if node is not None:
                for feature in range(len(node.data[0])):
                    threshold_options = pos_threshold(node, feature)
                    for threshold in threshold_options:
                        data_left, data_right, treatment_left, treatment_right, target_left, target_right = \
                            make_split(node, feature, threshold)
                        # Check to make sure split data fulfills minimum size requirements.
                        if len(data_left) < self.min_samples_leaf or len(data_right) < self.min_samples_leaf:
                            continue
                        if len(np.where(treatment_left == 0)[0]) < self.min_samples_leaf_control or \
                                len(np.where(treatment_right == 0)[0]) < self.min_samples_leaf_treated:
                            continue
                        # Calculate DeltaDeltaP value for each split.
                        ddp = calc_ddp(treatment_left, treatment_right, target_left, target_right)
                        # Select best split by keeping track of max DeltaDeltaP value. Also keep track of new
                        # left/right treatment flags and target variable values.
                        if ddp > node.max_ddp:
                            node.max_ddp = ddp
                            best_data_left = data_left
                            best_data_right = data_right
                            best_treatment_left = treatment_left
                            best_treatment_right = treatment_right
                            best_target_left = target_left
                            best_target_right = target_right
                            best_feature = feature                  # Keep track of best feature and threshold value.
                            best_threshold = threshold

                node.feature = best_feature                         # Update node's feature and threshold values.
                node.threshold = best_threshold

                node.print_node()                                   # Visualization
            # Check to make sure valid split exists
            if node.feature is not None:
                node.left = Node(best_data_left, best_treatment_left, best_target_left)
                node.right = Node(best_data_right, best_treatment_right, best_target_right)
                node.left.title = "Left"                            # Update node title
                node.right.title = "Right"
                node.left.depth = node.depth + 1                    # Update node depth
                node.right.depth = node.depth + 1
                self.build(node.left)
                self.build(node.right)


class UpliftTreeRegressor:
    """
    A class to represent Uplift Tree Regressor.

    Attributes
    ----------
    max_depth: int
        maximum depth of tree
    min_samples_leaf: int
        minimum # data points in a (child) node for a split to be considered
    min_samples_leaf_treated: int
        minimum # data points in a (child) node's treatment group for a split to be considered
    min_samples_leaf_control: int
        minimum # data points in a (child) node's control group for a split to be considered
    tree: UpliftTree
        Uplift Tree to build and perform the regression on

    Methods
    -------
    fit(data, treatment, target):
        Builds and fits Uplift Tree
    predict(X):
        Computes prediction values for a dataset
    """
    def __init__(self, max_depth=3, min_samples_leaf=1000, min_samples_leaf_treated=300, min_samples_leaf_control=300):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.tree = None

    def fit(self, data, treatment, target):
        """
        Builds and fits Uplift Tree.

                Parameters:
                        data: ndarray - feature data to fit Uplift Tree
                        treatment: ndarray - treatment flags to fit Uplift Tree
                        target: ndarray - target variable values to fit Uplift Tree

                Returns:
                        None
        """
        self.tree = UpliftTree(self.max_depth, self.min_samples_leaf, self.min_samples_leaf_treated,
                               self.min_samples_leaf_control, data, treatment, target)
        self.tree.build(self.tree.root)

    def predict(self, X):
        """
        Computes prediction values for a dataset.

                Parameters:
                        X: ndarray - feature data to be predicted on

                Returns:
                        Predictions: ndarray - predicted ATE values for each data point using fitted Uplift Tree
        """
        predictions = []
        for point in X:
            current = self.tree.root
            while True:
                # Proceed down tree by, at each node, comparing data point's feature value to threshold.
                if current.feature is not None:
                    if point[current.feature] <= current.threshold:
                        current = current.left
                    else:
                        current = current.right
                else:
                    break
            predictions.append(current.ATE)
        return np.array(predictions)


"""
# Test using provided example data.
ex_x = np.load('examples/example_X.npy')
ex_y = np.load('examples/example_y.npy')
ex_treat = np.load('examples/example_treatment.npy')
ex_preds = np.load('examples/example_preds.npy')
reg = UpliftTreeRegressor(3, 6000, 2500, 2500)
s = reg.fit(ex_x, ex_treat, ex_y)
preds = reg.predict(ex_x)
print(np.sum(preds) - np.sum(ex_preds))
"""