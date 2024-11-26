import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

class DecisionTree:
    
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree on the given data.
        :param X: Input feature vectors, shape (n_samples, 1)
        :param y: Target labels, shape (n_samples,)
        """
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict labels for the input data.
        :param X: Input feature vectors, shape (n_samples, 1)
        :return: Predicted labels, shape (n_samples,)
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        :param X: Input feature vectors
        :param y: Target labels
        :param depth: Current depth of the tree
        :return: Node representation (dictionary or leaf value)
        """
        self.min_samples_split = 6  #5-10
        self.min_samples_leaf = 3 #1-5

        # Base cases
        if len(np.unique(y)) == 1:  # If all labels are the same
            return y[0]
        if depth >= self.max_depth or len(y) < self.min_samples_split:  # Stop if max depth or samples limit reached
            return self._majority_class(y)

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:  # If no split is possible
            return self._majority_class(y)

        # Recursively build left and right subtrees
        left_indices = X[:, 0] <= best_split
        right_indices = X[:, 0] > best_split

        if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
            # Stop if any child would violate the leaf size constraint
            return self._majority_class(y)

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the current node as a dictionary
        return {"split": best_split, "left": left_subtree, "right": right_subtree}
    

    def _find_best_split(self, X, y):
        """
        Find the best split based on Gini impurity.
        :param X: Input feature vectors
        :param y: Target labels
        :return: Best split value (or None if no valid split is found)
        """
        unique_values = np.unique(X[:, 0])
        best_split = None
        best_gini = float("inf")

        for value in unique_values:
            left_indices = X[:, 0] <= value
            right_indices = X[:, 0] > value

            gini = self._gini_impurity(y[left_indices], y[right_indices])
            if gini < best_gini:
                best_gini = gini
                best_split = value

        return best_split


    def _gini_impurity(self, left_labels, right_labels):
        """
        Calculate the Gini impurity for a split.
        :param left_labels: Labels for the left split
        :param right_labels: Labels for the right split
        :return: Gini impurity value
        """
        n = len(left_labels) + len(right_labels)
        if n == 0:
            return 0

        def gini(labels):
            if len(labels) == 0:
                return 0
            proportions = np.bincount(labels) / len(labels)
            return 1 - np.sum(proportions ** 2)

        gini_left = gini(left_labels) * len(left_labels) / n if len(left_labels) > 0 else 0
        gini_right = gini(right_labels) * len(right_labels) / n if len(right_labels) > 0 else 0
        return gini_left + gini_right

    def _majority_class(self, y):
        """
        Return the majority class label.
        :param y: Target labels
        :return: Majority class label
        """
        if len(y) == 0:
            return None
        return np.bincount(y).argmax()

    def _predict_single(self, x, tree):
        """
        Predict a single data point.
        :param x: Feature vector, shape (1,)
        :param tree: Current node in the tree
        :return: Predicted label
        """
        if isinstance(tree, dict):  # If the node is a split
            if x[0] <= tree["split"]:
                return self._predict_single(x, tree["left"])
            else:
                return self._predict_single(x, tree["right"])
        else:  # If the node is a leaf
            return tree

    def _calculate_depth(self, tree):
        """
        Recursively calculate the depth of the tree.
        :param tree: The current node of the tree (can be a dictionary or a leaf value).
        :return: Depth of the tree.
        """
        if not isinstance(tree, dict):
            return 0  # Leaf node has depth 0

        # Recursively calculate depth of left and right subtrees
        left_depth = self._calculate_depth(tree["left"])
        right_depth = self._calculate_depth(tree["right"])

        # Return the maximum depth of the subtrees + 1 (for the current node)
        return max(left_depth, right_depth) + 1

    def get_depth(self):
        """
        Public method to get the depth of the trained tree.
        :return: Depth of the tree.
        """
        if self.tree is None:
            return 0  # If the tree is not built yet
        return self._calculate_depth(self.tree)


    def save_model(self, file_path):
        # Extract parameters to save
        model_data = {
            "max_depth": self.max_depth,
            "tree": self.tree
        }
        # Save the model data using joblib
        joblib.dump(model_data, file_path)
    

    @staticmethod
    def load_model(file_path):
        # Load the model data
        model_data = joblib.load(file_path)
        # Reconstruct the DecisionTreeClassifier instance
        model = DecisionTree(max_depth=model_data["max_depth"])
        model.tree = model_data["tree"]
        return model




# Run the training code only if the script is executed directly
if __name__ == "__main__":

    # Load train data
    train_features = np.load('./data/features/train/train_features_pca.npy')
    train_labels = np.load('./data/features/train/train_labels.npy')

    # Generate a random permutation of indices
    shuffle_indices = np.random.permutation(len(train_labels))

    # Apply the permutation to both data and labels --> AKA Shuffle data before training models
    train_features_shuffled = train_features[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]

    # Train the 50-layer Decision Tree Model
    dt = DecisionTree(max_depth=50)  # max accuracy is 0.287 when max_depth=16
    dt.fit(train_features_shuffled, train_labels_shuffled)

    # Train the 100-layer Decision Tree Model
    dt_100_layer = DecisionTree(max_depth=100)  # max accuracy is 0.287 when max_depth=16
    dt_100_layer.fit(train_features_shuffled, train_labels_shuffled)

    # Train the 16-layer Decision Tree Model
    dt_16_layer = DecisionTree(max_depth=16)  # max accuracy is 0.287 when max_depth=16
    dt_16_layer.fit(train_features_shuffled, train_labels_shuffled)

    # Initialize the Scikit Decision Tree with Gini coefficient and maximum depth of 50
    scikit_dt = DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=42)  # max accuracy is 0.629 when max_depth=9
    # Train Scikit-Learn Decision Tree Model
    scikit_dt.fit(train_features_shuffled, train_labels_shuffled)

    # Save my DT models using the save_model method of the DT class definition
    dt.save_model("./models/decision_tree/DT.pkl")
    dt_100_layer.save_model("./models/decision_tree/DT_100_layer.pkl")
    dt_16_layer.save_model("./models/decision_tree/DT_16_layer.pkl")

    # Save Scikit DT with scikit-learn format using joblib
    joblib.dump(scikit_dt, "./models/decision_tree/scikit_DT.pkl")

    # Display DONE message
    print("\nDecision Tree Models trained and saved successfully. See ./models or run the evaluate_models.py script\n")

