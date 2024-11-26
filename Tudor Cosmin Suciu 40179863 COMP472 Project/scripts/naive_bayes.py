import numpy as np
from sklearn.naive_bayes import GaussianNB
import joblib



# Gaussian Naive Bayes implementation
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = X_cls.mean(axis=0)
            self.var[cls] = X_cls.var(axis=0) + 1e-6  # small value to avoid division by zero
            self.prior[cls] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        posteriors = []
        for x in X:
            posteriors.append(self._predict_single(x))
        return np.array(posteriors)

    def _predict_single(self, x):
        posteriors = []
        for cls in self.classes:
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[cls])) - 0.5 * np.sum(((x - self.mean[cls])**2) / self.var[cls])
            log_prior = np.log(self.prior[cls])
            posterior = log_likelihood + log_prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def save_model(self, file_path):
        # Extract parameters to save
        model_data = {
            "mean": self.mean,
            "var": self.var,
            "prior": self.prior,
            "classes": self.classes
        }
        # Save the dictionary to a file
        joblib.dump(model_data, file_path)
    

    @staticmethod
    def load_model(file_path):
        # Load model parameters
        model_data = joblib.load(file_path)
        # Reconstruct the GaussianNaiveBayes instance
        model = GaussianNaiveBayes()
        model.mean = model_data["mean"]
        model.var = model_data["var"]
        model.prior = model_data["prior"]
        model.classes = model_data["classes"]
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

    # Train Gaussian Naive Bayes Model
    gnb = GaussianNaiveBayes()
    gnb.fit(train_features_shuffled, train_labels_shuffled)

    # Train Scikit-Learn Gaussian Naive Bayes
    scikit_gnb = GaussianNB()
    scikit_gnb.fit(train_features_shuffled, train_labels_shuffled)

    # Save Scikit GNB with scikit-learn format using joblib
    joblib.dump(scikit_gnb, "./models/naive_bayes/scikit_GNB.pkl")

    # Save my GNB model using the save_model method of the GNB class definition
    gnb.save_model("./models/naive_bayes/GNB.pkl")

    # Display DONE message
    print("\nGaussian Naive Bayes Models trained and saved successfully. See ./models\n")

