# Linear SVM model with PCA - Brady Napier

"""
Note: actual training was done through a .ipynb file in Google Colab. This .py file was created using the training .ipynb using AI so that everything would be in one place, easy to run, and structured.

Note: make sure all libraries and requirements have been properly installed and you are running in an environment where the libraries are accessible.

Usage Instructions: from the root dir (machine_learning_group_11) run: python -m models.linear_svm.model using the command line arguments appropriate
- To run with a saved model use: --mode load (ex: --mode load --model_path models/linear_svm/results/linear_svm_model.pkl)
    - To specify the number of PCA components use: --pca 250 (default is 200)
- To train a model from scratch use: --mode train
    - To specify the value of C use: --C 0.5 (default is 1.0)
    - To specify the number of PCA components use: --pca 250 (default is 200)
    - To specify the number of samples use: max_samples 20000 (default is None)
- To run grid search use: --mode grid
    - To specify the number of PCA components use: --pca 250 (default is 200)
"""

from dataset.loader import DataConfig, get_numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import joblib
import argparse


class LinearSVMModel:
    def __init__(self, C=1.0, max_iter=5000, max_samples=10000, pca_components=None):
        """
        Args:
            C: SVM regularization parameter
            max_iter: max iterations for SVM solver
            max_samples: max samples to load for testing/training
            pca_components: number of PCA components, None = no PCA
        """
        self.C = C
        self.max_iter = max_iter
        self.max_samples = max_samples
        self.pca_components = pca_components

        # Model will be initialized later after scaling/PCA
        self.model = None
        self.scaler = StandardScaler()  # scale features
        self.pca = PCA(n_components=self.pca_components) if self.pca_components else None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_data(self):
        """Load dataset using your shared data loader."""
        config = DataConfig(max_samples=self.max_samples, normalize=False)
        self.X_train, self.y_train, self.X_val, self.y_val = get_numpy(config)

        print("Data loaded:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_val:   {self.X_val.shape}")

    def preprocess(self):
        """Scale and optionally apply PCA to reduce dimensionality."""
        print("\nPreprocessing data...")

        # Scale
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)

        # PCA
        if self.pca:
            print(f"Applying PCA: reducing to {self.pca_components} components...")
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_val = self.pca.transform(self.X_val)
        print(f"Preprocessed shapes: X_train={self.X_train.shape}, X_val={self.X_val.shape}")

    def train(self):
        """Train the SVM model."""
        print("\nTraining Linear SVM...")
        # Initialize model with dual=False for faster convergence in high dimensions
        self.model = LinearSVC(C=self.C, max_iter=self.max_iter, dual=False)
        print(f"Initialized model: C = {self.C}, max_iter = {self.max_iter}")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
      """Evaluate on validation set with full metrics."""
      print("\nEvaluating on testing set...")

      y_pred = self.model.predict(self.X_val)

      acc = accuracy_score(self.y_val, y_pred)

      # For multi-class, use macro average
      precision = precision_score(self.y_val, y_pred, average='macro', zero_division=0)
      recall = recall_score(self.y_val, y_pred, average='macro', zero_division=0)
      f1 = f1_score(self.y_val, y_pred, average='macro', zero_division=0)

      print("\n=== Evaluation Results ===")
      print(f"Accuracy : {acc * 100:.2f}%")
      print(f"Precision: {precision * 100:.2f}%")
      print(f"Recall   : {recall * 100:.2f}%")
      print(f"F1 Score : {f1 * 100:.2f}%")

      return acc, precision, recall, f1

    def evaluate_train(self):
        """Evaluate on training set."""
        print("\nEvaluating on training set...")
        y_pred_train = self.model.predict(self.X_train)
        acc_train = accuracy_score(self.y_train, y_pred_train)
        print(f"Training Accuracy: {acc_train * 100:.2f}")
        return acc_train

    def run(self):
        """Full pipeline: load → preprocess → train → evaluate"""
        self.load_data()
        self.preprocess()

        start = time.time()
        self.train()
        total = time.time() - start
        print(f"Total time for training: {total:.2f} seconds")

        train_eval = self.evaluate_train()
        val_eval = self.evaluate()

        return train_eval, val_eval

    def save_model(self, path="linear_svm_model.pkl"):
      """Save the trained pipeline to disk."""
      print(f"\nSaving model to {path}...")

      joblib.dump({
          "model": self.model,
          "scaler": self.scaler,
          "pca": self.pca
      }, path)

      print("Model saved successfully!")

    def load_model(self, path="linear_svm_model.pkl"):
      """Load a trained pipeline from disk."""
      print(f"\nLoading model from {path}...")

      data = joblib.load(path)

      self.model = data["model"]
      self.scaler = data["scaler"]
      self.pca = data["pca"]

      print("Model loaded successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear SVM Pipeline")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "load", "grid"],
        help="Mode to run: train | load | grid"
    )

    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    parser.add_argument("--pca", type=int, default=200, help="Number of PCA components")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_path", type=str, default="linear_svm_model.pkl", help="Path to save/load model")

    args = parser.parse_args()

    # =========================
    # TRAIN MODE
    # =========================
    if args.mode == "train":
        model = LinearSVMModel(
            C=args.C,
            max_iter=5000,
            max_samples=args.max_samples,
            pca_components=args.pca
        )

        model.run()
        model.save_model(args.model_path)

    # =========================
    # LOAD MODE
    # =========================
    elif args.mode == "load":
        model = LinearSVMModel(pca_components=args.pca)

        model.load_data()
        model.load_model(args.model_path)

        # IMPORTANT: apply preprocessing (transform only)
        model.X_val = model.scaler.transform(model.X_val)
        if model.pca:
            model.X_val = model.pca.transform(model.X_val)

        model.evaluate()

    # =========================
    # GRID SEARCH MODE
    # =========================
    elif args.mode == "grid":
        C_values = [0.5, 1.0, 3.0, 5.0, 10.0]
        pca_values = [args.pca]

        results = []

        for C in C_values:
            for pca in pca_values:
                print(f"\n=== Running: C={C}, PCA={pca} ===")

                model = LinearSVMModel(
                    C=C,
                    max_iter=5000,
                    max_samples=20000,
                    pca_components=pca
                )

                train_acc, val_acc = model.run()
                results.append((C, pca, train_acc, val_acc))

        results.sort(key=lambda x: x[3], reverse=True)

        print("\nFinal Results:")
        for r in results:
            print(f"C={r[0]}, PCA={r[1]} | Train={r[2]:.4f}, Val={r[3]:.4f}")