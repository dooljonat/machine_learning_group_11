# Naive Bayes Classifier model - Austin Robinson
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

kFolds = 5
TOTAL_SAMPLES = 100000
BATCH_SIZE = 10000  
ALL_CLASSES = np.arange(200) # Tiny ImageNet has 200 distinct classes that are just numbers 0-199

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set as {seed}")

set_seed()

print("Loading dataset...")
dataset = load_dataset("zh-plus/tiny-imagenet")

# Isolate the exact subset we are working with
#train_subset = dataset['train'].select(range(TOTAL_SAMPLES))

train_data = dataset['train']

# Extract Only Labels
print("Extracting labels to compute K-Fold stratification...")
y_all = np.array(train_data['label'])

def process_batch(hf_split, indices):
    """
    Takes a specific list of indices, fetches those images, 
    flattens, and normalizes them.
    """
    batch = hf_split.select(indices)
    
    X = []
    y = np.array(batch['label'])
    
    for img in batch['image']:
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
            
        X.append(img_array.flatten() / 255.0)
        
    return np.array(X), y


# Init Cross Validation
kf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=42)

accuracies = []
recalls = []
precisions = []
f1Scores = []

best_val_acc = 0.0
best_model_path = "best_gnb_model.pkl"

# K-Fold Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(TOTAL_SAMPLES), y_all), 1):
    print(f'\n--- Fold {fold} ---')
    
    # Initialize a fresh model for each fold
    gnb = GaussianNB()


    # Train in batches
    for start in range(0, len(train_idx), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(train_idx))
        batch_indices = train_idx[start:end]
        
        # Load batch into memory
        X_train_batch, y_train_batch = process_batch(train_data, batch_indices.tolist())
        
        # Train Individual Batch
        gnb.partial_fit(X_train_batch, y_train_batch, classes=ALL_CLASSES)

    print("Evaluating validation set in batches...")
    y_true_fold = []
    y_pred_fold = []
    
    for start in range(0, len(val_idx), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(val_idx))
        batch_indices = val_idx[start:end]
        
        X_val_batch, y_val_batch = process_batch(train_data, batch_indices.tolist())
        
        batch_preds = gnb.predict(X_val_batch)
        
        y_true_fold.extend(y_val_batch)
        y_pred_fold.extend(batch_preds)

    # Calculate metrics for the entire fold
    foldAcc = accuracy_score(y_true_fold, y_pred_fold) * 100
    foldRecall = recall_score(y_true_fold, y_pred_fold, average='macro')
    foldPrecision = precision_score(y_true_fold, y_pred_fold, average='macro', zero_division=np.nan)
    foldF1 = f1_score(y_true_fold, y_pred_fold, average='macro')
    

    # If we found a new best model save parameters
    if foldAcc > best_val_acc:
        best_val_acc = foldAcc
        with open(best_model_path, 'wb') as f:
            pickle.dump(gnb, f)
        print(f"*** New best model saved with Validation Accuracy: {best_val_acc:.2f}% ***")

    accuracies.append(foldAcc)
    recalls.append(foldRecall)
    precisions.append(foldPrecision)
    f1Scores.append(foldF1)

    print(f"Fold {fold} Metrics -> Acc: {foldAcc:.2f}%, Recall: {foldRecall:.4f}, Precision: {foldPrecision:.4f}")


print("\n--- Cross-Validation Results ---")
print(f"Accuracies: {accuracies}")
print(f"Recalls: {recalls}")
print(f"Precisions: {precisions}")
print(f"F1 Scores: {f1Scores}")

print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}%")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average Precisions: {np.mean(precisions):.4f}")
print(f"Average F1 Score: {np.mean(f1Scores):.4f}")


# Final Test
print("\nLoading and preparing final test dataset...")


# Load test data
test_data = dataset['valid']
TEST_SAMPLES = len(test_data)

# Load the best weights saved during cross-validation
with open(best_model_path, 'rb') as f:
    final_model = pickle.load(f)

print("Evaluating final test set in batches...")
y_true_test = []
y_pred_test = []

for start in range(0, TEST_SAMPLES, BATCH_SIZE):
    end = min(start + BATCH_SIZE, TEST_SAMPLES)
    
    # Generate sequential indices for the test batch
    batch_indices = list(range(start, end))
    
    # Fetch, flatten, and normalize the test batch
    X_test_batch, y_test_batch = process_batch(test_data, batch_indices)
    
    # Predict using the loaded model
    batch_preds = final_model.predict(X_test_batch)
    
    # Store results for final metric calculation
    y_true_test.extend(y_test_batch)
    y_pred_test.extend(batch_preds)

# Calculate final metrics across all collected predictions
test_acc = accuracy_score(y_true_test, y_pred_test) * 100
test_precision = precision_score(y_true_test, y_pred_test, average='macro', zero_division=np.nan)
test_recall = recall_score(y_true_test, y_pred_test, average="macro")
test_f1 = f1_score(y_true_test, y_pred_test, average='macro')

print("\n--- Final Test Set Metrics ---")
print(f"Final Test Accuracy:  {test_acc:.2f}%")
print(f"Final Test Precision: {test_precision:.4f}")
print(f"Final Test Recall:    {test_recall:.4f}")
print(f"Final Test F1 Score:  {test_f1:.4f}")



# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_true_test, y_pred_test)
# Removes class labels since with 200 classes this is impossible to read
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("ConfusionMatrix.png")
