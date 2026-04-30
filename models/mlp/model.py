# Multi-layer Perceptron (MLP) model - Austin Robinson
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


num_epochs = 50
batchSize = 256
numFolds = 5

def set_seed(seed=42):
    """Sets the seed for reproducible results."""
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set Python built-in random seed
    random.seed(seed)
    
    # 3. Set NumPy random seed
    np.random.seed(seed)
    
    # 4. Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    
    # 5. Configure PyTorch cuDNN backend for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")


set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")



def prepare_data(hf_split, num_samples=None):
    """
    Extracts images and labels, flattens the images, and normalizes them.
    """
    # Using a subset can save memory, as Tiny ImageNet has 100,000 training images.
    if num_samples:
        hf_split = hf_split.select(range(num_samples))
        
    X = []
    y = hf_split['label']
    
    print(f"Processing {len(hf_split)} images...")
    for img in hf_split['image']:
        img_array = np.array(img)
        
        # Tiny ImageNet usually has RGB images (64, 64, 3), but some might be grayscale (64, 64).
        # We ensure all images have 3 channels to maintain a consistent feature count.
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
            
        # Flatten the 3D array (64x64x3) into a 1D array of 12,288 features
        # and normalize pixel values to [0, 1] for numerical stability.
        X.append(img_array.flatten() / 255.0)
        
    return np.array(X), np.array(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc1 = nn.Linear(12288, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(p=0.5)
    
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(512,200)
      
    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)



print("Loading dataset...")
dataset = load_dataset("zh-plus/tiny-imagenet")

X, y = prepare_data(dataset['train'])



X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.long)


combinedData = TensorDataset(X,y)



kf = StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=42)


# index i gives the metric for the ith k-cross fold iteration
precisions = []
recalls = []
accuracies = []
f1Scores = []
losses = []


# arr[i][j] is metric in epoch j during fold i
# calculated to plot later
trainingLosses = np.zeros((numFolds, num_epochs))
trainingAccuracies = np.zeros((numFolds,num_epochs))

best_val_acc = 0.0
best_model_path = "best_mlp_model.pth"

for fold, (train_ids, val_ids) in enumerate(kf.split(X,y.numpy())):
    print(f'--- Fold {fold + 1} ---')

    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(
        combinedData, 
        batch_size=batchSize, 
        sampler=train_subsampler
    )

    val_loader = DataLoader(
        combinedData, 
        batch_size=batchSize, 
        sampler=val_subsampler
    )

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001,weight_decay=1e-4)




    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            
            classPred = torch.argmax(outputs, dim=1)
            correct_preds += (classPred == batch_y).sum().item()
            total_samples += batch_y.size(0)

        # Calculate true epoch averages
        epoch_loss = running_loss / total_samples
        epoch_acc = 100 * correct_preds / total_samples

        trainingLosses[fold][epoch] = epoch_loss
        trainingAccuracies[fold][epoch] = epoch_acc
        
        print(f"Epoch: {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")


    # Get metrics on validation data

    model.eval()

    # Used to track loss for final calculation
    running_loss_val = 0

    # Total Number of validation samples
    total_samples_val = 0

    # Total number of correct predictions
    correct_preds_val = 0

    # List of all predictions
    y_pred_total = []

    # List of all real labels
    y_real_total = []
    
    with torch.no_grad():

        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs,batch_y)

            running_loss_val += loss.item() * batch_X.size(0)

            classPred = torch.argmax(outputs,dim=1)

            correct_preds_val += (classPred == batch_y).sum().item()
            
            total_samples_val += batch_y.size(0)

            y_real_total.extend(batch_y.cpu().tolist())
            y_pred_total.extend(classPred.cpu().tolist())

        # Calculate fold metrics

        foldAcc = 100 * correct_preds_val / total_samples_val 
        foldLoss = running_loss_val / total_samples_val
        foldPrecision = precision_score(y_real_total,y_pred_total, average='macro', zero_division=np.nan)
        foldRecall = recall_score(y_real_total,y_pred_total,average="macro")
        foldF1 = f1_score(y_real_total,y_pred_total,average='macro')

        if foldAcc > best_val_acc:
            best_val_acc = foldAcc
            
            torch.save(model.state_dict(),best_model_path)
            print(f"*** New best model saved with Validation Accuracy: {best_val_acc:.2f}% ***")

        accuracies.append(foldAcc)
        precisions.append(foldPrecision)
        recalls.append(foldRecall)
        f1Scores.append(foldF1)
        losses.append(foldLoss)

print(f"Accuracies: {accuracies}")
print(f"Recalls: {recalls}")
print(f"Precisions: {precisions}")
print(f"F1 Scores: {f1Scores}")
print(f"Losses: {losses}")


print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average Precisions: {np.mean(precisions):.4f}")
print(f"Average F1 Scores: {np.mean(f1Scores):.4f}")
print(f"Average Loss: {np.mean(losses):.4f}")



# Plot Graphs


# Training Loss
plt.figure(figsize=(10,6))

epochNums = np.arange(num_epochs)

plt.plot(epochNums, trainingLosses[0],marker='o',label="Fold 1")
plt.plot(epochNums, trainingLosses[1],marker='s',label="Fold 2")
plt.plot(epochNums, trainingLosses[2],marker='^',label="Fold 3")
plt.plot(epochNums, trainingLosses[3],marker='p',label="Fold 4")
plt.plot(epochNums, trainingLosses[4],marker='*',label="Fold 5")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("kCrossFoldLosses.png")

# Training Accuracy
plt.figure(figsize=(10,6))


plt.plot(epochNums, trainingAccuracies[0],marker='o',label="Fold 1")
plt.plot(epochNums, trainingAccuracies[1],marker='s',label="Fold 2")
plt.plot(epochNums, trainingAccuracies[2],marker='^',label="Fold 3")
plt.plot(epochNums, trainingAccuracies[3],marker='p',label="Fold 4")
plt.plot(epochNums, trainingAccuracies[4],marker='*',label="Fold 5")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("kCrossFoldTraining.png")
    



print("\nLoading and preparing final test dataset...")

X_test, y_test = prepare_data(dataset['valid'])


X_test = torch.from_numpy(X_test).to(torch.float32)
y_test = torch.from_numpy(y_test).to(torch.long)

test_data = TensorDataset(X_test,y_test)
test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=False)


# Load best saved parameters
final_model = MLP().to(device)
final_model.load_state_dict(torch.load(best_model_path))
final_model.eval()

test_running_loss = 0.0
test_correct_preds = 0
test_total_samples = 0

y_real_test = []
y_pred_test = []

# Test
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        test_running_loss += loss.item() * batch_X.size(0)
        classPred = torch.argmax(outputs, dim=1)
        
        test_correct_preds += (classPred == batch_y).sum().item()
        test_total_samples += batch_y.size(0)
        
        y_real_test.extend(batch_y.cpu().tolist())
        y_pred_test.extend(classPred.cpu().tolist())


# Calculate metrics
test_acc = 100 * test_correct_preds / test_total_samples
test_loss = test_running_loss / test_total_samples
test_precision = precision_score(y_real_test, y_pred_test, average='macro', zero_division=np.nan)
test_recall = recall_score(y_real_test, y_pred_test, average="macro")
test_f1 = f1_score(y_real_test, y_pred_test, average='macro')

print(f"Final Test Accuracy:  {test_acc:.2f}%")
print(f"Final Test Loss:      {test_loss:.4f}")
print(f"Final Test Precision: {test_precision:.4f}")
print(f"Final Test Recall:    {test_recall:.4f}")
print(f"Final Test F1 Score:  {test_f1:.4f}")

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_real_test, y_pred_test)
# Removes class labels since with 200 classes this is impossible to read
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("ConfusionMatrix.png")
