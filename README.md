# Machine Learning Final Project - Group 11

**Course:** CAP 5610

## Team Members

| Name | Role |
|------|------|
| Armin Delmo | Member |
| Austin Robinson | Team Coordinator |
| Brady Napier | Member |
| Jonathan Dooley | Member |
| Thomas Kern | Member |

---

## Problem: Image Classification

Given an image, the model should correctly identify the class of the image.

---

## Dataset: Tiny ImageNet

A downscaled version of ImageNet used as a benchmark for image classification.

- 64x64 RGB images
- 100,000 training images / 10,000 testing images
- 12,288 features (3 color channels per pixel)
- 200 classes
- Possible preprocessing: PCA to reduce feature dimensionality

---

## Member Assignments

Each member is responsible for implementing and evaluating 2 models on Tiny ImageNet.

### Armin Delmo
| Model | About | Strengths | Weaknesses |
|-------|-------|-----------|------------|
| **Decision Tree** | Interprets a series of decisions made to output an expected outcome; typically built with greedy heuristics (ID3, C4.5) | Very interpretable; handles numerical/categorical data well | Struggles with noise and variance; easy to overfit |
| **LSTM** | Special type of RNN for long-term dependencies like stock markets and sequences | Preserves information over long spans of time via memory cells and gates | Relies on dependent inputs; slow training; computationally expensive |

### Austin Robinson *(Team Coordinator)*
| Model | About | Strengths | Weaknesses |
|-------|-------|-----------|------------|
| **Naive Bayes Classifier** | Uses Bayes Theorem to calculate probability of class given data, selects class with highest probability | Fast training | Assumes conditional feature independence (often untrue for images); no spatial awareness |
| **Multi-layer Perceptron (MLP)** | Uses multiple fully connected perceptron layers with non-linear activation functions to learn complex decision boundaries | Can learn non-linear decision boundaries; versatile | Slow training with large feature sets; no spatial awareness |

### Brady Napier
| Model | About | Strengths | Weaknesses |
|-------|-------|-----------|------------|
| **Linear SVM** | Finds the hyperplane which maximizes the margin between classes | Works well when features outnumber samples | Cannot capture spatial relations (images are flattened); requires feature engineering |
| **CNN** | Uses convolutional filters to learn spatial patterns (edges, textures, shapes) through convolution and pooling layers | Learns spatial and hierarchical visual features; designed for image classification | Requires significant compute; risk of overfitting (mitigated by augmentation/dropout) |

### Jonathan Dooley
| Model | About | Strengths | Weaknesses |
|-------|-------|-----------|------------|
| **Kernel SVM** | Finds the optimal hyperplane; uses the "kernel trick" to handle non-linearly separable data by mapping to a higher-dimensional space | Works well when features outnumber samples | Requires feature engineering; training scales poorly with dataset size; multiclass requires k(k-1)/2 models |
| **LLM (Decoder-only)** | Next-token predictor — fed a sequence of tokens, predicts the next most probable token | Scales well with dataset size; potential for transfer learning via fine-tuning | Requires image tokenization; high compute cost from self-attention; must learn translation/transformation invariance |

### Thomas Kern
| Model | About | Strengths | Weaknesses |
|-------|-------|-----------|------------|
| **Logistic Regression** | Learns a decision plane using sigmoid function | Outputs probabilities (not just 0/1); quick and efficient to train | Designed for binary classification; weights may explode with large feature counts |
| **Transformer (Encoder-only)** | Turns sequences of features into tokens and combines them with context | Uses spatial relationships between features; non-recurrent (faster training) | May require a very large number of tokens; each token might not appear enough times |

---

## Repository Structure

```
├── dataset/
│   └── loader.py               # Tiny ImageNet loading and configuration
│
├── decision_tree/
│   └── model.py                # Decision Tree (Armin)
├── lstm/
│   └── model.py                # LSTM (Armin)
│
├── naive_bayes/
│   └── model.py                # Naive Bayes Classifier (Austin)
├── mlp/
│   └── model.py                # Multi-layer Perceptron (Austin)
│
├── linear_svm/
│   └── model.py                # Linear SVM (Brady)
├── cnn/
│   └── model.py                # CNN (Brady)
│
├── kernel_svm/
│   └── model.py                # Kernel SVM (Jonathan)
├── llm/
│   └── model.py                # LLM Decoder-only (Jonathan)
│
├── logistic_regression/
│   └── model.py                # Logistic Regression (Thomas)
└── transformer/
    └── model.py                # Transformer Encoder-only (Thomas)
```

---

## Evaluation Metrics

**Standard:**
- Accuracy
- Precision
- Recall

**Validation:**
- K-fold Cross Validation — determines if data sampling is a fluke; averages results across runs

**Special:**
- Confusion Matrix — compares actual vs. predicted classes
- F1 Score — Dice coefficient, specialized for imagery

---

## Project Timeline

| Deliverable | Deadline |
|-------------|----------|
| 1st model completed by each member | April 10th |
| 2nd model completed by each member | April 17th |
| Final presentation | May 4th |
