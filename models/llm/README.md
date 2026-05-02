# LLM (Decoder-Only) — Tiny ImageNet Classifier

**Author:** Jonathan Dooley  
**Course:** CAP 5610

---

## How It Works

This model applies the decoder-only transformer architecture (GPT-style) to image classification on Tiny ImageNet.

### Image Tokenization

Each 64×64 RGB image is split into a grid of non-overlapping **8×8 patches**, producing a sequence of **64 tokens**. Each patch is flattened into a 192-dimensional vector (8 × 8 × 3 channels) and projected into the model's embedding dimension via a learned linear layer. Learned positional embeddings are added to each token to encode spatial position.

### Decoder-Only Architecture

The token sequence is passed through a stack of **decoder blocks**. Each block uses **causal (masked) self-attention**, meaning each patch token can only attend to patches that came before it in the sequence — exactly as a language model processes text left-to-right. This is what makes it a decoder-only architecture.

Each decoder block consists of:
1. **Pre-norm LayerNorm**
2. **Causal Multi-Head Self-Attention** — QKV fused into one linear, upper-triangular causal mask applied before softmax
3. **Pre-norm LayerNorm**
4. **Feed-Forward Network** — Linear → GELU → Linear (expansion factor 4×)

After all decoder blocks, a final LayerNorm is applied and the **last token's** hidden state is passed to a linear classification head over 200 classes. Using the last token for classification mirrors how decoder-only language models generate the next token after seeing the full context.

### Model Sizes

| Size  | Embed Dim | Layers | Heads | FFN Dim | Params |
|-------|-----------|--------|-------|---------|--------|
| small | 128       | 4      | 3     | 512     | ~3M    |
| base  | 256       | 6      | 3     | 1024    | ~12M   |

---

## Files

| File           | Description                                      |
|----------------|--------------------------------------------------|
| `model.py`     | Model definition and training loop               |
| `evaluate.py`  | K-fold cross-validation and final test metrics   |
| `checkpoints/` | Saved `.pth` checkpoints per epoch and best run  |

---

## Setup

```bash
cd models/llm
python -m venv venv
source venv/bin/activate
pip install torch torchvision datasets matplotlib scikit-learn pillow
```

---

## Training

```bash
source venv/bin/activate
python model.py
```

To train the small variant, edit the last line of `model.py`:
```python
train(size="small")
```

Checkpoints are saved every 5 epochs and whenever a new best validation accuracy is achieved, under `checkpoints/llm/`.

---

## Evaluation

```bash
source venv/bin/activate
python evaluate.py
```

This runs:
1. **5-fold cross-validation** on the training set using the best saved checkpoint
2. **Final test evaluation** on the validation split

### Metrics Reported

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Confusion Matrix (saved to `confusion_matrix.png`)
