# Logistic Regression model - Thomas Kern

from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from official.nlp.modeling.models import TransformerEncoder

import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

import matplotlib.pyplot as plt

def save_history_graphs(history_history):
    # Print out the graphs for each metric
    for metric_name in history_history.keys():
        print("Metric: ", metric_name)
        metric_data = history_history[metric_name]
        plt.plot(metric_data)
        plt.title(str(metric_name) + " by epoch")
        plt.xlabel("epoch")
        plt.ylabel(metric_name)
        plt.savefig(f"../images/{metric_name}_plot.png", dpi=300, bbox_inches="tight")
        plt.show()

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"../images/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

def save_score_report(y_true, y_pred):
    os.makedirs("report", exist_ok=True)
    with open("report/report.txt", "w") as out:

        # F1 score
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        micro_f1 = f1_score(y_true, y_pred, average="micro")

        print("Macro F1:", macro_f1, file=out)
        print("Weighted F1:", weighted_f1, file=out)
        print("Micro F1:", micro_f1, file=out)

        # Precision
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)

        print("Macro precision:", macro_precision, file=out)
        print("Weighted precision:", weighted_precision, file=out)
        print("Micro precision:", micro_precision, file=out)

        # Recall
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)

        print("Macro recall:", macro_recall, file=out)
        print("Weighted recall:", weighted_recall, file=out)
        print("Micro recall:", micro_recall, file=out)

        # Full report
        print(classification_report(y_true, y_pred), file=out)

def save_full_report(y_true, y_pred, history_history):
    save_score_report(y_true, y_pred)
    save_history_graphs(history_history)
    save_confusion_matrix(y_true, y_pred)

def build_logistic_model(num_classes=200):
    # Produce a model to distinguish each class.
    # This is a logistic regression model with 200 output classes.
    return tf.keras.Sequential([
        # Input shape
        tf.keras.Input(shape=(64, 64, 3)),
        tf.keras.layers.Flatten(),
        # Softmax replaces sigmoid, since it has the same effect anyways.
        tf.keras.layers.Dense(units=num_classes, activation="softmax"),
    ])


def compile_train_test(model, epochs, train_ds, val_ds):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    our_history = model.fit(train_ds, epochs=epochs, steps_per_epoch=781, validation_data=val_ds, validation_steps=78)
    # Get predictions on validation set
    steps = 40
    y_prob_chunks = []
    y_true_chunks = []

    for i, (x, y) in enumerate(val_ds):
        if i >= steps:
            break
        y_prob_chunks.append(model(x, training=False))
        y_true_chunks.append(y)

    y_prob = tf.concat(y_prob_chunks, axis=0)
    y_true = tf.concat(y_true_chunks, axis=0)

    y_pred = tf.argmax(y_prob, axis=1).numpy()
    y_true = y_true.numpy()
    save_full_report(y_true, y_pred, our_history.history)
    return y_true, y_pred, our_history.history

def get_datasets_v4(
    batch_size=128,
    image_size=(64, 64), # Fix this later
    name="zh-plus/tiny-imagenet",
):
    ds = load_dataset(name, streaming=True)
    hf_train = ds["train"]
    hf_val = ds["valid"]
    def gen(split):
        def ingen():
            for sample in split:
                img = sample["image"]
                lab = sample["label"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size != image_size:
                    img = img.resize(image_size)
                yield img, lab
        return ingen
    output_signature = (
        tf.TensorSpec(shape=(64, 64, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    train = tf.data.Dataset.from_generator(
        gen(hf_train),
        output_signature=output_signature
    )
    val = tf.data.Dataset.from_generator(
        gen(hf_val),
        output_signature=output_signature
    )
    train = train.shuffle(128 * batch_size).batch(batch_size)
    val = val.shuffle(128 * batch_size).batch(batch_size)
    train = train.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).repeat()
    val = val.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    return train, val

    
TINY_IMAGENET_CLASSES = 200

epochs = 40

train_ds, val_ds = get_datasets_v4()
model = build_logistic_model(TINY_IMAGENET_CLASSES)

compile_train_test(model, epochs, train_ds, val_ds)