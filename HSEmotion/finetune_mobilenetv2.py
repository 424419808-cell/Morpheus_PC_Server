#!/usr/bin/env python3
"""
Fine-tune MobileNetV2 FER2013 model on user's collected face data.
Saves fine-tuned model → ONNX → RKNN, copies to board.

Usage:
  conda activate rknn
  python3 finetune_mobilenetv2.py

Data expected in:  ./training_data/<emotion>/*.jpg
  7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
"""

import os, sys, time, glob
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ─── Config ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
EMO_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-4

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_H5 = "/tmp/FER_MobileNetV2_best.h5"
FINETUNED_H5 = os.path.join(BASE_DIR, "mobilenetv2_emotion_finetuned.h5")
ONNX_PATH = os.path.join(BASE_DIR, "mobilenetv2_emotion_finetuned.onnx")
RKNN_PATH = os.path.join(BASE_DIR, "mobilenetv2_emotion_finetuned.rknn")
BOARD_PATH = "/home/elf/Morpheus/emotion_mobilenetv2.rknn"
BOARD_IP = "10.192.48.233"
BOARD_PORT = 2222

# ─── Load data ───────────────────────────────────────────────────
def load_data(data_dir, labels, img_size):
    images, targets = [], []
    for i, label in enumerate(labels):
        pattern = os.path.join(data_dir, label, "*.jpg")
        files = sorted(glob.glob(pattern))
        print(f"  {label}: {len(files)} images")
        for f in files:
            img = tf.keras.preprocessing.image.load_img(f, target_size=img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
            targets.append(i)
    images = np.array(images, dtype=np.float32)
    targets = np.array(targets, dtype=np.int32)
    return images, targets

print("Loading data...")
X, y = load_data(DATA_DIR, EMO_LABELS, IMG_SIZE)
print(f"Total: {len(X)} images, {len(EMO_LABELS)} classes")

# Shuffle
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Train/val split (80/20)
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Normalize to [-1, 1] (MobileNetV2 format)
X_train = X_train / 127.5 - 1.0
X_val = X_val / 127.5 - 1.0

# Class weights: give more weight to minority classes
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {dict(zip(EMO_LABELS, class_weights))}")

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# ─── Model ───────────────────────────────────────────────────────
print("\nLoading pre-trained model...")
base_model = tf.keras.models.load_model(PRETRAINED_H5, compile=False)

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Unlock the top 15 layers (classifier head + last few feature blocks)
# to adapt to the user's face while preserving pretrained features.
for layer in base_model.layers[-15:]:
    layer.trainable = True

print("Trainable layers:")
for layer in base_model.layers:
    if layer.trainable:
        print(f"  {layer.name} ({layer.output_shape})")

# Re-compile with new optimizer
base_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

base_model.summary()

# ─── Train ───────────────────────────────────────────────────────
print("\nFine-tuning...")

# Data augmentation (mild, to avoid overfitting)
datagen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(X_train)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    ),
]

history = base_model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
)

# ─── Evaluate ────────────────────────────────────────────────────
val_loss, val_acc = base_model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
print(f"   Validation loss: {val_loss:.4f}")

# Per-class accuracy
y_pred = np.argmax(base_model.predict(X_val, verbose=0), axis=1)
from sklearn.metrics import classification_report
print("\nPer-class report:")
print(classification_report(y_val, y_pred, target_names=EMO_LABELS, digits=3))

# ─── Save fine-tuned model ───────────────────────────────────────
base_model.save(FINETUNED_H5)
print(f"\nSaved: {FINETUNED_H5}")

# ─── Convert to ONNX ────────────────────────────────────────────
print("\nConverting to ONNX...")
import tf2onnx

spec = (tf.TensorSpec((1, 128, 128, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(
    base_model,
    input_signature=spec,
    opset=11,
    output_path=ONNX_PATH,
)
print(f"Saved: {ONNX_PATH}")

# ─── Convert to RKNN ────────────────────────────────────────────
print("\nConverting to RKNN...")
# First convert ONNX to NCHW layout
import onnx
from onnx import helper

model = onnx.load(ONNX_PATH)
graph = model.graph
input_name = graph.input[0].name

# Add NCHW→NHWC transpose (declare input as NCHW, transpose to NHWC for the rest)
transpose_node = helper.make_node(
    "Transpose", name="nchw2nhwc",
    inputs=[input_name], outputs=["input_nhwc"],
    perm=[0, 2, 3, 1],  # NCHW → NHWC
)
graph.node[0].input[0] = "input_nhwc"
graph.node.insert(0, transpose_node)

# Fix input shape to NCHW
input_proto = graph.input[0]
dim = input_proto.type.tensor_type.shape.dim
dim[0].dim_value = 1
dim[1].dim_value = 3
dim[2].dim_value = 128
dim[3].dim_value = 128

onnx_nchw_path = ONNX_PATH.replace(".onnx", "_nchw.onnx")
onnx.save(model, onnx_nchw_path)
print(f"Saved NCHW ONNX: {onnx_nchw_path}")

# RKNN conversion
from rknn.api import RKNN

rknn = RKNN(verbose=False)
rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
    target_platform="rk3588",
    quantized_dtype="asymmetric_quantized-8",
    quantized_algorithm="normal",
)
rknn.load_onnx(model=onnx_nchw_path)
rknn.build(do_quantization=False)
rknn.export_rknn(RKNN_PATH)
rknn.release()
print(f"Saved: {RKNN_PATH} ({os.path.getsize(RKNN_PATH)/1024/1024:.1f} MB)")

# ─── Copy to board ──────────────────────────────────────────────
print(f"\nCopying to board ({BOARD_IP})...")
ret = os.system(
    f"scp -P {BOARD_PORT} {RKNN_PATH} elf@{BOARD_IP}:{BOARD_PATH}"
)
if ret == 0:
    print(f"✅ Deployed to board: {BOARD_PATH}")
else:
    print(f"⚠️  scp failed (ret={ret}), copy manually: scp -P {BOARD_PORT} {RKNN_PATH} elf@{BOARD_IP}:{BOARD_PATH}")

print("\nDone! Run face_lock.py on the board to test.")
