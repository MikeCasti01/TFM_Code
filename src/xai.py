import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from pathlib import Path
import shap


def get_gradcam_model(
    model: keras.Model,
    last_conv_layer_name: str = "conv5_block3_out",
) -> keras.Model:
    """
    Builds the Grad-CAM model.

    Returns a model whose outputs are:
        1. Activation maps of the last convolutional layer.
        2. Final class probabilities.

    Compatible with nested Functional backbones (Keras 3).
    """

    # ----------------------------------------------------------
    # Locate the backbone
    # ----------------------------------------------------------
    base_model = None

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError(
            "No backbone model was found inside the classification model."
        )

    # ----------------------------------------------------------
    # Locate the desired convolutional layer
    # ----------------------------------------------------------
    try:
        conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' was not found "
            "inside the backbone."
        ) from exc

    # ----------------------------------------------------------
    # Build a model from the backbone ONLY
    # ----------------------------------------------------------
    conv_model = keras.Model(
        inputs=base_model.input,
        outputs=conv_layer.output,
    )

    # ----------------------------------------------------------
    # Return both models
    # ----------------------------------------------------------
    return conv_model, base_model


def get_feature_extractor(model: keras.Model) -> keras.Model:
    """Construye un sub-modelo que produce el vector de características comprimido.

    Retorna la representación vectorial del espacio de características buscando la primera capa
    de tipo GlobalAveragePooling2D, sin importar su nombre autogenerado.

    Args:
        model (keras.Model): Modelo compilado.

    Returns:
        keras.Model: Sub-modelo de Keras cuya salida es el vector de características.
    """
    pooling_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            pooling_layer = layer
            break

    if pooling_layer is None:
        raise ValueError("El modelo no contiene ninguna capa del tipo GlobalAveragePooling2D.")

    feature_extractor = keras.Model(
        inputs=model.inputs,
        outputs=pooling_layer.output,
    )
    return feature_extractor


def generate_gradcam_heatmap(
    conv_model,
    classifier_model,
    img_array,
    pred_index=None,
):
    """
    Generates a Grad-CAM heatmap.

    Parameters
    ----------
    conv_model
        Model that outputs the last convolutional feature maps.

    classifier_model
        Complete classification model.

    img_array
        Tensor of shape (1, H, W, 3)

    pred_index
        Optional class index.
    """

    with tf.GradientTape() as tape:

        conv_outputs = conv_model(img_array)

        tape.watch(conv_outputs)

        x = conv_outputs

        classifier_started = False

        for layer in classifier_model.layers:

            if isinstance(layer, keras.Model):
                classifier_started = True
                continue

            if classifier_started:
                x = layer(x)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    superimposed = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return img, superimposed
