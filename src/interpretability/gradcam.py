import numpy as np
import cv2


def preprocess_image(img_np, target_size):
    # img_np: np.array HxWxC in [0,255]
    img = cv2.resize(img_np, target_size)
    img = img.astype(np.float32) / 255.0
    return img


def gradcam_keras(
    model,
    img_np,
    target_size=(224, 224),
    last_conv_name=None,
    class_index=None,
    colormap=cv2.COLORMAP_JET,
):
    import tensorflow as tf
    from tensorflow.keras.models import Model

    img = preprocess_image(img_np, target_size)
    x = np.expand_dims(img, axis=0)

    # Find the last conv layer if not provided
    if last_conv_name is None:
        conv_layer_names = [
            layer.name
            for layer in model.layers
            if "conv" in layer.name or "relu" in layer.name
        ]
        if not conv_layer_names:
            raise ValueError("No conv layer found. Specify last_conv_name.")
        last_conv_name = conv_layer_names[-1]

    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = Model(
        inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        if class_index is None:
            class_index = np.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    # Average the gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8

    # Resize and overlay
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    overlay = cv2.addWeighted(
        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0
    )
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return {"class_index": int(class_index), "heatmap": heatmap, "overlay": overlay}
