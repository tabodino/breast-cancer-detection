import numpy as np
import cv2


def preprocess_batch(imgs, target_size):
    batch = []
    for img_np in imgs:
        img = cv2.resize(img_np, target_size).astype(np.float32) / 255.0
        batch.append(img)
    return np.array(batch)


def shap_keras(model, img_np, background_imgs, target_size=(224, 224), nsamples=100):
    import shap

    x_bg = preprocess_batch(background_imgs, target_size)
    x = preprocess_batch([img_np], target_size)

    # Try GradientExplainer in priority
    explainer = shap.GradientExplainer(model, x_bg)
    shap_values = explainer.shap_values(x, nsamples=nsamples)

    # shap_values can be [values_class0, values_class1]
    return shap_values
