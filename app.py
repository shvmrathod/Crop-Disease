import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.keras")

# Class labels (must match training order)
class_labels = [
    'Tomato_healthy',
    'Potato___Early_blight',
    'Tomato_Early_blight',
    'Tomato__Target_Spot',
    'Potato___Late_blight',
    'Tomato_Leaf_Mold',
    'Pepper__bell___Bacterial_spot',
    'Tomato_Late_blight',
    'Pepper__bell___healthy',
    'Potato___healthy'
]

def predict(image):
    # Preprocess
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    # Basic sanity check (reject weird images)
    if image.mean() < 0.2 or image.mean() > 0.9:
        return "Invalid image (lighting issue)"

    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)
    confidence = float(np.max(prediction))
    predicted_class = class_labels[np.argmax(prediction)]

    # Confidence check (strict)
    if confidence < 0.9:
        return f"Low confidence ({confidence:.2f}) - not a clear leaf image"

    return f"{predicted_class} ({confidence:.2f})"


# UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Crop Disease Classifier",
    description="Upload a plant leaf image to detect disease"
)

# Launch
interface.launch()