import io
 
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
 
from config import IMG_SIZE, CLASS_NAMES, finetune_model_path, transfer_model_path
from custom_layers import RandomHue
 
 
# Default model: No_augmentation + Finetune (Current best model)
MODEL_PATH = finetune_model_path("no_augmentation")
 
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"RandomHue": RandomHue},
)
print("Model is ready.")
 
app = Flask(__name__)
 
 
def preprocess_image(image_bytes):
    # Reading image bytes from HTTP request, then return numpy array
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    return img_array
 
 
@app.route("/health", methods=["GET"])
def health():
    # Checking server
    return jsonify({"status": "ok", "model": str(MODEL_PATH)})
 
 
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Missing image file. Send POST and form-data key 'image'."
        }), 400
 
    file = request.files["image"]
    image_bytes = file.read()
 
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {str(e)}"}), 400
 
    batch = np.expand_dims(img_array, axis=0)
    probs = model.predict(batch, verbose=0)[0]
 
    predicted_index = int(np.argmax(probs))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(probs))
 
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
 
    print(f"[predict] {predicted_label} ({confidence * 100:.2f}%)  |  {prob_dict}")
 
    return jsonify({
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
    })
 
 
if __name__ == "__main__":
    # using host = "0.0.0.0" allows another device in the same wifi call this server
    app.run(host="0.0.0.0", port=5000, debug=False)