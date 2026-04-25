import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# 1. Charger le modèle ONNX
onnx_model = onnx.load("./trained_model/model.onnx")

# 2. Convertir ONNX en modèle TensorFlow (format SavedModel)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")

# 3. Convertir SavedModel TensorFlow en TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
tflite_model = converter.convert()

# 4. Sauvegarder le modèle TFLite
with open("./trained_model/synthese_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion terminée : model.tflite créé.")
