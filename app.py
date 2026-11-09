from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf, numpy as np, cv2, os, uuid
from PIL import Image
import matplotlib; matplotlib.use("Agg")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

model = tf.keras.models.load_model("model/malaria_mobilenetv2_final.keras")

last_conv_layer = None
for layer in reversed(model.layers):
    try:
        if len(layer.output_shape) == 4:
            last_conv_layer = layer.name
            break
    except AttributeError:
        continue

def preprocess(img_path):
    img = Image.open(img_path).resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def generate_gradcam(img_path, img_array):
    base_model = model.get_layer(last_conv_layer)
    idx = model.layers.index(base_model)
    classifier_layers = model.layers[idx + 1:]

    with tf.GradientTape() as tape:
        conv_out = base_model(img_array, training=False)
        tape.watch(conv_out)
        x = conv_out
        for layer in classifier_layers[:-1]:
            x = layer(x)
        final_layer = classifier_layers[-1]
        logit = tf.matmul(x, final_layer.kernel) + final_layer.bias  
        prob = tf.sigmoid(logit)
        class_channel = tf.where(prob[:, 0] > 0.5, logit[:, 0], -logit[:, 0])

    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = (tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    orig = cv2.resize(cv2.imread(img_path), (224, 224))
    hm = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, hm, 0.4, 0)
    name = str(uuid.uuid4()) + "_gradcam.jpg"
    cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], name), overlay)
    return name

def get_recommendation(label, confidence):
    if label == "Parasitized":
        if confidence >= 0.90:
            return "HIGH CONFIDENCE: Malaria parasite detected. Immediate clinical review recommended."
        elif confidence >= 0.70:
            return "MODERATE CONFIDENCE: Possible infection. Confirmatory testing (RDT/PCR) recommended."
        else:
            return "LOW CONFIDENCE: Inconclusive. Manual microscopy review strongly advised."
    else:
        if confidence >= 0.90:
            return "HIGH CONFIDENCE: No parasite detected. Monitor if symptoms persist."
        else:
            return "MODERATE CONFIDENCE: No parasite detected. Clinical judgement advised."


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("index"))
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    img_array = preprocess(filepath)
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Uninfected" if pred > 0.5 else "Parasitized"
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    gradcam_file = generate_gradcam(filepath, img_array)
    rec = get_recommendation(label, confidence)
    return render_template("result.html",
        label=label, confidence=f"{confidence:.1%}",
        original_img=filename, gradcam_img=gradcam_file,
        recommendation=rec)

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    app.run(debug=True)