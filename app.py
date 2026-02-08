import os
import pickle
from flask import Flask, request, render_template_string, jsonify
from train import ImageCaptioner, greedy_search, load_captions

app = Flask(__name__)

CHECKPOINT_PATH = "caption_model.pt"
FEATURES_PATH = "flickr30k_features.pkl"


def load_model_and_data():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model not found: {CHECKPOINT_PATH}. Train with train.py first.")
    import torch
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    model = ImageCaptioner(len(vocab.word2idx)).to("cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    pairs = load_captions("data/captions.txt")
    pairs = [(img, cap) for img, cap in pairs if img in features]
    return model, vocab, features, pairs


model, vocab, features, pairs = load_model_and_data()


HTML = """
<!DOCTYPE html>
<html>
<head><title>Image Captioning</title></head>
<body>
  <h1>Image Captioning</h1>
  <form method="post" action="/caption">
    <label>Image (filename from dataset):</label>
    <input type="text" name="image_name" placeholder="e.g. 1000092795.jpg" size="30" />
    <button type="submit">Get caption</button>
  </form>
  {% if caption %}
  <p><b>Caption:</b> {{ caption }}</p>
  {% endif %}
  {% if error %}
  <p style="color:red;">{{ error }}</p>
  {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/caption", methods=["POST"])
def caption():
    import torch
    image_name = request.form.get("image_name", "").strip()
    if not image_name:
        return render_template_string(HTML, error="Enter an image filename.")
    if image_name not in features:
        return render_template_string(HTML, error=f"Image '{image_name}' not in features.")
    cap = greedy_search(model, torch.tensor(features[image_name]), vocab, device="cpu")
    return render_template_string(HTML, caption=cap)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
