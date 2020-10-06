import flask
from flask import Flask
from flask import request, render_template
from io import BytesIO
from PIL import Image, ImageFile
from src.models import ModelTransfer
from torchvision import transforms
import torch
from src.labeller import *
ImageFile.LOAD_TRUNCATED_IMAGES = True


learn_dict = torch.load('model_transfer_0.83', map_location='cpu')
model = ModelTransfer(pretrained=False)
model.load_state_dict(learn_dict['model'])
model.eval()
dog_human_labeller = learn_dict['dog_human_labeller']
breed_labeller = learn_dict['breed_labeller']
dog_idx = dog_human_labeller.label_dict['dog']
human_idx = dog_human_labeller.label_dict['human']
dog_thresh = 0.9
human_thresh = 0.5


def get_tfms(stats):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(**stats)
    ])


def get_prediction(img):
    tfms = get_tfms(learn_dict['model_normalization_stats'])
    img_tensor = tfms(img)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None, ...]
    preds = torch.sigmoid(model(img_tensor))
    dog_score = preds[:, dog_idx]
    human_score = preds[:, human_idx]
    breed = torch.argmax(preds[:, 2:])
    breed_class = breed_labeller.label_lookup[breed.item()]
    is_dog = True if dog_score > dog_thresh else False
    is_human = True if human_score > human_thresh else False
    out1 = 'dog' if is_dog else None
    out1 = 'human' if is_human else out1
    return out1, breed_class


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    img = request.files['file'].read()
    img = Image.open(BytesIO(img)).convert('RGB')
    pred = get_prediction(img)
    response = {}
    response["response"] = {
        "category": pred[0],
        "breed": pred[1]
    }
    if pred is None:
        response["response"]["category"] = "UNK"
    return flask.jsonify(response)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    app.run()
