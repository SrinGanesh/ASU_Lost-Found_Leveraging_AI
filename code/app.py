from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

model = None
processor = None

@app.before_first_request
def load_model():
    global model, processor
    model_id = 'Salesforce/blip-image-captioning-base'
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)

@app.route('/caption', methods=['POST'])
def caption_image():
    global model, processor
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(BytesIO(file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
