from flask import Flask, request, jsonify
import requests
import io
import base64
import struct
import socket
from PIL import Image
import numpy as np

app = Flask(__name__)

# Adresse du serveur de machine learning
ML_SERVER_URL = 'http://127.0.0.1:5000/predict'

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Handwriting Recognition</title>
    <h1>Draw a digit</h1>
    <canvas id="canvas" width="200" height="200" style="border:1px solid #000000;"></canvas>
    <br>
    <button onclick="clearCanvas()">Clear</button>
    <button onclick="sendImage()">Submit</button>
    <p id="result"></p>
    <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var painting = false;

        canvas.addEventListener('mousedown', startPosition);
        canvas.addEventListener('mouseup', endPosition);
        canvas.addEventListener('mousemove', draw);

		clearCanvas();

        function startPosition(e) {
            painting = true;
            draw(e);
        }

        function endPosition() {
            painting = false;
            ctx.beginPath();
        }

        function draw(e) {
            if (!painting) return;
            ctx.lineWidth = 10;
            ctx.lineCap = 'round';
            ctx.strokeStyle = 'black';

            ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        }

        function clearCanvas() {
			ctx.fillStyle = "rgb(255,255,255)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
			document.getElementById('result').innerText = '';
        }

        function sendImage() {
            var dataURL = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: dataURL }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = 'Predicted digit: ' + data.response;
            });
        }
    </script>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(",")[1]
    
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).reshape(1, -1).tolist()[0]
    binout = b''
    for i in range(len(image)):
        pixel = 255 - image[i]
        binout += struct.pack('B', pixel)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 55565))
    sent = s.send(binout)
    if sent == 0:
        print("Error")
    class_output = struct.unpack("B", s.recv(1))
    print(class_output)
    return jsonify({'response': class_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
