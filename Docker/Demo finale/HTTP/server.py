from flask import Flask, request, jsonify, send_from_directory
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import struct
import socket

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('templates', 'client.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data['image']

    # Décoder l'image
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Redimenssionage & Applatissage
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).reshape(1, -1).tolist()[0]
    print(image)
    binout = b''
    for i in range(len(image)):
        pixel = 255 - image[i]
        binout += struct.pack('B', pixel)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("172.17.0.1", 55565))
    sent = s.send(binout)
    if sent == 0:
        return jsonify({'status': 'failure'})
    class_output = struct.unpack("B", s.recv(1))
    return jsonify({'status': 'success', 'prediction': class_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
