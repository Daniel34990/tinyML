from flask import Flask, request, jsonify

app = Flask(__name__)

identified_pis = {}

@app.route('/identify', methods=['POST'])
def identify():
    data = request.json
    pi_ip = data.get('ip')
    if pi_ip:
        identified_pis[pi_ip] = data.get('name', 'Unnamed')
        # Send the list of identified Raspberry Pis
        return jsonify(list(identified_pis.keys()))
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/list', methods=['GET'])
def list_pis():
    return jsonify(list(identified_pis.keys()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)