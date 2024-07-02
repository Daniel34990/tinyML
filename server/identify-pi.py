import requests
import socket

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

identification_server_ip = '137.194.194.16'
pi_name = 'pi1'  # Change this for each Pi

data = {
    'ip': get_ip_address(),
    'name': pi_name
}

response = requests.post(f'http://{identification_server_ip}:5000/identify', json=data)
if response.status_code == 200:
    identified_pis = response.json()
    print(f'Identified Pis: {identified_pis}')
else:
    print(f'Error: {response.json()}')
