from flask import Flask, render_template, Response
from webcam import Webcam

import threading, time, socket, json
from waitress import serve

app = Flask(__name__)

webcam = Webcam()

def server_connection():
    server_ip = '192.168.0.3'
    server_port = 8585

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    ### Connect to the server
    while True:
        try:
            client_socket.connect((server_ip, server_port))
            print("Connected to the server!")
            break
        except Exception as e:
            print("Waiting for the server socket open...")
            time.sleep(1)
            continue
    
    ### Transfer data through the socket
    while True:
        try:
            ### Recieve the signal
            signal = client_socket.recv(2*1024).decode('utf-8')
            transfer_data = json.loads(signal)

            ### Collect the depth data
            try:
                depth_frame = webcam.get_depth_value()

                try:
                    while True:
                        if depth_frame != None:
                            break
                        depth_frame = webcam.get_depth_value()
                        print("Wait for frames ...")
                        time.sleep(1)
                except:
                    pass
                
                for object_dt in transfer_data["results"]:
                    object_dt['depth'] = int(depth_frame[object_dt['h'], object_dt['w']])
            except:
                print("Failed to get depth data!")
                pass

            ### Respone the depth data to the server
            json_str = json.dumps(transfer_data)
            # print(json_str)
            client_socket.send(json_str.encode('utf-8'))
            
        except:
            client_socket.close()
            print("Close the client socket!")
            break

def read_from_webcam():
    while True:
        # Read image from class Webcam
        image = next(webcam.get_frame())

        # Return to web by yield command
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/image_feed")
def image_feed():
    return Response( read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame" )

if __name__=="__main__":
    threading.Thread(target=server_connection).start()
    serve(app, host="0.0.0.0", port=5000)