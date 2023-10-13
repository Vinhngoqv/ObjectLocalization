import numpy as np
import cv2, time, threading, socket, json, math
import torch, ultralytics
from ultralytics import YOLO

class ObjectLocalization:
    def __init__(self):
        ### Depth camera configuration
        self.rgb_frame = None
        self.frame_width, self.frame_height = None, None

        ### Detection model configuration
        self.model = YOLO(r"ModelDetection\yolov8n.pt")
        self.object_labels = self.model.names
        self.detect_results = None
        self.detect_frame = None
        self.object_config = {
            "PERSON": (128, 255, 255)
        }

        ### Object position calculation
        self.dt_package = None
        self.object_notation = {
            "results": []
        }
        self.object_transfered_note = None
        self.object_dt_data = None

        ### Server configuration
        self.server_ip = '192.168.0.3'
        self.server_port = 8585
    
    def client_connect(self):
        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Bind the socket to the server IP and port
            server_socket.bind((self.server_ip, self.server_port))
        except:
            pass
        
        # Listen for incoming connections
        server_socket.listen(2)

        client_socket, client_address = server_socket.accept()
        print(f"Connected to: {client_address}")

        while True:
            try:
                while True:
                    json_str = json.dumps(self.object_notation)
                    client_socket.send(json_str.encode('utf-8'))

                    # Wait for a response from the client
                    response = client_socket.recv(4*1024).decode('utf-8')  # Assuming the response is "abc"

                    # Handle the response as needed
                    transfer_data = json.loads(response)
                    self.object_transfered_note = transfer_data
                    # for object_dt in transfer_data["results"]:
                    #     print(f"Received from client: {object_dt}")
            except Exception as e:
                time.sleep(0.5)
                print(f"Error: {e}")
                continue

    def calculate_position(self):
        ### Position calculation parameter
        x_c, y_c, beta = 350, 850, 35
        frame_range = [640,0]   # Pixel
        theta_range = [55,142]  # Degree
        bias_x_const, bias_y_const = -580, -100

        while True:
            try:
                crr_frame = np.copy(self.rgb_frame)

                if self.object_transfered_note != None and len(self.object_transfered_note["results"]) != 0:
                    ### Calculate and Visualize the results
                    for object_dt in self.object_transfered_note["results"]:
                        # Parameter
                        d = object_dt["depth"]
                        theta = int((object_dt["w"]-frame_range[0])*(theta_range[1]-theta_range[0])/(frame_range[1]-frame_range[0])+theta_range[0])
                        delta_x = math.sin((theta-beta)*math.pi/180)*d
                        x_o = int(x_c + delta_x)+bias_x_const
                        y_o = int(y_c + math.sqrt(d**2-delta_x**2))+bias_y_const  

                        crr_frame = cv2.circle(crr_frame, (object_dt["w"], object_dt["h"]), 3, (0,0,255), -1)
                        crr_frame = cv2.putText(crr_frame, str(object_dt["depth"]), (object_dt["w"]+2, object_dt["h"]-15), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.object_config[object_dt["class"]], 1)
                        crr_frame = cv2.putText(crr_frame, f"({x_o}, {y_o})", (object_dt["x1"], object_dt["y1"]-5), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.object_config[object_dt["class"]], 2)
                        crr_frame = cv2.rectangle(crr_frame, (object_dt["x1"], object_dt["y1"]), 
                                                        (object_dt["x2"], object_dt["y2"]), self.object_config[object_dt["class"]], 4)
                cv2.imshow("Position results", crr_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break 
            except Exception as e:
                print(f"Failed to calculate the position! (Error: {e})")
                time.sleep(0.5)
                continue

    def results_parasing(self):
        while True:
            self.dt_package = []
            try:
                for dt_results in self.detect_results:
                    box_data = dt_results.boxes.data.cpu().tolist()
                    for dt_data in box_data:
                        x1, y1, x2, y2, conf, obj_id = dt_data
                        obj_label = self.object_labels[obj_id].upper()
                        dt_frame = self.detect_frame
                        
                        if obj_label == "PERSON":
                            self.w_pixel = int(x1+(x2-x1)/2)
                            self.h_pixel = int(y1+(y2-y1)/2)-5

                            pixel_dict = {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "class": obj_label, 
                                "w": self.w_pixel,
                                "h": self.h_pixel,
                                "depth": 0
                            }
                            self.dt_package.append(pixel_dict)

                            ### Visualize the results
                            dt_frame = cv2.circle(dt_frame, (self.w_pixel, self.h_pixel), 5, (0,0,255), -1)
                            # dt_frame = cv2.putText(dt_frame, str(distance_value), (w_pixel+2, h_pixel-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.object_config[obj_label], 2)
                            # dt_frame = cv2.putText(dt_frame, obj_label, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.object_config[obj_label], 4)
                            # dt_frame = cv2.rectangle(dt_frame, (int(x1), int(y1)), (int(x2), int(y2)), self.object_config[obj_label], 5)
                            cv2.imshow("Detection results", dt_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                break 

                self.object_notation["results"] = self.dt_package
            except Exception as e:
                print("Failed to parasing the detection results!\n")
                time.sleep(0.1)
                continue
                    
    def object_detect(self):
        while True:
            try:
                if self.rgb_frame is not None:
                    current_frame = self.rgb_frame
                else:
                    time.sleep(0.5)
                    continue

                results = self.model.predict(source=current_frame, verbose=False, conf=0.4)

                self.detect_results = results
                self.detect_frame = current_frame
            except:
                print("Failed to detect objects!\n")
                time.sleep(0.5)
                continue

    def camera_reading(self, cam_url):
        while True:
            try:
                cap = cv2.VideoCapture(cam_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                while True:
                    try:
                        ret, crr_frame = cap.read()
                        if ret:
                            self.frame_height, self.frame_width = crr_frame.shape[:2]
                            self.rgb_frame = crr_frame

                            ### Grid line
                            # point = [int(self.frame_width/2), int(self.frame_height/2)]
                            # cv2.line(self.rgb_frame, (point[0],0), (point[0],self.frame_height), (0,255,0), 2)
                            
                            # cv2.line(self.rgb_frame, (point[0]+80,0), (point[0]+80,self.frame_height), (0,0,255), 2)
                            # cv2.line(self.rgb_frame, (point[0]+80*2,0), (point[0]+80*2,self.frame_height), (0,0,255), 2)
                            # cv2.line(self.rgb_frame, (point[0]+80*3,0), (point[0]+80*3,self.frame_height), (0,0,255), 2)

                            # cv2.line(self.rgb_frame, (point[0]-80,0), (point[0]-80,self.frame_height), (0,0,255), 2)
                            # cv2.line(self.rgb_frame, (point[0]-80*2,0), (point[0]-80*2,self.frame_height), (0,0,255), 2)
                            # cv2.line(self.rgb_frame, (point[0]-80*3,0), (point[0]-80*3,self.frame_height), (0,0,255), 2)

                            # cv2.imshow("Object Localization", self.rgb_frame)
                        
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     cv2.destroyAllWindows()
                            #     break 
                    except:
                        break
            except:
                time.sleep(0.5)
                continue
        
    def run(self):
        cam_url = "http://192.168.0.97:5000/image_feed"

        camera_thread = threading.Thread(target=self.camera_reading, args=(cam_url,))
        detect_thread = threading.Thread(target=self.object_detect)
        parasingresults_thread = threading.Thread(target=self.results_parasing)
        calculatePosition_thread = threading.Thread(target=self.calculate_position)
        client_thread = threading.Thread(target=self.client_connect)

        camera_thread.start()
        detect_thread.start()
        parasingresults_thread.start()
        calculatePosition_thread.start()
        client_thread.start()

OLMethod = ObjectLocalization()
OLMethod.run()