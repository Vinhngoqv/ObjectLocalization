import cv2, io, time
from realsense_depth import *
from datetime import datetime
from PIL import Image

class Webcam():
    def __init__(self):
        self.vid = DepthCamera()
        self.mywidth = 640
        self.depth_frame = None

    def get_frame(self):
        while True:
            ret, self.depth_frame, img, acc_xyz, gyro_angle = self.vid.get_frame()
            if not ret:
                break

            height, width = img.shape[:2]
            point = (int(width/2),int(height/2))
            # cv2.line(img, (point[0]-8,point[1]), (point[0]+8,point[1]), (0,0,255), 2)
            # cv2.line(img, (point[0],point[1]-8), (point[0],point[1]+8), (0,0,255), 2)

            # distance = str(self.depth_frame[point[1],point[0]])
            # cv2.putText(img, distance+" mm", (point[0]+2,point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            ### Font configuration
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (10, height-20)
            fontScale = 0.8
            color = (255, 255, 255)
            thickness = 2

            img = cv2.putText(img, datetime.now().strftime("%a %Y-%m-%d %H:%M:%S"), org, font,
                              fontScale, color, thickness, cv2.LINE_AA)
            
            ### Compress the image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            wpercent = (self.mywidth/float(img_pil.size[0]))
            hsize = int((float(img_pil.size[1])*float(wpercent)))
            img_pil = img_pil.resize((self.mywidth, hsize), Image.Resampling.LANCZOS)
            img_bytes = io.BytesIO()    
            img_pil.save(img_bytes, format="JPEG")
            img_compressed = img_bytes.getvalue()

            yield img_compressed

    def get_depth_value(self):
        return self.depth_frame