import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        self.frame = None

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # # Accelerometer available frequency: {63, 250}Hz
        config.enable_stream(rs.stream.accel)
        # # Gyroscope available frequency: {200, 400}Hz
        config.enable_stream(rs.stream.gyro)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        ### Get IMU data
        acc_data = frames[2].as_motion_frame().get_motion_data()
        gyro_data = frames[3].as_motion_frame().get_motion_data()
        acc_xyz = np.asanyarray([acc_data.x, acc_data.y, acc_data.z])
        gyro_angle = np.asanyarray([gyro_data.x, gyro_data.y, gyro_data.z])
        # print(acc_xyz, gyro_angle)
        
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image, acc_xyz, gyro_angle

    def release(self):
        self.pipeline.stop()