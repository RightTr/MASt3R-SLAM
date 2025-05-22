import rosbag
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from pathlib import Path
import numpy as np
     
datasets_path = Path('/mnt/Data/NTU4DRadLM/raw')   
imgs_path = Path('/mnt/Data/NTU4DRadLM/imgs')
    
bridge = CvBridge()
bag_files = list(datasets_path.glob('*/*.bag')) 
print(f"Found {len(bag_files)} bag files.")
for bag_file in bag_files:
    bag_name = bag_file.stem
    rgb_path = imgs_path / bag_name / 'RGB'
    thermal_path = imgs_path / bag_name / 'thermal'
    rgb_path.mkdir(parents=True, exist_ok=True)
    thermal_path.mkdir(parents=True, exist_ok=True)

    with rosbag.Bag(str(bag_file), 'r') as bag: 
        i_rgb, i_thermal = 0, 0 
        topic_list = bag.get_type_and_topic_info().topics.keys()
        has_rgb = "/rgb_cam/image_raw/compressed" in topic_list
        has_thermal = "/thermal_cam/thermal_image/compressed" in topic_list
        for topic,msg,t in bag.read_messages():

            timestr = "%.6f" % t.to_sec()
            image_name = timestr + ".png"

            if topic == "/rgb_cam/image_raw/compressed" and i_rgb < 2000:
                if msg._type == 'sensor_msgs/CompressedImage':
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(str(rgb_path / image_name), cv_image)
                print(f'i_rgb:{i_rgb}, {image_name}')
                i_rgb += 1


            elif topic == "/thermal_cam/thermal_image/compressed" and i_thermal < 2000:
                if msg._type == 'sensor_msgs/CompressedImage':
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(str(thermal_path / image_name), cv_image)
                print(f'i_thermal:{i_thermal}, {image_name}')
                i_thermal += 1
            
            if (not has_thermal and i_rgb >= 2000) or (has_thermal and i_rgb >= 2000 and i_thermal >= 2000):
                print(f"✅ Finished bag {bag_file}, RGB: {i_rgb}, Thermal: {i_thermal}")
                break




    


