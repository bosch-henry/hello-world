import rosbag
import argparse
import os
import numpy as np
import time
from datetime import datetime

def bag_pcd(bag_dir_path):

    print('Pre-processing Initialized')

    # Input folder
    bag_folder = os.listdir(bag_dir_path)
    print(bag_folder)
   

    # process the bag files in the Input folder
    for bag_file in bag_folder:

        t0 = time.time()

         # Output folder
        current_path = os.path.dirname(os.path.abspath(__file__))
        pcd_dir_path = os.path.join(current_path, bag_file[0:-4], "1")
        if os.path.exists(pcd_dir_path):
            print('Path exists')
        else:
            os.makedirs(pcd_dir_path)
            print('Path is created!')

        # judge the file type
        if bag_file[-4:] != '.bag':
            continue
        else:

            bag = rosbag.Bag(bag_file, "r")
            bag_data = bag.read_messages('/livox/lidar')
            for topic, msg, t in bag_data:
                # get UTC_time to handle every data-line
                UTC_time  = int(msg.timebase/1000000)
                # msg.point0.x 
                # set a new pcd_file if the folder is empty
                if not os.listdir(pcd_dir_path): 
                    pcd_file_path = os.path.join(pcd_dir_path, str(UTC_time)) + '.pcd'

                    handle = open(pcd_file_path, 'w')

                    for i in range(len(msg.points)):

                        x_value = msg.points[i].x
                        y_value = msg.points[i].y
                        z_value = msg.points[i].z
                        reflect_value = msg.points[i].reflectivity

                        #print(type(x_value))

                        point_str = '\n' + str(x_value ) \
                            + ' ' + str(y_value) \
                                + ' ' + str(z_value) \
                                    + ' ' + str(reflect_value )
                        handle.write(point_str)

                    handle.close()

                else:
                        pcd_list = os.listdir(pcd_dir_path)

                        # judge the relationship between the current data-line and the pcd-file 
                        for i_pcd in range(len(pcd_list)):
                            time_exist = np.uint64(int(pcd_list[i_pcd][0:-4])) #ms
                            if abs(UTC_time-time_exist)<50:
                                file_isexist = 1
                                # print(file_isexist)
                                break
                            else:
                                file_isexist = 0

                        # add the content to the existed pcd file
                        if file_isexist == 1:
                            # print(pcd_file_path)
                            pcd_file_path = os.path.join(pcd_dir_path, str(time_exist)) + '.pcd'

                            handle = open(pcd_file_path, 'a')

                            for i in range(len(msg.points)):

                                x_value = msg.points[i].x
                                y_value = msg.points[i].y
                                z_value = msg.points[i].z
                                reflect_value = msg.points[i].reflectivity

                        

                                point_str = '\n' + str(x_value ) \
                                    + ' ' + str(y_value) \
                                        + ' ' + str(z_value) \
                                            + ' ' + str(reflect_value )
                                handle.write(point_str)
                            
                            handle.close()

                        # establish a new pcd file 
                        elif file_isexist == 0:
                            pcd_file_path = os.path.join(pcd_dir_path, str(UTC_time)) + '.pcd'

                            handle = open(pcd_file_path, 'w')

                            for i in range(len(msg.points)):

                                x_value = msg.points[i].x
                                y_value = msg.points[i].y
                                z_value = msg.points[i].z
                                reflect_value = msg.points[i].reflectivity

                                point_str = '\n' + str(x_value ) \
                                    + ' ' + str(y_value) \
                                        + ' ' + str(z_value) \
                                            + ' ' + str(reflect_value )
                                handle.write(point_str)

                            handle.close()

        t1 = time.time()

        print('processing time per bag_file:', t1 - t0)
if __name__ == "__main__":

    # bag_dir_path = "/home/vav/rosbag2pcd/"

    # ret = os.path.exists(bag_dir_path)

    # print(ret)

    # bag_folder = os.listdir(bag_dir_path)

    # for bag_file in bag_folder:
    #     if bag_file[-4:] == '.bag':
    #         # print(bag_file)
    #         bag = rosbag.Bag(bag_file, "r")
    #         bag_data = bag.read_messages('/livox/lidar')
    #         for topic, msg, t in bag_data:
    #             print(len(msg.points))
    #             for i in range(len(msg.points)):
    #                 print(msg.points[i].x)



    t0 = time.time()

    bag_dir_path = os.path.dirname(os.path.abspath(__file__))

    ret  = bag_pcd(bag_dir_path)

    t1 = time.time()
    
    print('processing time:', t1 - t0)
    
