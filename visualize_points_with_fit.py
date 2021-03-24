from config import *
from util.color_table_for_class import color_table_for_class
from util.file_util import *
from data_process.point_transform import *
from data_process.bv_data_process import *
import cv2

import os
import glob

#LIDAR_IDs is defined in config.py
BV_RANGE_SETTINGS = GetBVRangeSettings(LIDAR_IDs)

if not os.path.exists(VIS_FOLDER):
    os.mkdir(VIS_FOLDER)

test_data_subfolders = glob.glob(os.path.join(POINTS_WITH_CLASS_FOLDER, "*"))
test_data_subfolders.sort()

#vars' name end with "_set" means it containes some items which are indexed by lidar_name
for subfolder in test_data_subfolders:
    print("processing %s" % subfolder)

    pointcloud_name_set_list, _, para_name_set = GetTestDataList(
        subfolder, LIDAR_IDs)

    parameters = ReadSelectedPara(para_name_set)

    output_path = os.path.join(VIS_FOLDER, subfolder.split("/")[-1])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for j, pc_name_set in enumerate(pointcloud_name_set_list):
        print("%d / %d" % (j + 1, len(pointcloud_name_set_list)))

        points_input_set = ReadSelectedPoints(pc_name_set)
        points_trans_set = ProjectPointsToWorld(points_input_set, parameters)
        points_merge = MergePoints(points_trans_set)
        points_solid, points_dash = AdjustIntensity_vis(points_merge, BV_COMMON_SETTINGS)
        points_line = np.concatenate((points_solid, points_dash), axis=0)
       
        #origin_right,origin_left = histogram_view(points_solid[:,1])
     
        left_fit, right_fit, left_lane_inds, right_lane_inds = find_line(points_line)

        '''
        vis_img = VisualizePointsClass(points_merge)
        cv2.imshow("1", vis_img)
        cv2.waitKey(0)
        cv2.destoryAllWindows()
        #SaveVisImg(vis_img, pc_name_set, output_path)
        
        imgpos1 = cv2.cvtColor(vis_img,cv2.COLOR_RGB2GRAY)
        print(imgpos1.shape)
        height = imgpos1.shape[0]        #将tuple中的元素取出，赋值给height，width，channels
        width = imgpos1.shape[1]
        
        #cv2.imshow("1", imgpos1)
        #cv2.waitKey(0)
        #cv2.destoryAllWindows()
        
        for row in range(height):    #遍历每一行
            for col in range(width): #遍历每一列   
                if imgpos1[row][col] == 181 or imgpos1[row][col] == 176 :
                    imgpos1[row][col] = 1
                else :
                    imgpos1[row][col] = 0
        
        
        left_fit, right_fit, left_lane_inds, right_lane_inds = find_line(imgpos1)
        print(left_fit)
        '''


      