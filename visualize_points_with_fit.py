from config import *
from util.color_table_for_class import color_table_for_class
from network.sync_batchnorm.replicate import patch_replication_callback
from network.bv_parsing_net import *
from util.file_util import *
from data_process.point_transform import *
from data_process.bv_data_process import *
import cv2
import torch
#import pandas as pd
import time

import os
import glob

# from torch.multiprocessing import Pool, Process, set_start_method
import multiprocessing as mp


def Inference(model, bv_data):
    h, w = bv_data.shape[:2]
    input_data = np.zeros((1, 2, h, w)).astype("float32")

    input_data[0, 0, :, :] = bv_data[:, :, 0]
    input_data[0, 1, :, :] = bv_data[:, :, 1]

    input_data = torch.from_numpy(input_data).cuda()

    _, _, output_label_p_map = model(input_data)

    output_label_p_map = np.array(output_label_p_map.detach().cpu()).squeeze()

    output = np.argmax(output_label_p_map, 0)
    label_map = np.asarray(output).astype("uint8")

    return label_map

def points_class_chose(subfolder, Saving_intensity_img, Saving_fitting_img, model):

    print("processing %s" % subfolder)
    npz_path = os.path.join(subfolder,'*.npz')
    print(npz_path)
    if glob.glob(npz_path) != None:
        print('npz exists')
        pass
    else:
        pcd_name_set_list, num, para_name_set = GetTestDataList(subfolder, LIDAR_IDs)

        output_path = subfolder
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        vis_path = os.path.join(output_path, "vis")
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)

        BV_RANGE_SETTINGS = bv_settings
        coef_output = []
        index_name = []
        point_x = []
        point_lefty = []
        point_righty = []
        point_raw_leftx = []
        point_raw_lefty = []
        point_raw_rightx = []
        point_raw_righty = []

        for j, pc_name_set in enumerate(pcd_name_set_list):
            print("%d / %d" % (j+1, num))
            time1 = time.time()
            points_input_set = ReadPcdFile(pc_name_set)
            print(pc_name_set)
            points_trans_set = ProjectPointsToWorld_one(points_input_set)

            points_near_ground = SelectPointsNearGround(points_trans_set, BV_COMMON_SETTINGS)
            AdjustIntensity(points_near_ground, BV_COMMON_SETTINGS)
            time2 = time.time()

            bv_data = ProduceBVData(points_near_ground, BV_COMMON_SETTINGS, BV_RANGE_SETTINGS)
            bv_label_map = Inference(model, bv_data)
            time3 = time.time()

            points_class_set = GetPointsClassFromBV_one(points_trans_set, bv_label_map, BV_COMMON_SETTINGS, BV_RANGE_SETTINGS)
            points_with_class = np.concatenate([points_trans_set, points_class_set], axis=1)

            if Saving_intensity_img == True:
                AdjustIntensity(points_with_class, BV_COMMON_SETTINGS)
                vis_img = VisualizePointsClass(points_with_class)
                SaveVisImg(vis_img, pc_name_set, vis_path)
            else:
                pass
                #cv2.imshow("1", vis_img)
                #cv2.waitKey(0)
                #cv2.destoryAllWindows()

            points_solid, points_dash = AdjustIntensity_fit(points_with_class, BV_COMMON_SETTINGS)
            if points_dash.size != 0 and points_solid.size != 0:
                points_line = np.concatenate((points_solid, points_dash), axis=0)
            elif points_dash.size == 0 and points_solid.size != 0:
                points_line = points_solid
            elif points_dash.size != 0 and points_solid.size == 0:
                points_line = points_dash

            one_coef_name = pc_name_set
            pure_name = one_coef_name.split("/")[-1][:-4]
            line_paint_address = os.path.join(vis_path, pure_name + ".png")
            lefty_center, righty_center, x, leftx, lefty, rightx, righty = \
                find_line(points_line,line_paint_address,Saving_fitting_img)

            point_lefty.append(lefty_center)
            point_righty.append(righty_center)
            index_name.append(pure_name)
            time4 = time.time()
            print(time4-time1,time3-time2,time4-time3)

            point_raw_leftx.append(leftx)
            point_raw_lefty.append(lefty)
            point_raw_rightx.append(rightx)
            point_raw_righty.append(righty)

        index_name = np.vstack(index_name)
        point_lefty = np.vstack(point_lefty)
        point_righty = np.vstack(point_righty)

        npz_name = os.path.join(output_path, subfolder.split("/")[-1] + ".npz")
        np.savez(npz_name, point_x= x, point_lefty=point_lefty, point_righty=point_righty,
                time_index=index_name, point_raw_leftx = point_raw_leftx, point_raw_lefty = point_raw_lefty,
                point_raw_rightx = point_raw_rightx, point_raw_righty = point_raw_righty)

if __name__ == '__main__':

    model = BVParsingNet()
    model = torch.nn.DataParallel(model, device_ids=GPU_IDs)
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(MODEL_NAME)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_data_subfolders = glob.glob(os.path.join(TEST_DATA_FOLDER, "*"))
    test_data_subfolders.sort()
    for subfolder in test_data_subfolders:
        points_class_chose (subfolder, False, False, model)

    # try:
    #     # set_start_method('spawn')
    #     pool = mp.Pool(2)
    #     for subfolder in test_data_subfolders:
    #         pool.apply_async(points_class_chose, args=(subfolder,False, False, model))
    #     pool.close()
    #     pool.join()
    # except RuntimeError:
    #     pass