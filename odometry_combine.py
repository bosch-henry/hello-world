import numpy as np
from asammdf import MDF
import pandas as pd
import os
import shutil
import time
import numpy as np
import glob
from kpi_calculate_list import *
import matplotlib.pyplot as plt
import math
#from data_process.point_transform import *

def Timezone_problem_gps(week, msecond):

    time = 7*24*60*60*1000*week + msecond + 315964800*1000

    return time

def timestamps_UTC(GPS_timestamps, UTC_GPS_list, leftline_timestamps):

    time_gap = GPS_timestamps[1] - GPS_timestamps[0]
    UTC_gap = (UTC_GPS_list[-1] - UTC_GPS_list[0])/len(UTC_GPS_list)
    ratio = UTC_gap/time_gap

    leftline_UTC = []

    for i in range(len(leftline_timestamps)):

        leftline_UTC.append(UTC_GPS_list[0] + ratio * (leftline_timestamps[0] - GPS_timestamps[0]) + ratio * (leftline_timestamps[i] - leftline_timestamps[0]))

    return np.array(leftline_UTC)

def index_nearst_relation_dict_get_compare(timestamp1,timestamp2,stamp2_name):
    name = stamp2_name
    stamp1 = timestamp1
    stamp2 = timestamp2
    index_relationship = {}
    point_index = 0
    for point in stamp1:
        point_index += 1
        index_relationship[point_index] = {}
        index2 = 0
        # print(point, stamp2[1])
        if point <= stamp2[len(stamp2) - 1]:
            for times in stamp2:
                index2 += 1
                # print(point - times)
                if (point - times) < 0:
                    break
            # print(index2)
            index_relationship[point_index][name] = index2 - 1
            # index_relationship[point_index]['lidar'] = times
            # index_relationship[point_index]['mpc - lidar'] = point - times
        else:
            index_relationship[point_index][name] = None
            # print(point_index)
            pass

    return index_relationship

# remove the additional mpc point in a lidar cycle time
def delet_timestamps(index_relation, timestamp2, stamp2_name):

    name = stamp2_name
    stamp2 = timestamp2
    index_relationship = index_relation


    for point in range(1, len(index_relationship)):
        index2 = index_relationship[point][name]
        if index2 is not None and index2 < (len(stamp2) - 1):
            point_compare = point + 1
            if index_relationship[point_compare][name] == index_relationship[point][name]:
                index_relationship[point][name] = None
            else:
                pass
        else:
            pass

    result = {}

    for point in index_relationship:
        index2 = index_relationship[point][name]
        if index2 is not None:
            result[point] = {}
            result[point][name] = index2
        else:
            pass
    # print(result)

    return result

def load_npz(npz_folder_path):
    #npz_folder = os.listdir(npz_folder_path)
    result_data_files = glob.glob(os.path.join(npz_folder_path, "*.npz"))
    result_data_files.sort()
    time_index_all = []
    point_lefty_all = []
    point_righty_all = []

    for file in result_data_files:
        print("processing %s" % file)
        content = np.load(file, allow_pickle=True)
        time_index = content['time_index']
        # point_x = content['point_x']
        # point_raw_lefty = content['point_raw_lefty']
        # point_raw_leftx = content['point_raw_leftx']
        # point_raw_righty = content['point_raw_righty']
        # point_raw_rightx = content['point_raw_rightx']
        point_lefty = content['point_lefty']
        point_righty = content['point_righty']
        time_index_all.append(time_index)
        point_lefty_all.append(point_lefty)
        point_righty_all.append(point_righty)
    index_name = np.vstack(time_index_all)
    point_lefty = np.vstack(point_lefty_all)
    point_righty = np.vstack(point_righty_all)
    return index_name, point_lefty, point_righty

def load_dasy(dasy_folder_path):
    dasy_folder = os.listdir(dasy_folder_path)

        # distinguish and traverse MF4_files in the mpc_folder
    for MF4_file in dasy_folder:
        if MF4_file[-4:] != '.MF4':
            continue
        else:
            MF4_file_path = os.path.join(dasy_folder_path, MF4_file)
            print("processing %s" % MF4_file)
            mdf = MDF(MF4_file_path, channels=vvr_channels)
            npz_name = os.path.join(MF4_file_path + ".npz")

            weeksignal = mdf.get('INS_Time_Week').samples
            msecsignal = mdf.get('INS_Time_msec').samples
            GPS_timestamps = mdf.get('INS_Time_msec').timestamps
            HeadingAngle = mdf.get('INS_Yaw').samples
            HeadingAngle_timestamps = mdf.get('INS_Yaw').timestamps
            # curvDasyRaw = mdf.get('_g_PL_AD_fw_DACoreCyclic_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._vxvRef_sw')
            # print(curvDasyRaw)

            GPS_Latitude = mdf.get('INS_Lat_Abs').samples
            GPS_Longitude = mdf.get('INS_Long_Abs').samples
            GPS_Lat_timestamps = mdf.get('INS_Lat_Abs').timestamps
            GPS_Long_timestamps = mdf.get('INS_Long_Abs').timestamps

            UTC_GPS_list = []

            # transfor gps-time to UTC_time
            for i in range(len(msecsignal)):

                gpsdate = weeksignal[i]
                gpstime = msecsignal[i]

                UTC_gps_time = Timezone_problem_gps(gpsdate, gpstime)
                UTC_GPS_list.append(UTC_gps_time)
                GPS_signal_UTC = timestamps_UTC(GPS_timestamps, UTC_GPS_list, GPS_Lat_timestamps)
                #shi jian zhou dui ying (MF4 TIME DUIYING GPS)

                inter_dict = {}

                inter_dict['UTC_GPS'] = GPS_signal_UTC
                inter_dict['GPS_Latitude'] = GPS_Latitude
                inter_dict['GPS_Longitude'] = GPS_Longitude

        np.savez(npz_name, HeadingAngle=HeadingAngle, UTC_GPS=GPS_signal_UTC, GPS_Latitude=GPS_Latitude, GPS_Longitude=GPS_Longitude)
        print("Done")

def data_combine(dasy_file, lidar_file):

    dasy_data = np.load(dasy_file, allow_pickle=True)
    index_name, point_lefty, point_righty = load_npz(lidar_file)

    GPS_UTC = dasy_data['UTC_GPS']
    GPS_Latitude = dasy_data['GPS_Latitude']
    GPS_Longitude = dasy_data['GPS_Longitude']
    HeadingAngle = dasy_data['HeadingAngle']


    compare_list_inter = index_nearst_relation_dict_get_compare(GPS_UTC, index_name, 'UTC_match')

    compare_list = delet_timestamps(compare_list_inter, lidar_UTC, 'UTC_match')

    ret = []
    ret1 = []
    ret2 = []
    ret3 = []
    ret4 = []
    ret5 = []
    ret6 = []
    for cal in compare_list.items():
        # every cycle of mf4-trigger, limited in lidar cycle
        if cal[1]['UTC_match'] is not None:

            # print(cal)

            index = cal[0]

            if index < len(compare_list_inter):

                cal[1]['HeadingAngle'] = mpc['HeadingAngle'][index]
                cal[1]['GPS_Lat'] = mpc['GPS_Latitude'][index]
                cal[1]['GPS_Long'] = mpc['GPS_Longitude'][index]



                if 120.7845 < cal[1]['GPS_Long'] and cal[1]['GPS_Long'] < 120.7868:

                    lidar_index = cal[1]['UTC_match']

                    # print(cal[1]['GPS_Long'])
                    cal[1]['point_x'] = np.arange(0, 61, 2)
                    # print(lidar_content[lidar_index][1])
                    cal[1]['point_raw_leftx'] = lidar_content[lidar_index][1]['point_raw_leftx']
                    cal[1]['point_raw_lefty'] = lidar_content[lidar_index][1]['point_raw_lefty']
                    cal[1]['point_raw_rightx'] = lidar_content[lidar_index][1]['point_raw_rightx']
                    cal[1]['point_raw_righty'] = lidar_content[lidar_index][1]['point_raw_righty']
                    cal[1]['point_lefty'] = lidar_content[lidar_index][1]['point_lefty']
                    cal[1]['point_righty'] = lidar_content[lidar_index][1]['point_righty']

                    cal[1]['val_point'] = cal_point(cal[1]['HeadingAngle'], cal[1]['GPS_Lat'], cal[1]['GPS_Long'],
                                                    cal[1]['point_x'], cal[1]['point_lefty'], cal[1]['point_righty'])
                    # print(cal[1]['point_lefty'][10], lidar_content[lidar_index][1]['time_index'])
                    cal[1]['left_lidar_point'] = cal_point_lidar(cal[1]['HeadingAngle'], cal[1]['GPS_Lat'], cal[1]['GPS_Long'],
                                                    cal[1]['point_raw_leftx'], cal[1]['point_raw_lefty'], side='left')

                    cal[1]['right_lidar_point'] = cal_point_lidar(cal[1]['HeadingAngle'], cal[1]['GPS_Lat'], cal[1]['GPS_Long'],
                                                      cal[1]['point_raw_rightx'], cal[1]['point_raw_righty'], side='right')

                    # error_index = np.argwhere(abs(cal[1]['point_raw_lefty']) < 0.3)
                    #
                    # np.delete(cal[1]['val_point'], error_index)

                    # for i in range(50):
                    #     plt.plot(-cal[1]['mpc_y'][i], cal[1]['point_x'][i], 'h', color='b')
                    #
                    #     # print(cal[1]['point_x'][i], cal[1]['point_lefty'][i])
                    # for i in range(len(cal[1]['left_plc_y'])):
                    #     plt.plot(-cal[1]['left_plc_y'][i], cal[1]['left_plc_x'][i], 'o')
                    # plt.title("left: blue-mpc", color="r")
                    # plt.show()
                    # append the gps standard points to every time cycle and add the lidar point to the lidar_point_contained cycle
                    if cal[1]['left_lidar_point'] is None:
                        # ret.append(cal[1]['val_point'])
                        ret1.append(cal[1]['GPS_Lat'])
                        ret2.append(cal[1]['GPS_Long'])
                        ret3.append(cal[1]['HeadingAngle'])
                        ret6.append(cal[1]['point_x'])
                    else:
                        ret.append(cal[1]['val_point'])
                        ret1.append(cal[1]['GPS_Lat'])
                        ret2.append(cal[1]['GPS_Long'])
                        ret3.append(cal[1]['HeadingAngle'])
                        ret4.append(cal[1]['left_lidar_point'])
                        ret5.append(cal[1]['right_lidar_point'])
                        ret6.append(cal[1]['point_x'])

                else:
                    pass
            else:
                pass
    #print(len(ret), "aiyo, paowanle")
    return ret, ret1, ret2, ret3, ret4, ret5, ret6, compare_list


def pointdata_compare_left(self):

    # plot lidar points
    val_point_gps, veh_lat, veh_long, veh_ha, left_lidar_point, right_lidar_point, point_x, time_list = data_combine('/home/vav/Validation_KPI/line_detection_v1.0/DASY/7-7-dasy/gps_1625624367470.1145.npz', '/home/vav/Validation_KPI/line_detection_v1.0/result/7-9-lidar/1625624378499.npz')
    print(veh_lat)

    # calculate dy and n from one pair files

    n_left = 0
    n_right = 0
    Dy_left = []
    left_wing_points = []
    right_wing_points = []
    first_start_point = [veh_long[0], veh_lat[0]]

    x_ego = []
    y_ego = []
    lane_x_left = []
    lane_x_right = []
    lane_y_left = []
    lane_y_right = []

    headingangle = veh_ha[0]
    headingangle = headingangle * math.pi / 180

    # get the left-lane, right-lane and ego_traj
    for i in range(len(self.left_standard)):
        x, y = cal_xy(self.left_standard[i][1]['Latitude'], self.left_standard[i][1]['Longitude'],
                        headingangle, first_start_point[1], first_start_point[0])
        plt.plot(-x, y, 'h', color='blue') #plt
        lane_y_left.append(y)
        lane_x_left.append(x)

    for i in range(len(self.right_standard)):
        x, y = cal_xy(self.right_standard[i][1]['Latitude'], self.right_standard[i][1]['Longitude'],
                          headingangle, first_start_point[1], first_start_point[0])
        lane_y_right.append(y)
        lane_x_right.append(x)

    for i in range(len(veh_long)):
        x, y = cal_xy(veh_lat[i], veh_long[i],
                          headingangle, first_start_point[1], first_start_point[0])
        y_ego.append(y)
        x_ego.append(x)
        plt.plot(-x, y, 'h', color='black')  # plt


        ''' get the lidar points'''

        for cal in range(len(veh_long)):
            start_point = [veh_long[cal], veh_lat[cal]]
            left_lidar = left_lidar_point[cal]  # lidar point
            right_lidar = left_lidar_point[cal]
            headingangle = veh_ha[cal]
            headingangle = headingangle * math.pi / 180
            lidar_fit = val_point_gps[cal]
            #
        for i in range(len(veh_long)):
            x, y = cal_xy(veh_lat[i], veh_long[i],
                              headingangle, first_start_point[1], first_start_point[0])
            y_ego.append(y)
            x_ego.append(x)
            plt.plot(-x, y, 'h', color='black')  # plt

        for point in left_lidar:

            x, y = cal_xy(point[0], point[1],
                          headingangle, first_start_point[1], first_start_point[0])
            poi = np.array([x, y])
            left_wing_points.append(poi)
            # for point in lidar_fit:
            #     x, y = cal_xy(point[0], point[1],
            #                   headingangle, first_start_point[1], first_start_point[0])
            #     plt.plot(-x, y, '.')
            plt.plot(-x, y, '.', color='red')  # plt
        plt.show()

        ''' get the lidar fit points'''
    lidar_lefty_fit, lidar_leftx_fit = find_line_full(left_wing_points)
        # print(lidar_lefty_fit)
        # print(len(lidar_lefty_fit))
    lidar_leftx_fit = lidar_leftx_fit * (-1)
    plt.plot(-lidar_leftx_fit, lidar_lefty_fit, '.', color='red')  # plt
    gps_lefty_cross = []

    for x in lidar_leftx_fit:
        for i in range(0, len(lane_x_left) - 1):
            if lane_x_left[i] <= x and x < lane_x_left[i+1]:
                y = get_line_cross(lane_x_left[i], lane_y_left[i], lane_x_left[i+1], lane_y_left[i+1], x)

                gps_lefty_cross.append(y)
        # print(gps_lefty_cross)
        # print(len(gps_lefty_cross))

        compare_number = min(len(lidar_lefty_fit), len(gps_lefty_cross))

        for idx in range(compare_number):
            dy = lidar_lefty_fit[idx] - gps_lefty_cross[idx]
            Dy_left.append(dy)
        Dy_left = np.array(Dy_left)
        left_Dy_mean = np.mean(Dy_left)
        Left_Dy_sigma = np.nanstd(Dy_left)
        print(left_Dy_mean, Left_Dy_sigma)

        sigma_space = str(left_Dy_mean - 2 * Left_Dy_sigma) + 'to' + str(left_Dy_mean + 2 * Left_Dy_sigma)
        fig, ax = plt.subplots(1, 1)
        plt.hist(Dy_left, 100, (-5, 5))
        ax.set_title(
            "Left distribution for Lat deviation at the distance of %s m \n $\mu =%s,$  \n 2sigma_space = %s " % (
            str(i), str(left_Dy_mean), sigma_space), color="b")
        plt.show()



if __name__ == '__main__':
    #load_npz()
    load_dasy("/home/henry/Second_kepper/z_dasy")