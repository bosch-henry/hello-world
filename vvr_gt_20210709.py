######ZYQ#######
##########
import numpy as np
from asammdf import MDF
import pandas as pd
import os
import shutil
import time
import numpy as np
from kpi_calculate_list import *
import matplotlib.pyplot as plt
import math
import csv
#from openpyxl import load_workbook
from binascii import a2b_hex
from data_process.point_transform import *

class gps_kpi_calculate():
    def __init__(self):
        self.UTC_time = 0
        self.gps_list = {}
        self.gps_standard = {}
    '''
    def asc_data_get(self):  #xiaotuiche 
        # get all the asc_file_folder
        current_path = os.path.dirname(os.path.abspath(__file__))
        asc_folder_path = os.path.join(current_path, 'line')
        asc_folder = os.listdir(asc_folder_path)

        for file in asc_folder:

            if file[-4:] != '.asc':
                continue
            else:

                name = str(file)

                self.gps_list[name] = []

                file_path = os.path.join(
                    asc_folder_path, file)

                content = pd.read_csv(file_path, skiprows=5, sep='\s+', names=['start_time', 'block_counter', 'pos', 'Rx/Tx',
                                                                               'd', '0', 'bit0', 'bit1', 'bit2', 'bit3', 'bit4', 'bit5',
                                                                               'bit6',  'bit7'])

                for i in range(len(content)):
                    pos = content.iloc[i]['pos']
                    if pos == '90':

                        bit0 = content.iloc[i]['bit0']
                        bit1 = content.iloc[i]['bit1']
                        bit2 = content.iloc[i]['bit2']
                        bit3 = content.iloc[i]['bit3']
                        bit4 = content.iloc[i]['bit4']
                        bit5 = content.iloc[i]['bit5']
                        bit6 = content.iloc[i]['bit6']
                        bit7 = content.iloc[i]['bit7']

                        latitude = a2b_hex(bit0 + bit1 + bit2 + bit3)
                        longitude = a2b_hex(bit4 + bit5 + bit6 + bit7)

                        latitude = int.from_bytes(latitude, byteorder='little', signed=True)*0.0000001
                        longitude = int.from_bytes(longitude, byteorder='little', signed=True)*0.0000001
                        # #
                        # latitude = '%.7f' % latitude
                        # longitude = '%.7f' % longitude
                        #
                        # print([latitude, longitude])

                        self.gps_list[name].append([latitude, longitude])

                        np.savez('%s/%s%s' % (asc_folder_path, 'asc_', name), gps_list=self.gps_list)

        return self.gps_list
  '''
    def excel_data_get(self):   # stand GPS
        left_standard ={}
        right_standard = {}
        wb = load_workbook('vvr.xlsx', read_only=True)
        x = wb.sheetnames
        GPS_worksheet = wb.get_sheet_by_name('vvr')
        count = 0
        for row in list(GPS_worksheet.rows)[1:23]:

            left_standard[count] = {}
            left_standard[count]['Latitude'] = row[2].value
            left_standard[count]['Longitude'] = row[3].value
            count += 1

        self.left_standard = sorted(left_standard.items(), key=lambda dd: dd[1]['Longitude'])
        count = 0
        for row in list(GPS_worksheet.rows)[1:41]:

            right_standard[count] = {}
            right_standard[count]['Latitude'] = row[4].value
            right_standard[count]['Longitude'] = row[5].value
            count += 1
        self.right_standard = sorted(right_standard.items(), key=lambda dd: dd[1]['Longitude'])
        return 0

    def dasy_data_get(self,dasy_folder_path):

        # get all the mpc_MF4_folder
        #current_path = os.path.dirname(os.path.abspath(__file__))
        #dasy_folder_path = os.path.join(current_path, 'DASY/7-7-dasy')
        # print(dasy_folder_path)
        mpc_folder = os.listdir(dasy_folder_path)

        # distinguish and traverse MF4_files in the mpc_folder
        for MF4_file in mpc_folder:
            if MF4_file[-4:] != '.MF4':
                continue
            else:
                MF4_file_path = os.path.join(dasy_folder_path, MF4_file)
                mdf = MDF(MF4_file_path, channels=vvr_channels)


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

                # plt.plot(GPS_signal_UTC, 'h')
                # plt.show()

                inter_dict = {}

                inter_dict['UTC_GPS'] = np.array(GPS_signal_UTC)
                inter_dict['GPS_Latitude'] = GPS_Latitude
                inter_dict['GPS_Longitude'] = GPS_Longitude

                # plt.plot(GPS_Latitude, GPS_Longitude, 'h', color='black')
                # plt.show()


                np.savez('%s/%s%s' % (dasy_folder_path, 'gps_', str(inter_dict['UTC_GPS'][0])), HeadingAngle=HeadingAngle,
                         UTC_GPS=GPS_signal_UTC, GPS_Latitude=GPS_Latitude, GPS_Longitude=GPS_Longitude)

    def lidar_data_get(self):
        # get all the npz_folder
        current_path = os.path.dirname(os.path.abspath(__file__))
        result_folder_path = os.path.join(current_path, 'result/7-9-lidar')
        # print(result_folder_path)
        npz_folder = os.listdir(result_folder_path)

        # distinguish and traverse npz_files in the result_folder
        npz_subfolder_list = []
        for npz_sub_folder in npz_folder:
            npz_sub_folder_path = os.path.join(result_folder_path, npz_sub_folder)
            if os.path.isdir(npz_sub_folder_path):
                npz_subfolder_list.append(npz_sub_folder_path)

        npz_file_list = []
        for path in npz_subfolder_list:
            sub_path = os.listdir(path)
            for npz_file in sub_path:
                if npz_file[-4:] == '.npz':
                    npz_file_path = os.path.join(path, npz_file)
                    # print(npz_file_path)
                    # npz_file_list.append(np.load(npz_file_path))
                    npz_file_list.append(npz_file_path)
        # print(npz_file_list)

        # resort the npz_file from lidar
        for file in npz_file_list:
            content = np.load(file, allow_pickle=True)

            time_index = content['time_index']
            point_raw_lefty = content['point_raw_lefty']
            point_raw_leftx = content['point_raw_leftx']
            point_raw_righty = content['point_raw_righty']
            point_raw_rightx = content['point_raw_rightx']
            point_lefty = content['point_lefty']
            point_righty = content['point_righty']

            # print(time_index)

            # y.sorted()
            lidar_npz = {}

            # print(y)
            count = 0
            for i in time_index:
                lidar_npz[count] = {}
                value = int(i)
                lidar_npz[count]['time_index'] = value
                # lidar_npz[count]['point_x'] = point_x[:]
                lidar_npz[count]['point_raw_lefty'] = point_raw_lefty[count]
                lidar_npz[count]['point_raw_leftx'] = point_raw_leftx[count]
                lidar_npz[count]['point_raw_righty'] = point_raw_righty[count]
                lidar_npz[count]['point_raw_rightx'] = point_raw_rightx[count]
                lidar_npz[count]['point_lefty'] = point_lefty[count]
                lidar_npz[count]['point_righty'] = point_righty[count]

                count += 1
            result = sorted(lidar_npz.items(), key=lambda dd: dd[1]['time_index'])

            print(result[0][0])
            # print(type(lidar_npz[0]['point_righty'][0]))
            # print(lidar_npz[0])
            np.savez('%s/%s' % (result_folder_path, str(result[0][1]['time_index'])), result)

    def pointdata_compare_left(self):
        # # plot asc points
        # # get all the asc_folder
        # current_path = os.path.dirname(os.path.abspath(__file__))
        # asc_folder_path = os.path.join(current_path, 'line')
        # # print(result_folder_path)
        # asc_folder = os.listdir(asc_folder_path)
        #
        #
        # for asc_file in asc_folder:
        #     if asc_file[-4:] == '.npz':
        #         name = asc_file[4:-4]
        #         asc_path = os.path.join(asc_folder_path, asc_file)
        #         asc = np.load(asc_path, allow_pickle=True)
        #
        #         inter = asc.files[0]
        #         asc_content = asc[inter]
        #         asc_dict = asc_content.item()
        #         asc_array = asc_dict['%s.asc'%(name)]
        #         for i in range(len(asc_array)):
        #             plt.plot(float(asc_array[i][1]), float(asc_array[i][0]), 'h', color='b')
        #
        # # plot excle points
        #
        # for i in range(len(self.left_standard)-2):
        #     plt.plot(self.left_standard[i][1]['Longitude'], self.left_standard[i][1]['Latitude'], 'h', color = 'red')
        #     plt.scatter(self.left_standard[i][1]['Longitude'], self.left_standard[i][1]['Latitude'], color = 'red')
        # for i in range(len(self.right_standard) - 2):
        #     plt.plot(self.right_standard[i][1]['Longitude'], self.right_standard[i][1]['Latitude'], 'h', color='blue')
        #     plt.scatter(self.right_standard[i][1]['Longitude'], self.right_standard[i][1]['Latitude'], color = 'blue')
        # plt.show()
        # print(len(self.left_standard))
        # print(len(self.right_standard))

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

        ################################

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
            # for i in range(len(self.left_standard)):
            #     x, y = cal_xy(self.left_standard[i][1]['Latitude'], self.left_standard[i][1]['Longitude'],
            #                   headingangle, first_start_point[1], first_start_point[0])
            #     plt.plot(-x, y, 'h', color='blue')  # plt
            #     lane_y_left.append(y)
            #     lane_x_left.append(x)
            # plt.show()

            # for point in right_lidar:
            #     x, y = cal_xy(point[0], point[1],
            #               headingangle, first_start_point[1], first_start_point[0])
            #     right_wing_points.append([x, y])

        # plt.show() #plt
        # print(left_wing_points)
        ################################

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
    ################################################
        # Dy_left = np.array(Dy_left)
        # Dy_left = Dy_left[1:]
        # # print(Dy_left[1][1])
        # for i in range(10, 60, 10):
        #     print(i)
        #     Dy = Dy_left[:][int(i / 2)]
        #     print(Dy)
        #     nan_index = np.argwhere(np.isnan(Dy))
        #     np.delete(Dy, nan_index)
        #     # delete_index = np.argwhere(abs(Dy) > 2)
        #     # print(len(delete_index) / len(Dy))
        #     # Dy[delete_index] = None
        #     left_Dy_mean = np.nanmean(Dy)
        #     maxx = max(Dy)
        #
        #     sigma = np.nanstd(Dy)
        #     sigma_space = str(left_Dy_mean - 2 * sigma) + 'to' + str(left_Dy_mean + 2 * sigma)
        #     fig, ax = plt.subplots(1, 1)
        #     plt.hist(Dy_left[:][int(i / 2)], 100, (-5, 5))
        #     ax.set_title(
        #         "Left distribution for Lat deviation at the distance of %s m \n $\mu =%s,$  \n 2sigma_space = %s " % (
        #         str(i), str(left_Dy_mean), sigma_space), color="b")
        #     plt.show()


    def pointdata_compare_right(self):


        # # plot asc points
        # # get all the asc_folder
        # current_path = os.path.dirname(os.path.abspath(__file__))
        # asc_folder_path = os.path.join(current_path, 'line')
        # # print(result_folder_path)
        # asc_folder = os.listdir(asc_folder_path)
        #
        #
        # for asc_file in asc_folder:
        #     if asc_file[-4:] == '.npz':
        #         name = asc_file[4:-4]
        #         asc_path = os.path.join(asc_folder_path, asc_file)
        #         asc = np.load(asc_path, allow_pickle=True)
        #
        #         inter = asc.files[0]
        #         asc_content = asc[inter]
        #         asc_dict = asc_content.item()
        #         asc_array = asc_dict['%s.asc'%(name)]
        #         for i in range(len(asc_array)):
        #             plt.plot(float(asc_array[i][1]), float(asc_array[i][0]), 'h', color='b')

        # plot excle points

        # for i in range(len(self.left_standard)-2):
        #     plt.plot(self.left_standard[i][1]['Longitude'], self.left_standard[i][1]['Latitude'], 'h', color = 'red')
        #     plt.scatter(self.left_standard[i][1]['Longitude'], self.left_standard[i][1]['Latitude'], color = 'red')
        # for i in range(len(self.right_standard) - 2):
        #     plt.plot(self.right_standard[i][1]['Longitude'], self.right_standard[i][1]['Latitude'], 'h', color='blue')
        #     plt.scatter(self.right_standard[i][1]['Longitude'], self.right_standard[i][1]['Latitude'], color = 'blue')
        # plt.show()
        # print(len(self.left_standard))
        # print(len(self.right_standard))

        # plot lidar points
        val_point_gps, veh_lat, veh_long, veh_ha, left_y, right_y, point_x = data_combine('/home/vav/Validation_KPI/line_detection_v1.0/DASY/7-7-dasy/gps_1625625432199.3904.npz', '/home/vav/Validation_KPI/line_detection_v1.0/result/7-7-lidar/1625625436699.npz')
        # print(veh_lat)

        # calculate dy and n from one pair files

        n_left = 0
        n_right = 0
        Dy_left = []
        for punkt in range(len(veh_long)):
            dy_left = []
            dy_right = []
            start_point = [veh_long[punkt], veh_lat[punkt]]
            # plt.plot(start_point[0], start_point[1], 'h', color='blue')
            # plt.plot(start_point[0], veh_ha[punkt], 'h', color='red')
            headingangle = veh_ha[punkt]
            headingangle = headingangle * math.pi / 180
            for cal in range(len(point_x)):
                x_distance = point_x[cal]
                left_y_distance = left_y[cal]  # lidar point
                right_y_distance = right_y[cal]
                left_ycross_distance = []   # standard point
                right_ycross_distance = []
                left_xcross_distance = []
                right_xcross_distance = []
                # get the lane x,y distance to ego vehicle
                x_ego = []
                y_ego = []
                lane_x_left = []
                lane_x_right = []
                lane_y_left = []
                lane_y_right = []
                # calculate the relativ distance of point on lane lines to the ego vehicle
                for i in range(len(self.left_standard)):
                    x, y = cal_xy(self.left_standard[i][1]['Latitude'], self.left_standard[i][1]['Longitude'],
                                  headingangle, start_point[1], start_point[0])
                    # plt.plot(x, y, 'h', color='blue') # plt
                    lane_y_left.append(y)
                    lane_x_left.append(x)

                # get the ego vehicle in x,y field
                for i in range(len(veh_long)):
                    x, y = cal_xy(veh_lat[i], veh_long[i],
                                  headingangle, start_point[1], start_point[0])
                    y_ego.append(y)
                    x_ego.append(x)

                # plt.plot(x_ego[punkt], y_ego[punkt], 'h', color='black') # plt
                # plt.show()
                # get the cross points of lane and ego lanes

                # print(headingangle * 180/ math.pi)
                inter_k = - math.cos(headingangle)/math.sin(headingangle)
                inter_b = [0, 0]
                k = -1 / inter_k
                for x_dist in x_distance[1:]:
                    b = x_dist / math.cos(headingangle)
                    for i in range(0, len(lane_x_left) - 1):
                        line1 = [lane_x_left[i], lane_y_left[i], lane_x_left[i+1], lane_y_left[i+1]]
                        left_cross = cross_point(line1, k, b)


                        if left_cross is not None:
                            left_ycross_distance.append(left_cross[1])
                            left_xcross_distance.append(left_cross[0])
                            # plt.plot(left_cross[0], left_cross[1], 'h', color='red') # plt
                        else:
                            pass
                # plt.show() # plt
                # print(left_ycross_distance)
                # print(right_y_distance)
                n_left += len(left_xcross_distance)
                n_right += len(right_xcross_distance)
                dy_left = []
                dy_right = []
                for i in range(len(left_xcross_distance)):

                    dy_left.append(left_ycross_distance[i] - (-right_y_distance[i]))
                    # print(left_ycross_distance[i], left_y_distance[i])
                dy_left = np.array(dy_left)
                Dy_left.append(dy_left)

        Dy_left = np.array(Dy_left)
        Dy_left = Dy_left[1:]
        print(Dy_left[1][1])
        for i in range(10, 60, 10):
            print(i)
            Dy = Dy_left[:][int(i / 2)]
            print(Dy)
            nan_index = np.argwhere(np.isnan(Dy))
            np.delete(Dy, nan_index)
            # delete_index = np.argwhere(abs(Dy) > 2)
            # print(len(delete_index) / len(Dy))
            # Dy[delete_index] = None
            left_Dy_mean = np.nanmean(Dy)
            maxx = max(Dy)

            sigma = np.nanstd(Dy)
            sigma_space = str(left_Dy_mean - 2 * sigma) + 'to' + str(left_Dy_mean + 2 * sigma)
            fig, ax = plt.subplots(1, 1)
            plt.hist(Dy_left[:][int(i / 2)], 100, (-2, 2))
            ax.set_title(
                "right distribution for Lat deviation at the distance of %s m \n $\mu =%s,$  \n 2sigma_space = %s " % (
                str(i), str(left_Dy_mean), sigma_space), color="b")
            plt.show()



            # for i in range(len(lane_x_left)):
            #     plt.plot(lane_x_left[i], lane_y_left[i], 'h', color='blue')
            # for i in range(len(lane_x_right)):
            #     plt.plot(lane_x_right[i], lane_y_right[i], 'h', color='green')
            #
            # plt.plot(x_ego[punkt], y_ego[punkt], 'h', color='black')
            # for i in range(len(left_xcross_distance)):
            #     # print(left_xcross_distance)
            #     plt.plot(left_ycross_distance, left_xcross_distance, 'h', color='red')
            # for i in range(len(right_xcross_distance)):
            #     # print(left_xcross_distance)
            #     plt.plot(right_ycross_distance, right_xcross_distance, 'h', color='red')
            # plt.show()













        # # plot the heading angle
        # for ha in range(len(ret1)):
        #     plt.plot(veh_long[ha], veh_ha[ha], 'h', color='r')
        # plt.show()
        # # plot the lidar points
        # for i in val_point_gps:
        #     for m in i:
        #         plt.plot(m[1], m[0], 'g^')
        # # plot the vehicle position
        # for x in range(len(ret1)):
        #     plt.plot(veh_long[x], veh_lat[x], 'h', color='black')
        # plt.show()
    ##################Iide Function##############################
def Timezone_problem_gps(week, msecond):

    time = 7*24*60*60*1000*week + msecond + 315964800*1000

    return time

def timestamps_UTC(GPS_timestamps, UTC_GPS_list, leftline_timestamps):

    time_gap = GPS_timestamps[1] - GPS_timestamps[0]
    UTC_gap = (UTC_GPS_list[-1] - UTC_GPS_list[0])/len(UTC_GPS_list)
    ratio = UTC_gap/time_gap
    # rightline_UTC = []
    leftline_UTC = []
    # print(UTC_GPS_list)
    # print(ratio)
    # print(UTC_GPS_list[101], UTC_GPS_list[100], UTC_GPS_list[2])

    # for i in range(len(rightline_timestamps)):
    #     rightline_UTC.append(UTC_GPS_list[0] + ratio * (rightline_timestamps[0] - GPS_timestamps[0]) + ratio * (rightline_timestamps[i] - rightline_timestamps[0]))
    for i in range(len(leftline_timestamps)):

        leftline_UTC.append(UTC_GPS_list[0] + ratio * (leftline_timestamps[0] - GPS_timestamps[0]) + ratio * (leftline_timestamps[i] - leftline_timestamps[0]))

    return leftline_UTC

# get the nearst timestamp relationship dict for two signals
def index_nearst_relation_dict_get(timestamp1,timestamp2,index_gap, stamp2_name):
    name = stamp2_name
    stamp1 = timestamp1
    stamp2 = timestamp2
    index_relationship = {}
    for point in stamp1:
        i = np.where(stamp1 == point)
        differenz_list = []
        point_index = int(i[0])
        if point_index >= index_gap and point_index <= (len(stamp2) - index_gap):
            begin_point = point_index - index_gap
            stop_point = point_index + index_gap
        elif point_index < index_gap:
            begin_point = 0
            stop_point = point_index + index_gap
        elif point_index > (len(stamp2) - index_gap):
            begin_point = point_index - index_gap
            stop_point = len(stamp2)
        else:
            pass
        # print(point_index, begin_point, stop_point)
        for times in stamp2[begin_point:stop_point]:
            differenz_list.append(abs(point - times))
        j = differenz_list.index(min(differenz_list))
        index_relationship[point_index] = {}

        index_relationship[point_index][stamp2_name] = j + begin_point

    return index_relationship

# link the mpc_point to the belonged lidar cycle time
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

# calculate the line points in front of the vehicle
def cal_point(HeadingAngle, GPS_Lat, GPS_Long, x_distance, left_y_distance, right_y_distance):

    # calculate the ratio between meter and GPS coordinate
    cal_Latitude = GPS_Lat

    a = 6378137

    b = 6356752.3142

    r_earth = math.sqrt(((((a ** 2) * math.cos(cal_Latitude)) ** 2) + (((b ** 2) * math.cos(cal_Latitude)) ** 2))/(((a * math.cos(cal_Latitude)) ** 2) + ((b * math.cos(cal_Latitude)) ** 2)))


    longi_unit = (math.pi/180) * r_earth * math.cos(cal_Latitude * math.pi/180)

    lati_unit = (math.pi/180) * r_earth

    # calculate the GPS coordinate of the lidar points
    Point = []

    HeadingAngle = HeadingAngle * math.pi / 180

    for i in range(len(left_y_distance)):

        x_distance[i] = abs(x_distance[i])
        left_y_distance[i] = abs(left_y_distance[i])
        right_y_distance[i] = abs(right_y_distance[i])

        Point_Longi = (- math.sin(HeadingAngle) * x_distance[i] - math.cos(HeadingAngle) * left_y_distance[i])/longi_unit + GPS_Long
        Point_Lat = (math.cos(HeadingAngle) * x_distance[i] - math.sin(HeadingAngle) * left_y_distance[i])/lati_unit + GPS_Lat
        Point.append([Point_Lat, Point_Longi])

    for i in range(len(right_y_distance)):

        x_distance[i] = abs(x_distance[i])
        left_y_distance[i] = abs(left_y_distance[i])
        right_y_distance[i] = abs(right_y_distance[i])

        Point_Longi = (- math.sin(HeadingAngle) * x_distance[i] + math.cos(HeadingAngle) * right_y_distance[i])/longi_unit + GPS_Long
        Point_Lat = (math.cos(HeadingAngle) * x_distance[i] + math.sin(HeadingAngle) * right_y_distance[i])/lati_unit + GPS_Lat
        Point.append([Point_Lat, Point_Longi])

    ret = Point
    # for i in ret:
    #     plt.plot(i[1], i[0], 'h')
    # plt.show()
    # calculate the lidar point at every 100m cycle

    return ret
# calculate the lidar points in front of the vehicle
def cal_point_lidar(HeadingAngle, GPS_Lat, GPS_Long, x_distance, y_distance, side):

    # calculate the ratio between meter and GPS coordinate
    cal_Latitude = GPS_Lat

    a = 6378137

    b = 6356752.3142

    r_earth = math.sqrt(((((a ** 2) * math.cos(cal_Latitude)) ** 2) + (((b ** 2) * math.cos(cal_Latitude)) ** 2))/(((a * math.cos(cal_Latitude)) ** 2) + ((b * math.cos(cal_Latitude)) ** 2)))


    longi_unit = (math.pi/180) * r_earth * math.cos(cal_Latitude * math.pi/180)

    lati_unit = (math.pi/180) * r_earth

    # calculate the GPS coordinate of the lidar points
    Point = []

    HeadingAngle = HeadingAngle * math.pi / 180

    if side == 'left':

        for i in range(len(y_distance)):

            x_distance[i] = abs(x_distance[i])
            y_distance[i] = abs(y_distance[i])


            Point_Longi = (- math.sin(HeadingAngle) * x_distance[i] - math.cos(HeadingAngle) * y_distance[i])/longi_unit + GPS_Long
            Point_Lat = (math.cos(HeadingAngle) * x_distance[i] - math.sin(HeadingAngle) * y_distance[i])/lati_unit + GPS_Lat
            Point.append([Point_Lat, Point_Longi])

    elif side == 'right':
        for i in range(len(y_distance)):

            x_distance[i] = abs(x_distance[i])
            y_distance[i] = abs(y_distance[i])

            Point_Longi = (- math.sin(HeadingAngle) * x_distance[i] + math.cos(HeadingAngle) * y_distance[i])/longi_unit + GPS_Long
            Point_Lat = (math.cos(HeadingAngle) * x_distance[i] + math.sin(HeadingAngle) * y_distance[i])/lati_unit + GPS_Lat
            Point.append([Point_Lat, Point_Longi])

    else:
        print('No side Input during the lidar points calculation')

    ret = Point
    # for i in ret:
    #     plt.plot(i[1], i[0], 'h')
    # plt.show()
    # calculate the lidar point at every 100m cycle

    return ret
# calculate the x,y distance of every point
def cal_xy(point_lat, point_longi, headingangle, start_lat, start_longi):

    # calculate the ratio between meter and GPS coordinate
    cal_Latitude = start_lat

    a = 6378137

    b = 6356752.3142

    r_earth = math.sqrt(((((a ** 2) * math.cos(cal_Latitude)) ** 2) + (((b ** 2) * math.cos(cal_Latitude)) ** 2)) / (
                ((a * math.cos(cal_Latitude)) ** 2) + ((b * math.cos(cal_Latitude)) ** 2)))

    longi_unit = (math.pi / 180) * r_earth * math.cos(cal_Latitude * math.pi / 180)
    lati_unit = (math.pi / 180) * r_earth

    longi = (point_longi - start_longi) * longi_unit
    lat = (point_lat - start_lat) * lati_unit

    headingangle = headingangle * math.pi / 180
    x_dist = lat * math.sin(headingangle) + longi * math.cos(headingangle)
    y_dist = longi * math.sin(headingangle) - lat * math.cos(headingangle)

    return x_dist, y_dist
# get the cross point from two lines
def cross_point(line1, line2_k, line2_b):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    k1 = (y2 - y1) / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 - x1 * k1  # 整型转浮点型是关键
    k2 = line2_k
    b2 = line2_b

    x = (b2 - b1)  / (k1 - k2)
    y = k1 * x  + b1

    if x >= min(x1, x2) and x <= max(x1, x2):
        return [x, y]
    else:
        return None
# get the line from two points and get the specific number at some x
def get_line_cross(x1, y1, x2, y2, x):
    k1 = (y2 - y1) / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 - x1 * k1  # 整型转浮点型是关键
    y = k1 * x + b1
    return y
def data_combine(mpc_file, lidar_file):

    mpc = np.load(mpc_file, allow_pickle=True)
    lidar = np.load(lidar_file, allow_pickle=True)

    inter = lidar.files[0]
    lidar_content = lidar[inter]

    lidar_UTC = []

    for i in range(len(lidar_content)):
        lidar_UTC.append(lidar_content[i][1]['time_index'])

    # get the mpc_UTC time
    GPS_UTC = mpc['UTC_GPS']


    # n = len(leftline_UTC)
    # print(n)
    # get the compare_list to do
    # print(GPS_UTC[400])
    # print(lidar_UTC)
    compare_list_inter = index_nearst_relation_dict_get_compare(GPS_UTC, lidar_UTC, 'UTC_match')
    # print(compare_list_inter)



    compare_list = delet_timestamps(compare_list_inter, lidar_UTC, 'UTC_match')
    # print(compare_list)




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
                        # for i in cal[1]['val_point']:
                        #     plt.plot(i[1], i[0], 'h')
                        # plt.show()
                        # count = 0
                        # for i in cal[1]['point_x']:
                        #     plt.plot(i, cal[1]['point_lefty'][count], 'h')
                        #     plt.plot(i, cal[1]['point_righty'][count], 'h')
                        #     count += 1
                        # plt.show()
                else:
                    pass
            else:
                pass
    #print(len(ret), "aiyo, paowanle")
    return ret, ret1, ret2, ret3, ret4, ret5, ret6, compare_list


if __name__ == '__main__':
    cal = gps_kpi_calculate()
    # ret = cal.asc_data_get()
    #ret = cal.excel_data_get()
    ret = cal.dasy_data_get("/home/henry/test_data/GT_VVR/VVR_0715")
    # rer = cal.lidar_data_get()
    ret = cal.pointdata_compare_left()
    ret = cal.pointdata_compare_right()

