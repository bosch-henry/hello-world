#  BOSCH
#  xhe4szh
#  7/21/2021 5:30 PM
import numpy as np
from asammdf import MDF
import pandas as pd
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

vvr_channels = ['INS_Time_Week','INS_Time_msec','INS_Yaw','INS_Lat_Abs','INS_Long_Abs']



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

    return leftline_UTC


def dasy_data_get(dasy_folder_path):

    mpc_folder = os.listdir(dasy_folder_path)

    # distinguish and traverse MF4_files in the mpc_folder
    for MF4_file in mpc_folder:
        if MF4_file[-4:] != '.MF4':
            continue
        else:
            MF4_file_path = os.path.join(dasy_folder_path, MF4_file)
            print("processing..%s" % MF4_file )
            mdf = MDF(MF4_file_path, channels=vvr_channels)

            weeksignal = mdf.get('INS_Time_Week').samples
            msecsignal = mdf.get('INS_Time_msec').samples
            GPS_timestamps = mdf.get('INS_Time_msec').timestamps
            HeadingAngle = mdf.get('INS_Yaw').samples
            HeadingAngle_timestamps = mdf.get('INS_Yaw').timestamps
            #curvDasyRaw = mdf.get('_g_PL_AD_fw_DACoreCyclic_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._vxvRef_sw')
            #print(curvDasyRaw)

            GPS_Latitude = mdf.get('INS_Lat_Abs').samples
            GPS_Longitude = mdf.get('INS_Long_Abs').samples
            GPS_Lat_timestamps = mdf.get('INS_Lat_Abs').timestamps
            GPS_Long_timestamps = mdf.get('INS_Long_Abs').timestamps
            UTC_GPS_list = []
            if weeksignal != [] and msecsignal != []:
                # transfor gps-time to UTC_time
                for i in range(len(msecsignal)):
                    gpsdate = weeksignal[i]
                    gpstime = msecsignal[i]

                    UTC_gps_time = Timezone_problem_gps(gpsdate, gpstime)

                    UTC_GPS_list.append(UTC_gps_time)

                GPS_signal_UTC = timestamps_UTC(GPS_timestamps, UTC_GPS_list, GPS_Lat_timestamps)
                # shi jian zhou dui ying (MF4 TIME DUIYING GPS)

                # plt.plot(GPS_signal_UTC, 'h')
                # plt.show()

                inter_dict = {}
                inter_dict['UTC_GPS'] = np.array(GPS_signal_UTC)
                inter_dict['GPS_Latitude'] = GPS_Latitude
                inter_dict['GPS_Longitude'] = GPS_Longitude
                npz_name = os.path.join(dasy_folder_path, MF4_file[:-5] + '.npz')
                np.savez(npz_name, HeadingAngle=HeadingAngle, UTC_GPS=GPS_signal_UTC, GPS_Latitude=GPS_Latitude, GPS_Longitude=GPS_Longitude)
            else:
                print("%s have no ADMA data" % MF4_file)

def load_print_NPZ(npz_folder_file):
    npz_folder = os.listdir(npz_folder_file)

    # distinguish and traverse MF4_files in the mpc_folder
    for npz_file in npz_folder:
        if npz_file[-4:] != '.npz':
            continue
        else:
            npz_file_path = os.path.join(npz_folder_file, npz_file)
            print("loading..%s" % npz_file)
            npz_data = np.load(npz_file_path)
            print(npz_data["HeadingAngle"])


if __name__ == '__main__':
    dasy_data_get("/home/henry/Second_kepper/z_dasy")
 #   load_print_NPZ("/home/henry/Second_kepper/z_dasy")

