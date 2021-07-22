# Selected lidar ids
LIDAR_IDs = ["1"]

# Bird view map setttings
BV_COMMON_SETTINGS = {
    # Height shift to make the z-axis value of ground be 0
    "train_height_shift": 0,
    # Minimum z-axis value of the interval to select points near the ground
    "shifted_min_height": -1,
    # Maximum z-axis value of the interval to select points near the ground
    "shifted_max_height": 1,
    # 1 meter in x-axis corresponds to "distance_resolution_train" pixels on the bird view map
    "distance_resolution_train": 6,
    # 1 meter in y-axis corresponds to "width_resolution_train" pixels on the bird view map
    "width_resolution_train": 30,
    # Point radius on the bird view
    "point_radius_train": 1.5,
    # If intensity value of one point is bigger thatn truncation_max_intensiy, the intensity will be set to truncation_max_intensiy
    "truncation_max_intensiy": 0.08,
    # Intensity shift to make the area with points(intensity may be 0) different with that without points
    "train_background_intensity_shift": 0.1,

    "line_fitted_start_value":0
}

MODEL_NAME = "./model/livox_lane_det.pth"
GPU_IDs = [0]

TEST_DATA_FOLDER = "/media/henry/7457C_Lidar/10minutes_test"
#RESULT_FOLDER = "./result/Second_kepper"
#VIS_FOLDER = "./result/points_vis/"
#POINTS_WITH_CLASS_FOLDER = "./result/points_with_class"
bv_settings = {
        # farthest detection distnace in front of the car
        "max_distance": 60,
        # farthest detection distnace behind the car
        "min_distance": -20,
        # farthest detection distance to the left of the car
        "left_distance": 20,
        # farthest detection distance to the right of the car
        "right_distance": 20
    }

