import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from PIL import Image

sys.path.append("..")
from util.color_table import color_table
from util.color_table_for_class import color_table_for_class
from shapely import geometry


P_world_default = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]).astype("float32")
'''
P_world_default_2 = np.array([[0.99998629, 0, 0.005236, 0],
                            [0, 1, 0, 0],
                            [-0.005236, 0, 0.99998629, 0],
                            [0, 0, 0, 1]]).astype("float32")                          
P_world_default = np.array([[0.99997563, 0.00698126, 0],
                            [-0.00698126, 0.99997563, 0],
                            [0, 0, 1]]).astype("float32")
P_world_default_2 = np.array([[0.9998477, 0, 0.0174524],
                              [0, 1, 0],
                              [-0.0174524, 0, 0.9998477]]).astype("float32")
'''
Alfa = 0  * math.pi/180     #rote by x axle, roll angle
Beta = 0  * math.pi/180     #rote by y axle, pich angle
Gama = 0.7  * math.pi/180    #rote by z axle, yaw angle

R_Alfa = np.array([[1, 0 ,0],[0, math.cos(Alfa), -math.sin(Alfa)], [0, math.sin(Alfa), math.cos(Alfa)]]).astype("float32")
R_Beta = np.array([[math.cos(Beta), 0, math.sin(Beta)], [0, 1, 0], [-math.sin(Beta), 0, math.cos(Beta)]]).astype("float32")
R_Gama = np.array([[math.cos(Gama), -math.sin(Gama), 0],[math.sin(Gama), math.cos(Gama), 0], [0, 0, 1]]).astype("float32")

R = np.zeros((3, 4))
T = np.array([0, 0, 0])
R[:3, :3] = np.array([R_Alfa.dot(R_Beta).dot(R_Gama)])
R[:, 3] = T.T

P = np.vstack((R, np.array([0, 0, 0, 1])))


def ProjectPointsToWorld_one(points_input_set):

    P_world = P

    points0 = points_input_set[:, :4].copy()
    points0[:, 3] = 1
    points1 = P_world.dot(points0.T)
    points1 = points1.T

    points_output_set= points_input_set.copy()
    points_output_set[:, :3] = points1[:, :3]
    return points_output_set



def ProjectPointsToWorld(points_input_set):
    lidar_id_list = points_input_set.keys()
    points_output_set = {}

    for lidar_id in lidar_id_list:
        P_world = P_world_default

        points0 = points_input_set[lidar_id][:, :4].copy()
        points0[:, 3] = 1
        points1 = P_world.dot(points0.T)
        points1 = points1.T

        points_output_set[lidar_id] = points_input_set[lidar_id].copy()
        points_output_set[lidar_id][:, :3] = points1[:, :3]

    return points_output_set


def MergePoints(points_input_set):
    lidar_id_list = points_input_set.keys()
    points_list = []

    for lidar_id in lidar_id_list:
        points_list.append(points_input_set[lidar_id])

    points_output = np.concatenate(points_list, axis = 0)

    return points_output


# R = RzRxRy
def GetWorldToCamMatrix(new_angle, T):
    angle_x, angle_y, angle_z = new_angle
    R_x = [[1,         0,                 0         ],
           [0, math.cos(angle_x), -math.sin(angle_x)],
           [0, math.sin(angle_x),  math.cos(angle_x)]]
    R_y = [[ math.cos(angle_y), 0, math.sin(angle_y)],
           [0,                  1,        0        ],
           [-math.sin(angle_y), 0, math.cos(angle_y)]]
    R_z = [[math.cos(angle_z), -math.sin(angle_z), 0],
           [math.sin(angle_z),  math.cos(angle_z), 0],
           [       0,                   0,         1]]
    R_x = np.array(R_x)
    R_y = np.array(R_y)
    R_z = np.array(R_z)
    R = np.dot(R_z, np.dot(R_x, R_y))
    P = np.zeros([4, 4])
    P[:3, :3] = R
    P[:3, 3] = T
    P[3, 3] = 1
    return P


def GetMatrices(img_shape):
    img_h, img_w = img_shape
    '''
    P_world = np.array([[0.999972   ,           0,  0.00750485,           0],
                        [0          ,           1,           0,           0],
                        [-0.00750485,           0,    0.999972,       1.909],
                        [0          ,           0,          0,           1]])
    '''
    P_world = np.array([[ 1,   0,    0,    0],
                        [ 0,   1,    0,    0],
                        [ 0,   0,    1,    0],
                        [ 0,   0,    0,    1]]).astype("float32")

    K = np.array([[ 380.0,      0, float(img_w / 2), 0],
                  [     0,  380.0, float(img_h / 2), 0], 
                  [     0,      0,                1, 0]])

    #if you want to change the view point, you can change T and angle
    T = np.array([0, 0, 38]).T
    #angle = [0, -math.pi*(16.0/18.0), math.pi/2]
    angle = [0, -math.pi, math.pi/2]
    P = GetWorldToCamMatrix(angle, T)

    #print(K)
    #print(P)
    #print(P_world)  
    return K, P, P_world


def DrawPointOnImg(img, im_x, im_y, color):
    vis_point_radius = 1

    im_height, im_width = img.shape[:2]

    start_x = max(0, np.ceil(im_x - vis_point_radius))   #np.ceil 向上取整
    end_x = min(im_width -1, np.ceil(im_x + vis_point_radius) - 1) + 1
    start_y = max(0, np.ceil(im_y - vis_point_radius))
    end_y = min(im_height -1, np.ceil(im_y + vis_point_radius) - 1) + 1
    start_x = int(start_x)
    start_y = int(start_y)
    end_x = int(end_x)
    end_y = int(end_y)

    for yy in range(start_y, end_y):
        for xx in range(start_x, end_x):
            img[yy, xx, :] = color


def GetProjectImage(points, color_info, img_shape, K, P, P_world, color_info_table, draw_virtual_car=True):
    BLANK_COLOR = [10, 10, 10]

    img_h, img_w = img_shape

    N = points.shape[0]

    points_pos = np.concatenate([points[:, :3].copy(), np.ones([N, 1])], axis=1)
    points2 = np.dot(np.linalg.inv(P_world), np.transpose(points_pos))
    points3 = np.dot(P, points2)
    points_cam = np.dot(K, points3)

    img_show = (np.ones([img_h, img_w, 3])*BLANK_COLOR[0]).astype(np.uint8)

    for i in range(N):
        if points_cam[2, i] < 0:
            continue
        im_x = points_cam[0, i] / points_cam[2, i]
        im_y = points_cam[1, i] / points_cam[2, i]

        if 0 <= im_x < img_w and 0 <= im_y < img_h:
            color_info_cur = color_info[i]
            point_color = color_info_table[int(color_info_cur)]
            DrawPointOnImg(img_show, im_x, im_y, point_color)

    if draw_virtual_car:
        car_ctx = img_w / 2
        car_cty = img_h / 2 
        car_w = 10
        car_h = 15
        car_points = [(int(car_ctx-car_w), int(car_cty)),
                      (int(car_ctx-car_w/2), int(car_cty-car_h)),
                      (int(car_ctx+car_w/2), int(car_cty-car_h)),
                      (int(car_ctx+car_w), int(car_cty)), 
                      (int(car_ctx+car_w/2), int(car_cty+car_h)),
                      (int(car_ctx-car_w/2), int(car_cty+car_h))]
        img_class = cv2.fillPoly(img_show, [np.array(car_points)], [255, 255, 255])

    return img_show


def VisualizePointsClass(points_input):
    output_img_h = 1080
    output_img_w = 1920
    output_img_w1 = int(output_img_w / 2)

    K, P, P_world = GetMatrices([output_img_h, output_img_w1])

    intensity_show = GetProjectImage(points_input, 
                                     points_input[:, 3]*255, 
                                     [output_img_h, output_img_w1], 
                                     K, P, P_world, color_table)
    class_show = GetProjectImage(points_input, 
                                     points_input[:, 4], 
                                     [output_img_h, output_img_w1], 
                                     K, P, P_world, color_table_for_class)

    vis_img = np.concatenate([intensity_show, class_show], axis = 1)
    #vis_img = np.concatenate(class_show, axis = 0)

    return vis_img


def histogram_view(value):
    nonzeroy = value[:, 1]
    #nonzeroy = point[:, 1] * (-1)
    nonzerox = value[:, 0]
    # matplotlib.axes.Axes.hist() 方法的接口
    n, bins, patches = plt.hist(nonzeroy, 50,(-4,0))
    bins_count = np.argmax(n)
    origin_left = 0.5*(-4+(4/50)*bins_count+(-4+(4/50)*(bins_count+1)))

    n, bins, patches = plt.hist(nonzeroy, 50,(0,4))
    bins_count = np.argmax(n)
    origin_right = 0.5*((4/50)*bins_count+((4/50)*(bins_count+1)))
    
    print(bins_count)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Dy')
    plt.ylabel('count')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # 设置y轴的上限
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def center_exchange(center_y, point_fit, dx):
    dev = center_y[1:] - center_y[:-1]
    exchange_index = np.where(dev == 0)[0] + 1
    func_ = np.poly1d(np.array(point_fit).astype(float))
    center_y[exchange_index] = func_(dx[exchange_index])
    center_y[0] = func_(dx[0])

    return center_y

def line_fit_reshape():
    pass



def find_line(point, paint_address, Saving_fitting_img):
    nonzeroy = point[:, 1]
    #nonzeroy = point[:, 1] * (-1)
    nonzerox = point[:, 0]

    n, bins, patches = plt.hist(nonzeroy, 20, (-4, 0))
    bins_count = np.argmax(n)
    origin_right = 0.5 * (-4 + (4 / 20) * bins_count + (-4 + (4 / 20) * (bins_count + 1)))

    n, bins, patches = plt.hist(nonzeroy, 20, (0, 4))
    bins_count = np.argmax(n)
    origin_left =  0.5 * ((4 / 20) * bins_count + ((4 / 20) * (bins_count + 1)))

    lefty_current = origin_left
    righty_current = origin_right

    plt.close()

    window_height = 2
    fit_distence = 41
    start_distence = -1
    margin = 0.8
    minpix = 2
    nwindows = np.int((fit_distence - start_distence) / window_height)

    leftx = []
    lefty = []
    rightx = []
    righty = []
    lefty_center = []
    righty_center = []
    dx_center = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_x_low = window * window_height + start_distence
        win_x_high = (window + 1) * window_height + start_distence
        dx = win_x_low + window_height/2
        win_lefty_low = lefty_current - margin
        win_lefty_high = lefty_current + margin
        win_righty_low = righty_current - margin
        win_righty_high = righty_current + margin
        left_lane_inds = []
        right_lane_inds = []
        win_center_L = []
        win_center_R = []
        dx_center.append(dx)

        # Identify the nonzero pixels in x and y within the window
        for i in range(len(nonzerox)):
            if nonzerox[i] >= win_x_low and nonzerox[i] < win_x_high and \
                    nonzeroy[i] >= win_lefty_low and nonzeroy[i] < win_lefty_high:
                left_lane_inds.append(i)
                leftx.append(nonzerox[i])
                lefty.append(nonzeroy[i])
                win_center_L.append(nonzeroy[i])

            elif nonzerox[i] >= win_x_low and nonzerox[i] < win_x_high and \
                    nonzeroy[i] >= win_righty_low and nonzeroy[i] < win_righty_high:
                right_lane_inds.append(i)
                rightx.append(nonzerox[i])
                righty.append(nonzeroy[i])
                win_center_R.append(nonzeroy[i])

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_lane_inds) > minpix:
            lefty_current = np.mean(win_center_L)
        else:
            pass
        lefty_center.append(lefty_current)
        if len(right_lane_inds) > minpix:
            righty_current = np.mean(win_center_R)
        else:
            pass
        righty_center.append(righty_current)

    dx = np.array(dx_center)
    size = dx.size

    if (np.array(righty)).size == 0 and (np.array(lefty)).size == 0:
        righty_center = np.full((1, size), np.nan)
        lefty_center = np.full((1, size), np.nan)
    elif (np.array(righty)).size == 0 and (np.array(lefty)).size != 0:
        righty_center = np.full((1, size), np.nan)
        left_fit = np.polyfit(leftx, lefty, 2)
        lefty_center = center_exchange(np.array(lefty_center), left_fit, dx)
    elif (np.array(righty)).size != 0 and (np.array(lefty)).size == 0:
        lefty_center = np.full((1, size), np.nan)
        right_fit = np.polyfit(rightx, righty, 2)
        righty_center = center_exchange(np.array(righty_center),right_fit,dx)
    else :
        left_fit = np.polyfit(leftx, lefty, 2)
        lefty_center = center_exchange(np.array(lefty_center), left_fit, dx)
        right_fit = np.polyfit(rightx, righty, 2)
        righty_center = center_exchange(np.array(righty_center), right_fit, dx)


    if Saving_fitting_img == True:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # plt.scatter(lefty, leftx, s = 6,alpha=0.5, c='#A52A2A')
        # plt.scatter(righty, rightx, s = 6, alpha=0.5, c='#A52A2A')
        plt.scatter(nonzeroy, nonzerox, s=6, alpha=0.5, c='#A52A2A')

        # rect0 = plt.Rectangle(((lefty_center[0] - margin), start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect0)
        # rect1 = plt.Rectangle(((lefty_center[1] - margin), window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect1)
        # rect2 = plt.Rectangle(((lefty_center[2] - margin), 2 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect2 = plt.Rectangle(((lefty_center[3] - margin), 3 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect2 = plt.Rectangle(((lefty_center[4] - margin), 4 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect2 = plt.Rectangle(((lefty_center[5] - margin), 5 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect2 = plt.Rectangle(((lefty_center[6] - margin), 6 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect2 = plt.Rectangle(((lefty_center[7] - margin), 7 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect2 = plt.Rectangle(((lefty_center[8] - margin), 8 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect2)
        # rect3 = plt.Rectangle(((lefty_center[9] - margin), 9 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect3)
        # rect3 = plt.Rectangle(((lefty_center[10] - margin), 10 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect3)
        # rect3 = plt.Rectangle(((lefty_center[11] - margin), 11 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect3)
        # rect3 = plt.Rectangle(((lefty_center[12] - margin), 12 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect3)
        # rect3 = plt.Rectangle(((lefty_center[13] - margin), 13 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect3)
        # rect4 = plt.Rectangle(((righty_center[13] - margin), 13 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[12] - margin), 12 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[11] - margin), 11 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[10] - margin), 10 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[9] - margin), 9 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[8] - margin), 8 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[7] - margin), 7 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[6] - margin), 6 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[5] - margin), 5 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[4] - margin), 4 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[3] - margin), 3 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[2] - margin), 2 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[1] - margin), 1 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)
        # rect4 = plt.Rectangle(((righty_center[0] - margin), 0 * window_height + start_distence), 2 * margin, window_height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect4)

        #plt.plot(left_fited,x)
        #plt.plot(right_fited,x)

        plt.scatter(lefty_center, dx, s=10, alpha=0.8, c='b',marker='x')
        plt.scatter(righty_center, dx, s=10, alpha=0.8, c='b',marker='x')
        # plt.grid()
        plt.xlabel('m')
        plt.ylabel('m')
        plt.axis([-8,8,-10, 60])
        # plt.legend()
        plt.title('Ego line cluster and fitting')
        plt.savefig(paint_address)
        # plt.show()

    return lefty_center, righty_center, dx, np.array(leftx), np.array(lefty), np.array(rightx), np.array(righty)

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

def calculate_center(angle, posego, D_l, D_r):
    if angle >= 0:
        center_leftx = posego[0] - D_l * np.sin(angle)
        center_lefty = posego[1] + D_l * np.cos(angle)
        center_rightx = posego[0] + D_r * np.sin(angle)
        center_righty = posego[1] - D_r * np.cos(angle)
    else:
        center_leftx = posego[0] + D_l * np.sin(angle)
        center_lefty = posego[1] - D_l * np.cos(angle)
        center_rightx = posego[0] - D_r * np.sin(angle)
        center_righty = posego[1] + D_r * np.cos(angle)

    return center_leftx,center_lefty,center_rightx,center_righty


def find_line_full (point_list, posego, angle):
    point = np.array(point_list)
    nonzeroy = point[:, 1]
    nonzerox = point[:, 0]

    left_current = [0, 1.5]
    right_current = [0, -1.5]
    radius = 1
    minpix = 5

    # leftx = []
    # lefty = []
    # rightx = []
    # righty = []
    left_center = []
    right_center = []

    for j in range(posego[:,0]):
        left_lane_inds = []
        right_lane_inds = []
        win_center_L = []
        win_center_R = []

        #center_leftx, center_lefty, center_rightx, center_righty = calculate_center(angle[j], posego[j,:], D_l, D_r)

        for i in range(len(nonzerox)):
            distence_center_left = np.sqrt(np.square(nonzerox[i]-left_current[0])+
                                           np.square(nonzeroy[i]-left_current[1]))
            distence_center_right = np.sqrt(np.square(nonzerox[i]-right_current[0])+
                                            np.square(nonzeroy[i]-right_current[1]))
            if distence_center_left <= radius:
                left_lane_inds.append(i)
                # leftx.append(nonzerox[i])
                # lefty.append(nonzeroy[i])
                win_center_L.append(nonzeroy[i])
            if distence_center_right <= radius:
                right_lane_inds.append(i)
                # rightx.append(nonzerox[i])
                # righty.append(nonzeroy[i])
                win_center_R.append(nonzeroy[i])

        if len(left_lane_inds) > minpix:
            lefty_current = np.mean(win_center_L)
            leftx_current = posego[j,0]-np.tan(angle[j])(lefty_current-posego[j,1])
            left_current = np.array(leftx_current, lefty_current)
        else:
            pass
        left_center.append(left_current)

        if len(right_lane_inds) > minpix:
            righty_current = np.mean(win_center_R)
            rightx_current = posego[j, 0] - np.tan(angle[j])(righty_current - posego[j, 1])
            right_current = np.array(rightx_current,righty_current)
        else:
            pass
        right_center.append(right_current)
    return np.vstack(left_center), np.vstack(right_center)