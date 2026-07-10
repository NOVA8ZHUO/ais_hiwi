#!/usr/bin/env python3

import numpy as np

def calculate_absolute_pose(cam_x, cam_y, cam_z):
    # 1. 目标物体在相机坐标系下的位置 P_camera
    P_camera = np.array([cam_x, cam_y, cam_z, 1.0])

    # 2. 手眼标定矩阵 T_hand_camera
    T_hand_camera = np.array([
    [-0.036813668,  0.999304948, -0.005863054, -0.045601445],
    [-0.999304818, -0.036777844,  0.006105050,  0.037541696],
    [ 0.005885176,  0.006083727,  0.999964176,  0.058262175],
    [ 0.000000000,  0.000000000,  0.000000000,  1.000000000],
])

    # 3. 当前机械臂末端的位姿矩阵 T_link0_hand
    T_link0_hand = np.array([
        [0.994464253, -0.101513246, 0.027127686, 0.351752559],
        [-0.100399912, -0.994156352, -0.039661127, -0.019353122],
        [0.030995291, 0.036717956, -0.998844875, 0.624762609],
        [0.000000000, 0.000000000, 0.000000000, 1.000000000],
    ])

    # --- 计算开始 ---
    P_hand = T_hand_camera @ P_camera
    P_link0 = T_link0_hand @ P_hand
    
    return P_hand, P_link0

def main():
    # 🌟 在这里填入你新测试的相机坐标
    new_cam_x = 0.465287		
    new_cam_y = -0.044779	
    new_cam_z = 0.561499
    
    print(f"📐 输入的相机坐标: [{new_cam_x}, {new_cam_y}, {new_cam_z}]")

    P_hand, P_link0 = calculate_absolute_pose(new_cam_x, new_cam_y, new_cam_z)

    print("--------------------------------------------------")
    print("步骤 A: 物体在夹爪 (panda_hand) 坐标系下的相对位置:")
    print(f"X = {P_hand[0]:.6f} m\nY = {P_hand[1]:.6f} m\nZ = {P_hand[2]:.6f} m")

    print("--------------------------------------------------")
    print("步骤 B (最终结果): 物体在机器人基座 (panda_link0) 坐标系下的绝对位置:")
    print(f"X = {P_link0[0]:.6f} m\nY = {P_link0[1]:.6f} m\nZ = {P_link0[2]:.6f} m")
    print("--------------------------------------------------")

if __name__ == '__main__':
    main()
