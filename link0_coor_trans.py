#!/usr/bin/env python3

import numpy as np

def calculate_absolute_pose(cam_x, cam_y, cam_z):
    # 1. 目标物体在相机坐标系下的位置 P_camera
    P_camera = np.array([cam_x, cam_y, cam_z, 1.0])

    # 2. 手眼标定矩阵 T_hand_camera
    T_hand_camera = np.array([
        [-0.00141984,  0.99994280, -0.01060082, -0.05150568],
        [-0.99998387, -0.00147804, -0.00548470,  0.02971965],
        [-0.00550006,  0.01059286,  0.99992877,  0.05996807],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
    ])

    # 3. 当前机械臂末端的位姿矩阵 T_link0_hand
    T_link0_hand = np.array([
        [0.994484450, -0.104394350, -0.010124177, 0.430363940],
        [-0.104163017, -0.994334701, 0.021179428, 0.042528753],
        [-0.012277833, -0.020008047, -0.999724428, 0.489648672],
        [0.000000000, 0.000000000, 0.000000000, 1.000000000],
    ])

    # --- 计算开始 ---
    P_hand = T_hand_camera @ P_camera
    P_link0 = T_link0_hand @ P_hand
    
    return P_hand, P_link0

def main():
    # 🌟 在这里填入你新测试的相机坐标
    new_cam_x = -0.019656
    new_cam_y = 0.073504
    new_cam_z = 0.427806
    
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