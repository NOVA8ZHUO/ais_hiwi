#!/usr/bin/env python3

import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient           
from moveit_msgs.action import MoveGroup        
from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand  
from moveit_msgs.msg import MoveItErrorCodes, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive       

class VisionTestCommander(Node):
    def __init__(self):
        super().__init__('vision_test_commander')
        
        # 1. 初始化缓存字典和订阅器字典
        self.global_pose_buffer = {}
        self.local_pose_buffer = {}
        
        self.global_subs = {}  # 存储不同零件的 global 订阅器
        self.local_subs = {}   # 存储不同零件的 local 订阅器
        
        # 2. 控制参数
        self.CAMERA_HAND_OFFSET = 0.05     # 视觉坐标到机械臂末端的偏移
        self.LOCAL_OB_HIGHT = 0.4           # 局部观测的悬停高度

        self._arm_client = ActionClient(self, MoveGroup, 'move_action')
        self._gripper_client = ActionClient(self, GripperCommand, '/panda_gripper/gripper_action') 

        self.get_logger().info("等待 MoveIt action /move_action ...")
        if not self._arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                "10 秒内未发现 /move_action。请先启动 Franka MoveIt，并确认 ROS_DOMAIN_ID 一致。"
            )

        # Direct-pose movement does not use the gripper. Do not block startup
        # when the optional gripper action is absent or has a different name.
        if self._gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().info("检测到可选夹爪 action /panda_gripper/gripper_action")
        else:
            self.get_logger().warning(
                "未检测到可选夹爪 action；本次直接位姿移动不使用夹爪，可继续。"
            )

        # Direct-pose test parameters. Override them with --ros-args -p ...
        # so recorded vision coordinates can be tested without editing source.
        direct_pose_defaults = {
            "target_x": 0.357124617,
            "target_y": 0.000100923,
            "target_z": 0.696015951,
            "target_q_x": 1.0,
            "target_q_y": 0.0,
            "target_q_z": 0.0,
            "target_q_w": 0.0,
            "target_frame": "panda_link0",
            "use_linear": False,
            "velocity_scale": 0.05,
            "acceleration_scale": 0.02,
        }
        for name, value in direct_pose_defaults.items():
            self.declare_parameter(name, value)

        self.get_logger().info("✅ 纯控制端测试节点已启动，等待视觉坐标注入...")

    def _moveit_error_name(self, code: int) -> str:
        names = {
            MoveItErrorCodes.SUCCESS: "SUCCESS",
            MoveItErrorCodes.FAILURE: "FAILURE",
            MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
            MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
            MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            MoveItErrorCodes.CONTROL_FAILED: "CONTROL_FAILED",
            MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
            MoveItErrorCodes.PREEMPTED: "PREEMPTED",
            MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
            MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
            MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
            MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "GOAL_CONSTRAINTS_VIOLATED",
            MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
            MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "INVALID_GOAL_CONSTRAINTS",
            MoveItErrorCodes.INVALID_ROBOT_STATE: "INVALID_ROBOT_STATE",
            MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
            MoveItErrorCodes.INVALID_OBJECT_NAME: "INVALID_OBJECT_NAME",
            MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: "FRAME_TRANSFORM_FAILURE",
            MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: "COLLISION_CHECKING_UNAVAILABLE",
            MoveItErrorCodes.ROBOT_STATE_STALE: "ROBOT_STATE_STALE",
            MoveItErrorCodes.SENSOR_INFO_STALE: "SENSOR_INFO_STALE",
            MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
        }
        return names.get(code, f"UNKNOWN({code})")

    def _log_target_pose_debug(self, x, y, z, q_w, q_x, q_y, q_z, frame_id, group_name, ee_link):
        q_norm = math.sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z)
        self.get_logger().info("========== MoveIt target pose debug ==========")
        self.get_logger().info(f"target frame: {frame_id}")
        self.get_logger().info(f"target position: x={x:.6f}, y={y:.6f}, z={z:.6f}")
        self.get_logger().info(
            f"target quaternion ROS xyzw: x={q_x:.6f}, y={q_y:.6f}, z={q_z:.6f}, w={q_w:.6f}"
        )
        self.get_logger().info(f"quaternion norm: {q_norm:.9f}")
        self.get_logger().info(f"planning group: {group_name}")
        self.get_logger().info(f"end-effector link: {ee_link}")
        self.get_logger().info("planning frame: action request constraint frame (no MoveGroupCommander current-state API in this node)")
        self.get_logger().info("current robot pose / target distance: unavailable in this action-only client")
        if abs(q_norm - 1.0) > 1e-3:
            self.get_logger().warning("WARNING: target quaternion norm is not close to 1.")
        if abs(q_x - 1.0) < 1e-9 and abs(q_y) < 1e-9 and abs(q_z) < 1e-9 and abs(q_w) < 1e-9:
            self.get_logger().info(
                "Using the workflow's fixed tool-down quaternion [1,0,0,0] in ROS xyzw."
            )
        if abs(q_w - 1.0) < 1e-9 and abs(q_x) < 1e-9 and abs(q_y) < 1e-9 and abs(q_z) < 1e-9:
            self.get_logger().warning(
                "WARNING: identity quaternion [0,0,0,1] in ROS xyzw may not be the intended top-down observation pose."
            )
        self.get_logger().info("==============================================")

    # ================= 动态订阅与回调 =================
    def send_gripper_goal(self, position):
        """发送夹爪指令并等待执行完成 (同步阻塞模式)"""
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        send_goal_future = self._gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('夹爪请求被拒绝')
            return False
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        return True
    
    def _ensure_subscriptions(self, part_name: str):
        """根据零件名称动态创建专属的话题订阅"""
        if part_name not in self.global_subs:
            global_topic = f'/vision/global/{part_name}/pose'
            # 使用 lambda 将 part_name 作为额外参数传递给回调函数
            self.global_subs[part_name] = self.create_subscription(
                PoseStamped, 
                global_topic, 
                lambda msg, p=part_name: self.global_vision_cb(msg, p), 
                10
            )
            self.get_logger().info(f"🔗 已动态订阅 Global 话题: {global_topic}")

        if part_name not in self.local_subs:
            local_topic = f'/vision/local/{part_name}/pose'
            self.local_subs[part_name] = self.create_subscription(
                PoseStamped, 
                local_topic, 
                lambda msg, p=part_name: self.local_vision_cb(msg, p), 
                10
            )
            self.get_logger().info(f"🔗 已动态订阅 Local 话题: {local_topic}")

    def global_vision_cb(self, msg: PoseStamped, part_name: str):
        """全局视觉回调，此时 frame_id 是基座标系 (如 panda_link0)"""
        self.global_pose_buffer[part_name] = msg
        self.get_logger().info(f"📍 收到 Global -位姿 | 零件: {part_name} | 基准系: {msg.header.frame_id}")

    def local_vision_cb(self, msg: PoseStamped, part_name: str):
        """局部视觉回调"""
        self.local_pose_buffer[part_name] = msg
        self.get_logger().info(f"🎯 收到 Local 精准位姿 | 零件: {part_name} | 基准系: {msg.header.frame_id}")

    def wait_for_global_data(self, target_part_name: str) -> PoseStamped:
        """阻塞等待直到获取指定零件的全局位姿"""
        self._ensure_subscriptions(target_part_name)
        
        # 清除旧缓存，确保拿到的不是历史数据
        self.global_pose_buffer.pop(target_part_name, None) 
        self.get_logger().info(f"⏳ 等待 {target_part_name} 的全局位姿 /vision/global/{target_part_name}/pose ...")
        
        while target_part_name not in self.global_pose_buffer and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
        return self.global_pose_buffer[target_part_name]

    def wait_for_local_data(self, target_part_name: str) -> PoseStamped:
        """阻塞等待直到获取指定零件的局部位姿"""
        self._ensure_subscriptions(target_part_name)
        
        self.local_pose_buffer.pop(target_part_name, None)
        self.get_logger().info(f"⏳ 等待 {target_part_name} 的局部精准位姿 /vision/local/{target_part_name}/pose ...")
        
        while target_part_name not in self.local_pose_buffer and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        pose = self.local_pose_buffer[target_part_name].pose
        self.get_logger().info(
            f"✅ 已获取 {target_part_name} 的局部位姿 "
            f"x:{pose.position.x}, y:{pose.position.y}, z:{pose.position.z}, "
            f"w:{pose.orientation.w}, x:{pose.orientation.x}, y:{pose.orientation.y}, z:{pose.orientation.z}，准备移动..."
        )
        return self.local_pose_buffer[target_part_name]

    # ================= 机械臂运动控制 Mock 接口 =================

    def move_to_pose(self, x, y, z, q_w=0.0, q_x=1.0, q_y=0.0, q_z=0.0, use_linear=True, vel=0.2, acc=0.05, max_retries=99, target_frame="world"):
        """笛卡尔空间姿态移动"""
        goal_msg = MoveGroup.Goal()                             
        group_name = "panda_arm"
        ee_link = "panda_hand"
        goal_msg.request.group_name = group_name
        goal_msg.request.num_planning_attempts = 50
        goal_msg.request.allowed_planning_time = 10.0 

        if use_linear:
            goal_msg.request.pipeline_id = "pilz_industrial_motion_planner"
            goal_msg.request.planner_id = "LIN"
            goal_msg.request.max_velocity_scaling_factor = vel
            goal_msg.request.max_acceleration_scaling_factor = acc
        else:
            goal_msg.request.pipeline_id = "ompl"
            goal_msg.request.planner_id = "RRTConnectkConfigDefault"
            goal_msg.request.max_velocity_scaling_factor = vel
            goal_msg.request.max_acceleration_scaling_factor = acc

        # 位置约束
        c1 = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = target_frame # 注意：如果真实环境中你的 base_frame 不是 world，请将此处改为适配的基座标系 (如 panda_link0)
        pc.link_name = ee_link
        
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.002]
        bv.primitives.append(sp)
        
        target_pose = PoseStamped()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        bv.primitive_poses.append(target_pose.pose)
        pc.constraint_region = bv
        pc.weight = 1.0 
        c1.position_constraints.append(pc)

        # 姿态约束
        oc = OrientationConstraint()
        oc.header.frame_id = target_frame # 同上
        oc.link_name = ee_link
        oc.orientation.w = q_w
        oc.orientation.x = q_x
        oc.orientation.y = q_y
        oc.orientation.z = q_z
        oc.absolute_x_axis_tolerance = 0.03
        oc.absolute_y_axis_tolerance = 0.03
        oc.absolute_z_axis_tolerance = 0.03
        oc.weight = 1.0 
        c1.orientation_constraints.append(oc)

        goal_msg.request.goal_constraints.append(c1) 
        self._log_target_pose_debug(x, y, z, q_w, q_x, q_y, q_z, target_frame, group_name, ee_link)
        
        attempt = 0
        while rclpy.ok() and attempt < max_retries:
            attempt += 1
            send_goal_future = self._arm_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error(f'第 {attempt} 次请求被 MoveGroup 拒绝。')
                time.sleep(1.0)
                continue 

            get_result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, get_result_future)
            action_result = get_result_future.result()
            error_code = action_result.result.error_code.val

            if error_code == MoveItErrorCodes.SUCCESS:
                return True
            else:
                self.get_logger().error(
                    f'第 {attempt} 次移动失败，错误码: {error_code} ({self._moveit_error_name(error_code)})'
                )
                time.sleep(1.0)
                
        return False

    def safe_test_mode_from_reference_pose(self, reference_pose: PoseStamped, dx=0.01, dy=0.01, dz=0.02):
        """Debug-only safe test: preserve a known reachable orientation and move by a tiny offset."""
        pose = reference_pose.pose
        self.get_logger().info("Running safe test mode from provided reference pose.")
        return self.move_to_pose(
            pose.position.x + dx,
            pose.position.y + dy,
            pose.position.z + dz,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            use_linear=False,
            vel=0.1,
            acc=0.05,
            max_retries=1,
            target_frame=reference_pose.header.frame_id or "world",
        )

    # ================= 核心状态机流程 =================
    
    def run_assembly_test(self, target_part_name="Gear_large_handle"):
        # Accept notebook-style aliases such as Gear_large, but always subscribe
        # to the canonical force-control topic name Gear_large_handle.
        target_part_name = force_control_part_name(target_part_name)
        
        self.send_gripper_goal(position=0.08)  # 
        # STATE 0: 移至全局高位视角
        self.get_logger().info("\n--- STATE 0: 移至全局拍照点 ---")
        self.move_to_pose(x=0.4, y=0.0, z=0.6) # 默认垂直向下
        
        # STATE 1: 等待全局视觉输出粗略 XYZ
        global_pose_msg = self.wait_for_global_data(target_part_name)
        
        g_x = global_pose_msg.pose.position.x + self.CAMERA_HAND_OFFSET
        g_y = global_pose_msg.pose.position.y
        g_z = self.LOCAL_OB_HIGHT

        self.get_logger().info(f"✅ 找到 {target_part_name} 的local检测坐标 XYZ: {g_x:.3f}, {g_y:.3f}, {g_z:.3f}")
        
        # STATE 2: 利用全局 XYZ，保持默认向下姿态，平移到目标正上方悬停
        self.get_logger().info("\n--- STATE 2: 移至local观测上方悬停 ---")
        self.move_to_pose(g_x, g_y, g_z)

        # STATE 3: 到达悬停点后，等待局部视觉输出精准的 XYZ + Orientation
        local_pose_msg = self.wait_for_local_data(target_part_name)

        l_x = local_pose_msg.pose.position.x
        l_y = local_pose_msg.pose.position.y
        l_z = self.LOCAL_OB_HIGHT
        q_x = local_pose_msg.pose.orientation.x
        q_y = local_pose_msg.pose.orientation.y
        q_z = local_pose_msg.pose.orientation.z
        q_w = local_pose_msg.pose.orientation.w

        # STATE 4: 移动到局部视觉输出的位姿上方，验证姿态是否正确
        self.get_logger().info("\n--- STATE 4: 验证精准抓取姿态 ---")
        self.move_to_pose(l_x, l_y, l_z, q_w, q_x, q_y, q_z)        

        # STATE 5: 最后下降到目标点，验证 XYZ 精度
        self.get_logger().info("\n--- STATE 5: 验证精准抓取位置 ---")
        verify_z = 0.2
        self.move_to_pose(l_x, l_y, verify_z, q_w, q_x, q_y, q_z)
        self.send_gripper_goal(position=0.017)  # 紧闭夹爪
        self.get_logger().info("🎉 控制端流程跑通！机械臂已到达对准点上方，请确认姿态是否正确。")

        self.move_to_pose(l_x, l_y, l_z, q_w, q_x, q_y, q_z)  # 抓取后抬起到悬停高度

def main(args=None):
    rclpy.init(args=args)
    commander = VisionTestCommander()
    
    # Use a safe hover Z for vision-coordinate checks. A Local PoseStamped
    # quaternion is planar object yaw, not a complete panda_hand orientation;
    # keep the default tool-down quaternion unless it has been converted first.
    target_position = [
        float(commander.get_parameter("target_x").value),
        float(commander.get_parameter("target_y").value),
        float(commander.get_parameter("target_z").value),
    ]
    target_quat_xyzw = [
        float(commander.get_parameter("target_q_x").value),
        float(commander.get_parameter("target_q_y").value),
        float(commander.get_parameter("target_q_z").value),
        float(commander.get_parameter("target_q_w").value),
    ]
    target_frame = str(commander.get_parameter("target_frame").value)
    use_linear = bool(commander.get_parameter("use_linear").value)
    velocity_scale = float(
        commander.get_parameter("velocity_scale").value
    )
    acceleration_scale = float(
        commander.get_parameter("acceleration_scale").value
    )


    try:
        # 传入的字符串会被动态拼装进 Topic 中： /vision/global/Shaft_large/pose
        commander.get_logger().info(
            f"move_to_link0_pose, 目标位置: {target_position}, 姿态 ROS xyzw: {target_quat_xyzw}, frame: {target_frame}"
        )
        moved = commander.move_to_pose(
            target_position[0],
            target_position[1],
            target_position[2],
            q_w=target_quat_xyzw[3],
            q_x=target_quat_xyzw[0],
            q_y=target_quat_xyzw[1],
            q_z=target_quat_xyzw[2],
            use_linear=use_linear,
            vel=velocity_scale,
            acc=acceleration_scale,
            max_retries=1,
            target_frame=target_frame,
        )
        if moved:
            commander.get_logger().info("目标运动由 MoveIt 报告为 SUCCESS。")
        else:
            commander.get_logger().error("目标运动未成功，机械臂未被确认移动到目标位置。")
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()





# python3 move_to_link0_pose.py
