#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient           
from moveit_msgs.action import MoveGroup        
from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand  
from moveit_msgs.msg import MoveItErrorCodes, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive       
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from vision_common import force_control_part_name

'''
测试从视觉系统直接输出的坐标进行机械臂控制
'''

BASE_HOLE_BY_INSERT_PART = {
    "Shaft_small": "hole_small",
    "Shaft_large": "hole_large",
    "Screw": "hole_screw",
}

# Existing force_control placement tool heights above the detected base plane.
ASSEMBLY_TOOL_HEIGHT_BY_PART_M = {
    "Shaft_small": 0.17,
    "Shaft_large": 0.19,
    "Screw": 0.22,
}

class VisionTestCommander(Node):
    def __init__(self):
        super().__init__('vision_test_commander')
        
        # 1. 初始化缓存字典和订阅器字典
        self.global_pose_buffer = {}
        self.local_pose_buffer = {}
        
        self.global_subs = {}  # 存储不同零件的 global 订阅器
        self.local_subs = {}   # 存储不同零件的 local 订阅器
        
        # 2. 控制参数
        self.LOCAL_OB_HIGHT = 0.4           # 局部观测的悬停高度

        self.declare_parameter("local_vision_node", "/local_vision_force_adapter")
        self.declare_parameter("target_part", "Shaft_large")
        self.declare_parameter("execute_assembly_descent", False)
        local_node = str(self.get_parameter("local_vision_node").value).rstrip("/")
        self._local_parameter_client = self.create_client(
            SetParameters, f"{local_node}/set_parameters"
        )

        self._arm_client = ActionClient(self, MoveGroup, 'move_action')
        self._gripper_client = ActionClient(self, GripperCommand, '/panda_gripper/gripper_action') 

        self._arm_client.wait_for_server()
        self._gripper_client.wait_for_server()

        self.get_logger().info("✅ 纯控制端测试节点已启动，等待视觉坐标注入...")

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
        self.get_logger().info(f"📍 收到 Global 位姿 | 零件: {part_name} | 基准系: {msg.header.frame_id}")

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

    def wait_for_global_part_and_base(self, target_part_name: str):
        """Acquire target and Base from the same running Global stage."""
        names = (target_part_name, "Base")
        for name in names:
            self._ensure_subscriptions(name)
            self.global_pose_buffer.pop(name, None)
        self.get_logger().info(
            f"⏳ 等待 Global 同时提供 {target_part_name} 与 Base..."
        )
        while rclpy.ok() and any(
            name not in self.global_pose_buffer for name in names
        ):
            rclpy.spin_once(self, timeout_sec=0.1)
        return tuple(self.global_pose_buffer[name] for name in names)

    def set_local_detection_target(self, target_part: str, hole_target: str):
        """Switch Local detector and require the ROS parameter update to succeed."""
        if not self._local_parameter_client.wait_for_service(timeout_sec=20.0):
            raise RuntimeError(
                "Local vision parameter service is unavailable; start the Local stage"
            )
        request = SetParameters.Request()
        request.parameters = [
            Parameter("target_part", value=target_part).to_parameter_msg(),
            Parameter("base_hole_target", value=hole_target).to_parameter_msg(),
        ]
        future = self._local_parameter_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or len(response.results) != 2:
            raise RuntimeError("Local vision returned an incomplete parameter response")
        failures = [
            result.reason or "parameter rejected"
            for result in response.results
            if not result.successful
        ]
        if failures:
            raise RuntimeError("Local target switch failed: " + "; ".join(failures))
        self.local_pose_buffer.clear()
        self.get_logger().info(
            f"✅ Local 检测目标已切换: target_part={target_part}, "
            f"base_hole_target={hole_target}"
        )

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

    def move_to_pose(self, x, y, z, q_w=0.0, q_x=1.0, q_y=0.0, q_z=0.0, use_linear=True, vel=0.2, acc=0.05, max_retries=99):
        """笛卡尔空间姿态移动"""
        goal_msg = MoveGroup.Goal()                             
        goal_msg.request.group_name = "panda_arm"               
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
            goal_msg.request.max_velocity_scaling_factor = 0.5
            goal_msg.request.max_acceleration_scaling_factor = 0.5

        # 位置约束
        c1 = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = "world" # 注意：如果真实环境中你的 base_frame 不是 world，请将此处改为适配的基座标系 (如 panda_link0)
        pc.link_name = "panda_hand" 
        
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.001] 
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
        oc.header.frame_id = "world" # 同上
        oc.link_name = "panda_hand"
        oc.orientation.w = q_w
        oc.orientation.x = q_x
        oc.orientation.y = q_y
        oc.orientation.z = q_z
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0 
        c1.orientation_constraints.append(oc)

        goal_msg.request.goal_constraints.append(c1) 
        
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
                self.get_logger().error(f'第 {attempt} 次移动失败，错误码: {error_code}')
                time.sleep(1.0)
                
        return False

    # ================= 核心状态机流程 =================
    
    def run_assembly_test(self, target_part_name="Gear_large_handle"):
        # Accept notebook-style aliases such as Gear_large, but always subscribe
        # to the canonical force-control topic name Gear_large_handle.
        target_part_name = force_control_part_name(target_part_name)
        if target_part_name not in BASE_HOLE_BY_INSERT_PART:
            raise ValueError(
                "Base/hole Local 装配阶段只适用于 Shaft_small、Shaft_large、Screw；"
                "齿轮应以相应 Shaft 为装配目标，不能映射到 Base hole"
            )
        assembly_hole = BASE_HOLE_BY_INSERT_PART[target_part_name]
        
        self.send_gripper_goal(position=0.08)  # 
        # STATE 0: 移至全局高位视角
        self.get_logger().info("\n--- STATE 0: 移至全局拍照点 ---")
        self.move_to_pose(x=0.4, y=0.0, z=0.6) # 默认垂直向下
        
        # STATE 1: 等待全局视觉输出粗略 XYZ
        global_pose_msg, global_base_pose_msg = self.wait_for_global_part_and_base(
            target_part_name
        )
        
        # Global PoseStamped is already in panda_link0; never add a fixed
        # camera-to-hand offset here.
        g_x = global_pose_msg.pose.position.x
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

        # STATE 6: Use cached Global Base pose to bring the camera over the base.
        self.get_logger().info("\n--- STATE 6: 移至 Base 上方进行装配孔 Local 检测 ---")
        base_x = global_base_pose_msg.pose.position.x
        base_y = global_base_pose_msg.pose.position.y
        self.move_to_pose(base_x, base_y, self.LOCAL_OB_HIGHT)

        # The same Local stage is switched at runtime. Its adapter clears the
        # old consensus and publishes only the requested semantic hole.
        self.set_local_detection_target("Base", assembly_hole)
        hole_pose_msg = self.wait_for_local_data(assembly_hole)
        hole_pose = hole_pose_msg.pose
        h_x = hole_pose.position.x
        h_y = hole_pose.position.y
        # Base-hole notebook output defines a position, not a grasp yaw. Keep
        # the already verified tool-down grasp orientation while carrying the
        # part; do not interpret the hole's legal identity quaternion as a TCP
        # assembly orientation.
        h_qx, h_qy, h_qz, h_qw = q_x, q_y, q_z, q_w

        # STATE 7: Always verify alignment above the hole first.
        self.get_logger().info(
            f"\n--- STATE 7: 对准装配孔 {assembly_hole} 上方 ---"
        )
        self.move_to_pose(
            h_x, h_y, self.LOCAL_OB_HIGHT, h_qw, h_qx, h_qy, h_qz
        )

        # Physical descent is explicitly gated. The target heights are copied
        # from force_control.py; no new insertion offset is invented here.
        if bool(self.get_parameter("execute_assembly_descent").value):
            target_tool_z = (
                hole_pose.position.z
                + ASSEMBLY_TOOL_HEIGHT_BY_PART_M[target_part_name]
            )
            self.get_logger().warning(
                f"执行装配下降: target_tool_z={target_tool_z:.3f} m"
            )
            self.move_to_pose(
                h_x, h_y, target_tool_z, h_qw, h_qx, h_qy, h_qz,
                use_linear=True, vel=0.10, acc=0.03,
            )
            self.send_gripper_goal(position=0.08)
            self.move_to_pose(
                h_x, h_y, self.LOCAL_OB_HIGHT, h_qw, h_qx, h_qy, h_qz
            )
            self.get_logger().info("🎉 Base/hole Local 装配阶段完成")
        else:
            self.get_logger().warning(
                "已完成装配孔视觉对准，但 execute_assembly_descent=false，"
                "未执行物理下降和释放"
            )

def main(args=None):
    rclpy.init(args=args)
    commander = VisionTestCommander()
        
    try:
        # The name is normalized before topic construction.
        commander.run_assembly_test(
            target_part_name=str(commander.get_parameter("target_part").value)
        )
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
