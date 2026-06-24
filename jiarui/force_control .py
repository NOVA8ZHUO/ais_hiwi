#!/usr/bin/env python3  

import time
import rclpy                                    # 导入ROS 2的Python客户端库.
import math
import random

from rclpy.node import Node                     # 导入ROS 2的节点基类.
from rclpy.action import ActionClient           # 导入ROS 2的动作客户端.
from moveit_msgs.action import MoveGroup        # 导入MoveIt 2的动作客户端.
from geometry_msgs.msg import PoseStamped, WrenchStamped, TwistStamped # 🌟 新增 Wrench 和 Twist
from control_msgs.action import GripperCommand  # 导入控制消息类型.
from tf2_ros import TransformException, Buffer, TransformListener # 导入TF2的缓冲区和监听器.
from moveit_msgs.msg import MoveItErrorCodes, Constraints, PositionConstraint, OrientationConstraint, JointConstraint, BoundingVolume, CollisionObject 
from shape_msgs.msg import SolidPrimitive       
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger  # 🌟 新增：用于调用 Servo 的启动/暂停服务
from franka_msgs.msg import FrankaState  # 🌟 新增：用于真机状态解析
from controller_manager_msgs.srv import SwitchController

# =========================================================================
# 控制开关与默认值
# =========================================================================
CONTACT_FORCE_THRESHOLD = 3.0 # Z轴碰撞阈值 (牛顿)，受力超过此值即判定为发生碰撞/卡死

class AssemblyCommander(Node):                   
    def __init__(self):
        super().__init__('assembly_commander')  
        
        # ---------------------------------------------------------
        # 1. 声明参数与初始化配置
        # ---------------------------------------------------------
        self.declare_parameter('use_sim', True)
        self.declare_parameter('use_velocity_control', False)
        self.declare_parameter('namespace', '')
        
        self.use_sim = self.get_parameter('use_sim').get_parameter_value().bool_value
        self.use_velocity_control = self.get_parameter('use_velocity_control').get_parameter_value().bool_value
        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        
        # 确保 namespace 格式正确
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        self.get_logger().info(
            f'启动装配控制器:\n'
            f'  仿真模式 (use_sim): {self.use_sim}\n'
            f'  速度控制 (use_velocity_control): {self.use_velocity_control}\n'
            f'  命名空间 (namespace): {self.ns}'
        )

        # ---------------------------------------------------------
        # 2. 动态设置话题和服务名称
        # ---------------------------------------------------------
        if self.use_sim:
            self.wrench_topic = self.ns + '/isaac_joint_wrench'
            self.gripper_action = self.ns + '/panda_hand_controller/gripper_cmd'
            self.servo_twist_topic = self.ns + '/servo_node/delta_twist_cmds'

        else:
            # 真实机器人环境下，根据 franka_ros2 规范调整
            self.wrench_topic = self.ns + '/franka/external_wrench' 
            self.gripper_action = self.ns + '/panda_gripper/gripper_action'
            # 🌟 直接向底层速度控制器发指令（请根据你底层 C++ 实际监听的话题名修改）
            self.servo_twist_topic = self.ns + '/cartesian_velocity_example_controller/commands'

        self.joint_states_topic = self.ns + '/joint_states'
        self.move_group_action = self.ns + '/move_action'
        
        self.start_servo_srv = self.ns + '/servo_node/start_servo'
        self.pause_servo_srv = self.ns + '/servo_node/pause_servo'

        # ---------------------------------------------------------
        # 3. TF 监听
        # ---------------------------------------------------------
        self.tf_buffer = Buffer()                                   
        self.tf_listener = TransformListener(self.tf_buffer, self)  
        
        # ---------------------------------------------------------
        # 4. Collision 监听
        # ---------------------------------------------------------
        self._scene_pub = self.create_publisher(CollisionObject, self.ns + '/collision_object', 10)

        # ---------------------------------------------------------
        # 5. Joint State 监听
        # ---------------------------------------------------------
        self._current_joints = None
        self._joint_state_sub = self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)

        # ---------------------------------------------------------
        # 6. Wrench (六轴力矩) 监听
        # ---------------------------------------------------------
        self._current_wrench = None

        if self.use_sim:
            self._wrench_sub = self.create_subscription(WrenchStamped, self.wrench_topic, self.wrench_callback, 10)
        else:
            # Real robot： 订阅 FrankaState
            self.wrench_topic = self.ns + '/franka_robot_state_broadcaster/robot_state'
            self._wrench_sub = self.create_subscription(FrankaState, self.wrench_topic, self.wrench_callback_real, 10)

        
        # ---------------------------------------------------------
        # 7. Twist (笛卡尔速度) 发布
        # ---------------------------------------------------------
        self._twist_pub = self.create_publisher(TwistStamped, self.servo_twist_topic, 10)

        # ---------------------------------------------------------
        # 8. Action & Service 客户端
        # ---------------------------------------------------------
        self._arm_client = ActionClient(self, MoveGroup, self.move_group_action) 
        self._gripper_client = ActionClient(self, GripperCommand, self.gripper_action) 

        self._switch_controller_client = self.create_client(SwitchController, self.ns + '/controller_manager/switch_controller')

        self._start_servo_client = self.create_client(Trigger, self.start_servo_srv)
        self._pause_servo_client = self.create_client(Trigger, self.pause_servo_srv)

        # 等待服务器
        self.get_logger().info('正在等待 MoveIt 和 Gripper 服务...')
        self._arm_client.wait_for_server()    
        self._gripper_client.wait_for_server()
        self.get_logger().info('所有系统就绪，准备开始任务！')


    # =========================================================================
    # 回调函数区
    # =========================================================================

    def joint_state_callback(self, msg):
        """实时更新关节角度数据，支持 panda_joint 和 fr3_joint"""
        joint_names = ['panda_joint1', 'fr3_joint1']
        for name in joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                self._current_joints = list(msg.position[idx : idx+7])
                break
            
    def wrench_callback(self, msg):
        """🌟 实时更新末端受力数据"""
        self._current_wrench = msg.wrench

    def wrench_callback_real(self, msg):
        """🌟 实时更新末端受力数据 (真机模式使用)，并保留完整的 Stamped 接口"""
        # 1. 构造标准的 WrenchStamped 消息
        stamped_msg = WrenchStamped()

        # 2. 填充 Header 信息
        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.header.frame_id = "panda_hand" 
        
        # 3. 填充 Wrench 信息 (务必使用 k_f_ext_hat_k 保持末端刚度坐标系对齐仿真)
        stamped_msg.wrench.force.x = msg.k_f_ext_hat_k[0]
        stamped_msg.wrench.force.y = msg.k_f_ext_hat_k[1]
        stamped_msg.wrench.force.z = msg.k_f_ext_hat_k[2]
        stamped_msg.wrench.torque.x = msg.k_f_ext_hat_k[3]
        stamped_msg.wrench.torque.y = msg.k_f_ext_hat_k[4]
        stamped_msg.wrench.torque.z = msg.k_f_ext_hat_k[5]
        
        # 4. 赋值给当前受力状态 
        self._current_wrench = stamped_msg.wrench
        
        # 💡 【预留接口】：
        # 如果你日后需要把真机的力矩数据像仿真一样 publish 出去录制 rosbag，
        # 只需要在 __init__ 里声明一个 self._real_wrench_pub，然后在这里执行：
        # self._real_wrench_pub.publish(stamped_msg)

    # =========================================================================
    # 场景与状态获取区
    # =========================================================================

    def add_table(self):
        """添加虚拟桌子到碰撞场景"""
        obj = CollisionObject()
        obj.header.frame_id = "world"
        obj.id = "table" 
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.5, 1.5, 0.1] 
        
        table_pose = PoseStamped().pose
        table_pose.position.x = 0.0
        table_pose.position.y = 0.0
        table_pose.position.z = -0.05 
        
        obj.primitives.append(box)
        obj.primitive_poses.append(table_pose)
        obj.operation = CollisionObject.ADD
        
        self._scene_pub.publish(obj)
        self.get_logger().info('虚拟桌子已添加到 MoveIt 场景！')
    
    def get_object_pose(self, target_part):
        """
        获取物体坐标。
        仿真模式下：直接从 TF 树中读取真实的坐标转换。
        真机模式下：使用预设的假坐标（mock poses）代替视觉输入。
        """
        if target_part == 'panda_hand' or self.use_sim:
            try:
                if not self.tf_buffer.can_transform('world', target_part, rclpy.time.Time()):
                    return None  
                trans = self.tf_buffer.lookup_transform('world', target_part, rclpy.time.Time())
                return trans.transform 
            except Exception as e:
                # 过滤掉频繁的异常打印
                return None
        else:
            # ---------------------------------------------------------
            # 真机模式：使用伪造的物体坐标 (参考 pick_place_node.py)
            # ---------------------------------------------------------
            from geometry_msgs.msg import Transform
            
            # 我们假设机器人基座在 (0,0,0)
            # 桌子在正前方 x = 0.3 到 0.6 的区域，y 是左右，z 是桌面高度（假设 z=0.01 是桌面上的零件中心）
            mock_poses = {
                "Base":                 {"x": 0.357, "y": -0.027, "z": 0.00}, # 组装基座，放在正中间
                "Shaft_small":          {"x": 0.509, "y": -0.123, "z": 0.00}, # 左边
                "Shaft_large":          {"x": 0.416, "y": -0.174, "z": 0.00}, # 右边
                "Screw":                {"x": 0.35,  "y": 0.15,  "z": 0.00},
                "Gear_small_handle":    {"x": 0.45,  "y": 0.20,  "z": 0.00},
                "Gear_compound_handle": {"x": 0.45,  "y": -0.20, "z": 0.00},
                "Gear_large_handle":    {"x": 0.50,  "y": 0.00,  "z": 0.00},
            }

            # 稍微模拟一下寻找时间
            time.sleep(0.1)

            if target_part not in mock_poses:
                self.get_logger().warn(f"[Mock视觉] 未知的零件名 {target_part}，返回 None")
                return None

            # 构造并填充 Transform 对象
            trans = Transform()
            trans.translation.x = mock_poses[target_part]["x"]
            trans.translation.y = mock_poses[target_part]["y"]
            trans.translation.z = mock_poses[target_part]["z"]
            
            # 默认假设所有零件刚开始都是“正放”在桌面上 (无倾斜，无旋转)
            trans.rotation.x = 0.0
            trans.rotation.y = 0.0
            trans.rotation.z = 0.0
            trans.rotation.w = 1.0
            
            return trans

    def get_downward_orientation(self, o):
        """将物体的旋转四元数转换为夹爪朝下的四元数"""
        q_w = -o.x
        q_x = o.w
        q_y = o.z
        q_z = -o.y
        return q_w, q_x, q_y, q_z
    
    def local_to_world(self, base_translation, base_rotation, local_offset):
        """将局部坐标系下的偏移量转换到世界坐标系"""
        w, x, y, z = base_rotation.w, base_rotation.x, base_rotation.y, base_rotation.z
        lx, ly, lz = local_offset[0], local_offset[1], local_offset[2]
        
        tx = 2.0 * (y * lz - z * ly)
        ty = 2.0 * (z * lx - x * lz)
        tz = 2.0 * (x * ly - y * lx)
        
        rx = lx + w * tx + (y * tz - z * ty)
        ry = ly + w * ty + (z * tx - x * tz)
        rz = lz + w * tz + (x * ty - y * tx)
        
        world_x = base_translation.x + rx
        world_y = base_translation.y + ry
        world_z = base_translation.z + rz
        
        return world_x, world_y, world_z
    
    def optimize_gripper_symmetry(self, q_w, q_x, q_y, q_z):
        """优化夹爪对称性，防止手腕扭转过大"""
        flipped_w = -q_z
        flipped_x = q_y
        flipped_y = -q_x
        flipped_z = q_w

        dot_original = abs(q_x) 
        dot_flipped = abs(flipped_x) 

        if dot_flipped > dot_original:
            return flipped_w, flipped_x, flipped_y, flipped_z
        else:
            return q_w, q_x, q_y, q_z
    
    def quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return w, x, y, z
    

    # =========================================================================
    # 动作控制区
    # =========================================================================

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

    def jiggle_joints(self):
        """轻微抖动关节，用于跳出规划器的局部极小值/奇异点"""
        if not self._current_joints:
            self.get_logger().warn("未获取到当前关节状态，无法执行抖动，默认回到原点。")
            self.move_to_joint_values([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
            return
            
        self.get_logger().info('尝试轻微抖动关节...')
        jiggled_joints = list(self._current_joints)
        for i in range(6):
            jiggled_joints[i] += random.uniform(-0.087, 0.087)
            
        self.move_to_joint_values(jiggled_joints)
    
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
        pc.header.frame_id = "world"
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
        oc.header.frame_id = "world"
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
                if attempt % 2 != 0:
                    self.jiggle_joints()
                else:
                    self.get_logger().info('尝试回到原点重新规划...')
                    self.move_to_joint_values([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
                time.sleep(1.0)
                
        return False

    def move_to_joint_values(self, joints):
        """关节空间角度移动"""
        if len(joints) != 7: return False
        
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "panda_arm"
        constraints = Constraints()
        
        for i in range(7):
            jc = JointConstraint()
            jc.joint_name = f"panda_joint{i+1}" 
            jc.position = float(joints[i])      
            jc.tolerance_above = 0.001          
            jc.tolerance_below = 0.001
            jc.weight = 1.0                     
            constraints.joint_constraints.append(jc)
            
        goal_msg.request.goal_constraints.append(constraints)
        goal_msg.request.pipeline_id = "ompl"   
        goal_msg.request.max_velocity_scaling_factor = 0.5
        
        send_goal_future = self._arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted: return False
        
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        return True

    def rotate_last_joint(self, angle_deg, vel=0.2, acc=0.01):
        """单独旋转第七轴 (偏航)"""
        if not self._current_joints: return False
        target_joints = list(self._current_joints)
        target_joints[6] += math.radians(angle_deg) 
        return self.move_to_joint_values(target_joints)


    # =========================================================================
    # 伺服与力控核心区
    # =========================================================================

    def set_servo_state(self, activate=True):
        """🌟 自动唤醒或挂起 Servo 节点"""
        client = self._start_servo_client if activate else self._pause_servo_client
        action_name = "启动 (Start)" if activate else "暂停 (Pause)"

        self.get_logger().info(f'正在尝试 {action_name} MoveIt Servo...')
        
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'❌ 找不到 Servo 服务 {client.srv_name}！')
            return False

        req = Trigger.Request()
        future = client.call_async(req)
        
        # 等待服务调用完成
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f'✅ {action_name} Servo 成功！')
                return True
            else:
                self.get_logger().warn(f'⚠️ {action_name} Servo 返回失败: {future.result().message}')
                return False
        else:
            self.get_logger().error(f'❌ 调用 {action_name} Servo 服务时发生异常')
            return False
    
    def stop_servo(self):
        """发送全 0 速度指令，强制伺服控制停机。"""
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.header.frame_id = "panda_hand"
        
        # 将线速度和角速度全部显式置为 0
        stop_msg.twist.linear.x = 0.0
        stop_msg.twist.linear.y = 0.0
        stop_msg.twist.linear.z = 0.0
        stop_msg.twist.angular.x = 0.0
        stop_msg.twist.angular.y = 0.0
        stop_msg.twist.angular.z = 0.0
        
        self._twist_pub.publish(stop_msg)
    
    def switch_control_mode(self, to_velocity=True):
        """🌟 动态切换底层控制器 (无缝切档)"""
        mode_str = "速度控制 (Velocity)" if to_velocity else "轨迹控制 (MoveIt JTC)"
        self.get_logger().info(f'🔄 正在将机械臂底层切换至: {mode_str} 模式...')
        
        if not self._switch_controller_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('无法连接到控制器切换服务！')
            return False
            
        req = SwitchController.Request()
        if to_velocity:
            req.activate_controllers = ['cartesian_velocity_controller']
            req.deactivate_controllers = ['panda_arm_controller']
        else:
            req.activate_controllers = ['panda_arm_controller']
            req.deactivate_controllers = ['cartesian_velocity_controller']
            
        # 必须使用 STRICT 模式，确保两个控制器同时且成功完成交接
        req.strictness = SwitchController.Request.STRICT 
        
        future = self._switch_controller_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None and future.result().ok:
            self.get_logger().info(f'✅ 成功切换至 {mode_str} 模式！')
            return True
        else:
            self.get_logger().error('❌ 控制器切换失败！请检查终端 Controller Manager 报错。')
            return False

    def wait_for_wrench(self):
        """确保能够读取到最新的力矩数据"""
        while rclpy.ok() and self._current_wrench is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.get_logger().info('正在等待力矩传感器数据...', throttle_duration_sec=1.0)
            time.sleep(0.01) # 稍微让渡一下 CPU 资源
        return self._current_wrench

    def force_guided_insert_moveit(self, start_x, start_y, start_z, target_z, q_w, q_x, q_y, q_z, wiggle_mode=False):
        """方案一改进版：Guarded Move + 相对插入深度判定"""
        self.get_logger().info('--- 启动改进版方案一 (Guarded Move 直达试探) ---')
        
        # 1. 传感器“去皮”
        rclpy.spin_once(self, timeout_sec=0.2)
        baseline_fz = self._current_wrench.force.z if self._current_wrench else 0.0
        self.get_logger().info(f'⚖️ 传感器已去皮，悬停基准 Z 轴受力: {baseline_fz:.2f} N')

        current_z = start_z
        current_x = start_x
        current_y = start_y
        search_radius = 0.0
        theta = 0.0
        
        # 🌟 新增：记录首次接触的高度
        first_contact_z = None
        insertion_depth_threshold = 0.010 # 10mm

        while current_z > target_z and rclpy.ok():
            self.get_logger().info(f'⬇️ 探针直接下压中 (目标高度: {target_z:.3f}) ...')

            # --- 构建直接“插到底”的运动请求 ---
            goal_msg = MoveGroup.Goal()                             
            goal_msg.request.group_name = "panda_arm"               
            goal_msg.request.pipeline_id = "pilz_industrial_motion_planner"
            goal_msg.request.planner_id = "LIN"
            goal_msg.request.max_velocity_scaling_factor = 0.05 
            goal_msg.request.max_acceleration_scaling_factor = 0.05

            c1 = Constraints()
            pc = PositionConstraint()
            pc.header.frame_id = "world"
            pc.link_name = "panda_hand" 
            
            bv = BoundingVolume()
            sp = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001])
            bv.primitives.append(sp)
            
            target_pose = PoseStamped()
            target_pose.pose.position.x = current_x
            target_pose.pose.position.y = current_y
            target_pose.pose.position.z = target_z
            bv.primitive_poses.append(target_pose.pose)
            
            pc.constraint_region = bv
            pc.weight = 1.0 
            c1.position_constraints.append(pc)

            oc = OrientationConstraint()
            oc.header.frame_id = "world"
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

            # --- 发送异步请求 (不阻塞) ---
            send_goal_future = self._arm_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error('下探请求被拒绝')
                break

            get_result_future = goal_handle.get_result_async()
            collision_detected = False

            # --- 🚀 开启高频监控与紧急刹车循环 ---
            while rclpy.ok() and not get_result_future.done():
                rclpy.spin_once(self, timeout_sec=0.01) 
                
                force_z = self._current_wrench.force.z if self._current_wrench else 0.0
                contact_force = abs(force_z - baseline_fz)

                if contact_force > CONTACT_FORCE_THRESHOLD:
                    self.get_logger().warn(f'💥 触发接触！纯接触力 = {contact_force:.2f} N。执行紧急刹车！')
                    cancel_future = goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future)
                    collision_detected = True
                    break 
            
            # --- 更新当前的真实高度 ---
            curr_pose = self.get_object_pose('panda_hand')
            current_z = curr_pose.translation.z if curr_pose else target_z

            if not collision_detected:
                self.get_logger().info('✅ 畅通无阻抵达目标深度，插入成功！')
                break

            # 🌟 新增：判断是否达到“从首次接触起下降了 10mm”的成功条件
            if first_contact_z is None:
                first_contact_z = current_z
                self.get_logger().info(f'📌 记录首次接触平面高度: {first_contact_z:.4f} m')

            if (first_contact_z - current_z) >= insertion_depth_threshold:
                self.get_logger().info(f'✅ 距离首次接触已顺利下降 {(first_contact_z - current_z)*1000:.1f} mm (>= 10mm)，判定为插入成功！')
                break

            # --- 发生碰撞后的退让与找正逻辑 ---
            self.get_logger().info('开始局部调整...')
            current_z += 0.002 # 抬起 2mm 释放挤压应力
            self.move_to_pose(current_x, current_y, current_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True, vel=0.1)
            
            if not wiggle_mode:
                search_radius += 0.0005 
                theta += 0.5            
                current_x = start_x + search_radius * math.cos(theta)
                current_y = start_y + search_radius * math.sin(theta)
                self.get_logger().info(f'螺旋搜索：平移至偏移量 d_x={current_x-start_x:.4f}, d_y={current_y-start_y:.4f}')
                self.move_to_pose(current_x, current_y, current_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True, vel=0.1)
            else:
                yaw_wiggle = random.choice([3.0, -3.0, 5.0, -5.0])
                self.get_logger().info(f'齿轮对准：微调第七轴 {yaw_wiggle} 度')
                self.rotate_last_joint(yaw_wiggle, vel=0.5, acc=0.05)


def force_guided_insert_servo(self, target_z, wiggle_mode=False):
        """方案二：基于硬件笛卡尔速度控制器的实时顺应控制"""
        self.get_logger().info('--- 启动方案二 (硬件级笛卡尔速度伺服装配) ---')
        
        # 1. 传感器去皮
        rclpy.spin_once(self, timeout_sec=0.2)
        baseline_fz = self._current_wrench.force.z if self._current_wrench else 0.0
        self.get_logger().info(f'⚖️ 传感器已去皮，悬停基准 Z 轴受力: {baseline_fz:.2f} N')
        
        # 🌟 2. 踩下离合，将控制权交给速度控制器
        if not self.switch_control_mode(to_velocity=True):
            self.get_logger().error("控制器切档失败，放弃本次插入！")
            return
            
        time.sleep(0.1) # 给予底层 100ms 握手时间
        
        base_down_vel = -0.01 
        first_contact_z = None
        insertion_depth_threshold = 0.010 # 10mm
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
                
                current_pose = self.get_object_pose('panda_hand')
                if not current_pose: continue
                current_z = current_pose.translation.z
                
                # 判断绝对高度是否抵达
                if current_z <= target_z:
                    self.stop_servo() 
                    self.get_logger().info('✅ 抵达绝对目标深度，伺服停机！')
                    break
                    
                # 判断相对插入深度 (10mm) 是否抵达
                if first_contact_z is not None and (first_contact_z - current_z) >= insertion_depth_threshold:
                    self.stop_servo()
                    self.get_logger().info(f'✅ 距离接触面已下降 {(first_contact_z - current_z)*1000:.1f} mm，插入成功！')
                    break

                # 计算纯接触力
                force_z = self._current_wrench.force.z if self._current_wrench else 0.0
                contact_force = abs(force_z - baseline_fz)
                
                twist_msg = TwistStamped()
                twist_msg.header.stamp = self.get_clock().now().to_msg()
                twist_msg.header.frame_id = "panda_hand" 
                
                # 🌟 碰撞/接触逻辑：平滑降速，绝不会亮白灯
                if contact_force > CONTACT_FORCE_THRESHOLD:
                    if first_contact_z is None:
                        first_contact_z = current_z
                        self.get_logger().info(f'📌 记录首次接触平面高度: {first_contact_z:.4f} m')

                    # 探测到接触时，保持轻微下压力，并根据横向受力滑动 (阻抗导纳特性)
                    twist_msg.twist.linear.z = -0.001 
                    
                    if not wiggle_mode:
                        force_x = self._current_wrench.force.x
                        force_y = self._current_wrench.force.y
                        k_admittance = 0.002 
                        twist_msg.twist.linear.x = force_x * k_admittance
                        twist_msg.twist.linear.y = force_y * k_admittance
                    else:
                        twist_msg.twist.angular.z = 0.2 * math.sin(time.time() * 10) 
                else:
                    twist_msg.twist.linear.z = base_down_vel
                    
                self._twist_pub.publish(twist_msg)
                time.sleep(0.01) 
                
        finally:
            # 🌟 3. 无论成功与否，务必停车并切回轨迹控制器
            self.stop_servo() 
            time.sleep(0.1)
            self.switch_control_mode(to_velocity=False)


# =========================================================================
# 重构的动作组合区：Pick & Prepare -> Inserting -> Retreat
# =========================================================================

def execute_pick_and_prepare(commander, config:dict, is_gear=False):
    """阶段 1：拾取并移动到预备位 (开环位置控制)"""
    target_name = config["target_name"]
    base_name = config["base_name"]
    pre_offset_z = 0.30
    grasp_offset_z = config["grasp_offset_z"]
    
    commander.send_gripper_goal(config["gripper_open"])

    # 寻找物体
    obj_pose = None
    while rclpy.ok() and obj_pose is None:
        rclpy.spin_once(commander, timeout_sec=0.1) 
        obj_pose = commander.get_object_pose(target_name)

    p = obj_pose.translation
    o = obj_pose.rotation
    q_w, q_x, q_y, q_z = commander.get_downward_orientation(o)
    q_w, q_x, q_y, q_z = commander.optimize_gripper_symmetry(q_w, q_x, q_y, q_z)
    
    commander.get_logger().info('-> 阶段 1: 拾取与预备 <-')
    commander.move_to_pose(p.x, p.y, p.z + pre_offset_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True)
    commander.move_to_pose(p.x, p.y, p.z + grasp_offset_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True, vel=0.15, acc=0.05)
    
    commander.send_gripper_goal(config["gripper_close"])
    time.sleep(1.0)
    commander.move_to_pose(p.x, p.y, p.z + pre_offset_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True, vel=0.15, acc=0.05)

    # 寻找基座
    base_pose = None
    while rclpy.ok() and base_pose is None:
        rclpy.spin_once(commander, timeout_sec=0.1) 
        base_pose = commander.get_object_pose(base_name)

    b = base_pose.translation
    o_b = base_pose.rotation 

    base_q_w, base_q_x, base_q_y, base_q_z = commander.get_downward_orientation(o_b)
    base_q_w, base_q_x, base_q_y, base_q_z = commander.optimize_gripper_symmetry(base_q_w, base_q_x, base_q_y, base_q_z)

    # 齿轮/手柄特殊姿态修正
    if target_name == "Gear_compound_handle":
        q_offset = (0.9962, 0.0, 0.0, -0.0872) 
        base_q_w, base_q_x, base_q_y, base_q_z = commander.quaternion_multiply((base_q_w, base_q_x, base_q_y, base_q_z), q_offset)
    elif target_name in ["Gear_large_handle", "Shaft_large"]:
        q_offset = (0.9998, 0.0, 0.0, 0.0175) 
        base_q_w, base_q_x, base_q_y, base_q_z = commander.quaternion_multiply((base_q_w, base_q_x, base_q_y, base_q_z), q_offset)

    local_offset = [config["place_offset_x"], config["place_offset_y"], 0]
    world_x, world_y, world_z = commander.local_to_world(b, o_b, local_offset)
    
    # 前往装配正上方的安全预备位
    prep_z = world_z + pre_offset_z
    commander.move_to_pose(world_x, world_y, prep_z, q_w=base_q_w, q_x=base_q_x, q_y=base_q_y, q_z=base_q_z, use_linear=True)
    
    # 确保力矩传感器已激活
    commander.wait_for_wrench()
    
    return world_x, world_y, prep_z, (world_z + config["place_offset_z"]), base_q_w, base_q_x, base_q_y, base_q_z

def execute_release_and_retreat(commander, config, current_x, current_y, prep_z, q_w, q_x, q_y, q_z):
    """阶段 3：释放并撤离"""
    commander.get_logger().info('-> 阶段 3: 释放与撤离 <-')
    commander.send_gripper_goal(config["gripper_open"])
    time.sleep(0.5)
    commander.move_to_pose(current_x, current_y, prep_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True)
    time.sleep(1.0)


# =========================================================================
# 具体技能函数
# =========================================================================

def inserting(commander, config:dict):
    """通用轴孔插入流程"""
    # 阶段 1：拾取准备
    x, y, prep_z, target_z, qw, qx, qy, qz = execute_pick_and_prepare(commander, config)
    
    # 阶段 2：力控装配 (普通插入模式，遇到阻力进行 XY 螺旋找正)
    commander.get_logger().info('-> 阶段 2: 力觉引导下沉 (Inserting) <-')
    if not commander.use_velocity_control:
        commander.force_guided_insert_moveit(x, y, prep_z, target_z, qw, qx, qy, qz, wiggle_mode=False)
    else:
        # 下降前，可以先用 MoveIt 降到比较近的距离，以节约伺服耗时
        commander.move_to_pose(x, y, target_z + 0.05, q_w=qw, q_x=qx, q_y=qy, q_z=qz, use_linear=True, vel=0.2)
        commander.force_guided_insert_servo(target_z, wiggle_mode=False)
        
    # 阶段 3：撤离
    execute_release_and_retreat(commander, config, x, y, prep_z, qw, qx, qy, qz)

def gear_meshing(commander, config:dict):
    """齿轮啮合装配流程"""
    # 阶段 1：拾取准备
    x, y, prep_z, target_z, qw, qx, qy, qz = execute_pick_and_prepare(commander, config, is_gear=True)
    
    # 阶段 2：力控装配 (齿轮模式，遇到阻力进行 偏航轴 旋转对正)
    commander.get_logger().info('-> 阶段 2: 齿轮力觉啮合 (Gear Meshing) <-')
    if not commander.use_velocity_control:
        commander.force_guided_insert_moveit(x, y, prep_z, target_z, qw, qx, qy, qz, wiggle_mode=True)
    else:
        commander.move_to_pose(x, y, target_z + 0.05, q_w=qw, q_x=qx, q_y=qy, q_z=qz, use_linear=True, vel=0.2)
        commander.force_guided_insert_servo(target_z, wiggle_mode=True)
        
    # 阶段 3：撤离
    execute_release_and_retreat(commander, config, x, y, prep_z, qw, qx, qy, qz)

def screw(commander, config:dict):
    """拧螺丝流程"""
    # 螺丝插入动作
    inserting(commander, config)

    target_name = config["target_name"]
    base_name = config["base_name"]
    grasp_offset_z = config["grasp_offset_z"]
    gripper_close = config["gripper_close"]
    gripper_open = config["gripper_open"]
    pre_offset_z = config["pre_offset_z"]
    count = 0

    while True:
        obj_pose = None
        base_pose = None
        while rclpy.ok() and (obj_pose is None or base_pose is None):
            rclpy.spin_once(commander, timeout_sec=0.1)
            obj_pose = commander.get_object_pose(target_name)
            base_pose = commander.get_object_pose(base_name)

        p = obj_pose.translation
        b = base_pose.translation
        q_w, q_x, q_y, q_z = commander.get_downward_orientation(obj_pose.rotation)
        
        if p.z < b.z + 0.01:
            commander.get_logger().info('螺丝已拧紧')
            commander.move_to_pose(p.x, p.y, p.z + pre_offset_z, q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z, use_linear=True)
            time.sleep(1.0)
            return

        count += 1
        commander.get_logger().info(f'第 {count} 次拧螺丝')
        commander.move_to_pose(p.x, p.y, p.z + grasp_offset_z, use_linear=True, vel=0.8, acc=0.05)
        time.sleep(2.0)
        
        if count % 2 == 0: 
            commander.rotate_last_joint(-90.0, vel=0.8, acc=0.05)
            
        commander.send_gripper_goal(gripper_close)
        time.sleep(1.0)
        
        commander.rotate_last_joint(90.0, vel=0.8, acc=0.05)
        time.sleep(2.0)
        
        commander.send_gripper_goal(gripper_open)
        time.sleep(2.0)


# =========================================================================
# 配置字典区
# =========================================================================
PARTS_CONFIGS = {
    "Shaft_small": {
        "target_name": "Shaft_small",
        "base_name": "Base",
        "skill": "inserting" ,
        "gripper_open": 0.04,
        "gripper_close": 0.015,
        "pre_offset_z": 0.30,
        "grasp_offset_z": 0.153,
        "place_offset_x": 0.0,
        "place_offset_y": 0.13,
        "place_offset_z": 0.17
    },
    "Shaft_large": {
        "target_name": "Shaft_large",
        "base_name": "Base",
        "skill": "inserting",
        "gripper_open": 0.04,  
        "gripper_close": 0.017,
        "pre_offset_z": 0.30,
        "grasp_offset_z": 0.17, 
        "place_offset_x": 0.127,
        "place_offset_y": 0.0, 
        "place_offset_z": 0.19
    },
    "Screw":{
        "target_name": "Screw",
        "base_name": "Base",
        "skill": "screw",
        "gripper_open": 0.02,
        "gripper_close": 0.005,
        "pre_offset_z": 0.30,
        "grasp_offset_z": 0.197,
        "place_offset_x": 0.0,
        "place_offset_y": 0.0,
        "place_offset_z": 0.22
    },
    "Gear_small_handle":{
        "target_name": "Gear_small_handle",
        "base_name": "Shaft_small",
        "skill": "gear_meshing",  
        "gripper_open": 0.04,
        "gripper_close": 0.01,
        "pre_offset_z": 0.30,
        "grasp_offset_z": 0.13,
        "place_offset_x": 0.0,
        "place_offset_y": 0.0,
        "place_offset_z": 0.18
    },
    "Gear_compound_handle":{
        "target_name": "Gear_compound_handle",
        "base_name": "Base",
        "skill": "gear_meshing",  
        "gripper_open": 0.04,
        "gripper_close": 0.01,
        "pre_offset_z": 0.30,
        "grasp_offset_z": 0.145,
        "place_offset_x": 0.0,
        "place_offset_y": 0.0,
        "place_offset_z": 0.18
    },
    "Gear_large_handle":{
        "target_name": "Gear_large_handle",
        "base_name": "Shaft_large",
        "skill": "gear_meshing",  
        "gripper_open": 0.04,
        "gripper_close": 0.01,
        "pre_offset_z": 0.30,
        "grasp_offset_z": 0.13,
        "place_offset_x": 0.0,
        "place_offset_y": 0.0,
        "place_offset_z": 0.18
    }
}


def main(args=None):
    rclpy.init(args=args) 
    commander = AssemblyCommander()

    try:
        parts_to_execute = [
            "Shaft_large", 
            "Shaft_small", 
            "Screw",
            "Gear_small_handle",
            "Gear_compound_handle",
            "Gear_large_handle"
        ] 
 
        for item_name in parts_to_execute:
            if item_name in PARTS_CONFIGS:
                current_config = PARTS_CONFIGS[item_name]
                skill = current_config["skill"]
                
                if skill == "inserting":
                    inserting(commander, current_config)
                elif skill == "gear_meshing":
                    gear_meshing(commander, current_config)
                elif skill == "screw":
                    screw(commander, current_config)
                else:
                    commander.get_logger().error(f"未找到 {item_name} 的技能信息！")
            else:
                commander.get_logger().error(f"未找到 {item_name} 的配置信息！")

    except KeyboardInterrupt:
        commander.get_logger().info('任务被用户手动中断')

    finally:
        commander.get_logger().info('正在移动到初始位置...')
        commander.move_to_joint_values([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        time.sleep(1.0)      
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
