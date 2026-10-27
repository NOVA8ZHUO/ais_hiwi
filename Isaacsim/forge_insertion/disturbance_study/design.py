"""Predeclared factors and adaptive budget. No training in this experiment."""
GROUPS = {
 'pose': ['base_position','base_yaw','hand_position','hand_orientation','grasp_offset'],
 'physics': ['mass','friction'],
 'control': ['gains','ema','position_threshold','rotation_threshold','dead_zone'],
 'observation': ['hole_observation','tip_position_noise','tip_rotation_noise','force_noise'],
 'other': ['contact_threshold','quaternion_sign'],
}
ALL = [f for fs in GROUPS.values() for f in fs]
LABELS = {'pose':'初始位姿','physics':'物理参数','control':'控制参数','observation':'观测扰动','other':'其他随机项'}
INITIAL_SEEDS = list(range(5))
MAX_SCREEN_SEEDS = 15
CONFIRM_SEEDS = list(range(100,105))
N = 128
STEPS = 160
RESET_SLOTS = 2048
IK_ATTEMPTS = 32

LABELS.update({
 'base_position':'base 初始位置','base_yaw':'base 初始朝向',
 'hand_position':'夹爪初始位置','hand_orientation':'夹爪初始姿态',
 'grasp_offset':'shaft 夹持偏移','mass':'shaft 质量','friction':'base 静摩擦',
 'gains':'控制增益','ema':'EMA 平滑系数','position_threshold':'位置动作阈值',
 'rotation_threshold':'旋转动作阈值','dead_zone':'死区阈值',
 'hole_observation':'孔位置观测误差','tip_position_noise':'指尖位置观测噪声',
 'tip_rotation_noise':'指尖旋转观测噪声','force_noise':'力传感噪声',
 'contact_threshold':'接触惩罚力阈值','quaternion_sign':'四元数符号增广',
})
