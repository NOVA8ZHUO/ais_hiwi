> 当前控制修复与验收以 [CONTROLLER_REPAIR.md](CONTROLLER_REPAIR.md) 为准。下文旧测试结果和旧配置保留作历史记录；不要将旧成功率当作当前版本结果。

# 当前插入任务的测试、episode与PPO训练指南

核对日期：2026-09-28。新增的逐回合指标和三阶段命令见 [下一步检查](NEXT_RL_CHECKS.md)；下面历史测试结果保留原记录。以当前 `FORGE-Shaft-Insert-Paper-v0` 代码为准；本次整理不改变任务、奖励、终止或训练默认参数。论文对照见 [FORGE_ALIGNMENT.md](FORGE_ALIGNMENT.md)。

## 1. 先明确目前完成到哪里

- 已完成：物理夹持和随机化的接口检查、固定Base检查、持续夹紧检查、批量环境步进与重置、短PPO训练链路检查。
- 未完成：正式训练收敛、训练策略多回合成功率和真机验证；新增零残差基准结果见NEXT_RL_CHECKS.md，几何成功不能替代稳定夹持验收。
- **成功不等于done**：当前成功只给奖励；没有成功后立即结束的termination。
- **测试PASS不等于插入成功**：随机测试允许出现失败reset；保持测试根本不执行下插。
- 允许非零滑移和不对称，并不意味着任意滑移都合理；随机动作测试曾出现约131 mm相对位移，不能据此认定夹持已接近真机。

## 2. 已保存的每次测试过程与结果

以下是本版本归档日志能支持的测试，不把历史调参试验或旧版测试当成本版本验收。日志在 [residual_validation](../outputs/residual_validation/)。

### 所有paper_smoke_test共同执行的检查

1. seed=42创建环境；检查actor/critic维度24/48、最长150控制步、手指PD增益为0、每指20 N effort限制和10 mm/s速度限制。
2. reset后检查轴偏移：夹爪系x±8 mm、z±5 mm、y采样为0，经手姿态旋转到世界系；孔估计xyz各±10 mm；真实Base完整位姿不变。
3. 对第一个环境单独reset，检查其他环境随机量未改变，再全体reset。
4. 人工设置下插时刻，用x动作0.1核对名义轨迹+2.5 mm残差及目标裁剪；同一时刻重复动作不会累加偏移。
5. 把框架episode_length_buf暂设100，检查实际夹持计时仍从0开始，不会跳过等待。
6. 每控制步检查观测/奖励有限、未重置环境孔估计偏差保持不变、Base固定、闭合驱动力目标持续为每指−20 N；自动reset后复核随机化。
7. 记录宽度、轴相对hand位移、手指不对称、轴–Base接触力、终止原因等。统计发生在env.step返回后，已自动reset的环境不是终止前状态；因此峰值统计可能漏掉终止瞬间。

| 测试/日志 | 设置与实际动作 | 结果 | 能证明什么/不能证明什么 |
|---|---|---|---|
| `hold8.log` | 8环境×160控制步；等待时间改为11 s，死区置0；保留抓取和孔误差；xyz/yaw动作0，预测动作−1 | 8次超时；无越界/宽度失败；宽度35.12–35.22 mm；TCP最大偏移4.18 mm；轴相对hand最大位移11.01 mm；最大两指位置差23.08 mm | 证明整回合持续夹持与reset链路；未验证下插，也未证明夹持无滑移 |
| `random32.log` | 32环境×160步；完整随机化；全部5维动作每步U[−1,1] | 26次超时、3次越界、3次宽度失败；轴–Base力最大48.88 N；最大相对位移130.90 mm；成功步0 | 证明随机压力下接口与数值可运行，失败会reset；不能证明任务可解或策略有效 |
| `ppo2.log` | 8环境，32步rollout，2轮PPO，共512 transitions | 32.60/31.09 s每轮；训练部分64.72 s；无插入奖励；回报−13.18→−17.68 | 证明数据采集、LSTM和优化器链路；两轮不足以判断学习趋势 |
| `continuous_grip.log` | 最终实现，4环境×20步，hold模式 | 宽度35.125–35.188 mm；75个环境步满足开口变化<0.01 mm且闭合力目标仍−20 N；无终止 | 证明开口稳定后不会撤力；不证明实际指尖接触力精确等于20 N；本次不足150步，不覆盖超时 |

`constant_width_force_steps=75`是4×20=80个环境步中的计数，不是75个episode。
当前脚本逐步断言的是力目标持续存在；“稳定宽度计数必须>0”不是脚本通用PASS条件，本次归档时额外确认了75>0。
宽度范围主要在实际经过0.6 s且未done的样本上统计，所以random日志中的最小宽度仍>30 mm不否定存在宽度失败。
力传感器只统计轴–Base接触，保持测试显示0 N不代表夹爪没有夹紧。

## 3. 一个episode究竟发生什么

### reset

只重置需要结束的子环境，不重置其他环境。机器人直接放到hole_large上方的固定关节姿态，不学习取料或从取料点横移。两指初始各17.5 mm，持续闭合力立即写入；轴为独立动态刚体，没有固定连接。

每回合重新采样：

| 量 | 范围/定义 |
|---|---|
| 轴初始抓取误差 | hand系x±8 mm、z±5 mm，y不额外采样；无额外初始姿态随机化 |
| 孔估计偏差 | 世界轴xyz各±10 mm；整回合固定；真实孔不动 |
| 平移Kp | U[400,800]；Kd=2√Kp |
| 目标相对当前TCP裁剪λ | U[16,25] mm，每个坐标分量 |
| 平移力死区 | 每轴U[0,5] N |
| 奖励力阈值 | U[5,10] N |
| 轴与Base摩擦 | 各自静/动摩擦0.5–1.0，保持动摩擦不大于静摩擦；恢复系数0 |

同时清零关节/轴速度、动作历史、实际计时、观测差分状态；PPO在done时清理相应LSTM状态。真实Base恢复同一个固定姿态，运动学刚体不会因插入力移动。

### 时间与控制流程

- 物理240 Hz；每16个物理步请求一次策略动作，即15 Hz，控制步66.67 ms。
- 前0.5 s保持标称reset目标，持续夹紧。按离散时刻，第9次动作处理开始满足等待时间。
- 满足等待时间且宽度30–40 mm才执行下插与残差；不满足时目标退回reset目标，不撤销夹紧力。
- 名义轨迹相对于**估计孔位**，从孔上19.285 mm以8 mm/s降至孔下47 mm，约在实际t=8.79 s到达末端，然后保持。
- 轨迹是按实际经过时间推进，宽度不合格不会暂停相位计时；它不是接触触发的搜索/下插状态机。
- 每次动作后执行16个物理步，然后计算termination、reward，对done环境自动reset，最后生成返回观测。
- 通常最多10 s=150控制步。但训练器初次启动会随机设置框架episode_length_buf，让第一批episode提前超时；实际下插相位用独立elapsed_steps，仍从0开始。后续正常reset恢复完整150步上限。

### state与observation

物理state包括机器人关节位置/速度、轴位姿/速度、固定Base、接触和控制器随机参数等；策略并不能读取全部物理state。
Actor输入24维，部署时需要这些可测/可估计信息；critic输入48维，仅训练时使用仿真特权信息。长度均不含LSTM隐藏状态。

| actor索引（从0开始） | 量 | 维度 |
|---|---|---:|
| 0–2 | 虚拟TCP位置−估计孔位置，世界轴，m；TCP位置叠加每轴σ=0.25 mm噪声 | 3 |
| 3–6 | hand世界姿态四元数wxyz | 4 |
| 7–9 | 带噪TCP位置相邻控制步差分/Δt，m/s；reset首次为0 | 3 |
| 10–12 | TCP/hand世界角速度，rad/s | 3 |
| 13–15 | 轴–Base接触力世界向量，叠加每轴σ=1 N噪声 | 3 |
| 16 | 当前回合Fth，N | 1 |
| 17–21 | 上次已执行的5维裁剪动作；reset为0 | 5 |
| 22 | 名义下插相位0–1 | 1 |
| 23 | 实际总夹爪开口，m | 1 |

虚拟TCP使用固定hand→标称轴底变换，不能直接反映轴滑移后的实际轴底。actor不直接看到真实轴位姿、真实孔误差、接触力矩、相机图像或7个关节位置。
`enable_corruption=False`只是没有使用框架噪声模块，函数内部仍主动加上述噪声。

Critic按顺序：TCP−真实孔3、hand四元数4、TCP线/角速度6、轴位置−真实孔3、轴四元数4、轴线/角速度6、真实接触力3、Kp1、λ1、死区3、Fth1、动作5、相位1、宽度1、孔偏差3、抓取偏差3，共48维。

### action

策略输出5维，执行前裁剪到[−1,1]。它不直接控制夹爪，也不直接输出关节力矩。

| 分量 | 实际含义 |
|---|---|
| a0,a1,a2 | 世界轴xyz位置残差，每轴±25 mm，叠加在当前时刻的名义下插目标上；不逐步累积 |
| a3 | 目标yaw=a3×45°，roll固定180°、pitch固定0；不是yaw增量 |
| a4 | 预测成功概率p=(a4+1)/2；只参与预测奖励，不触发结束或夹爪动作 |

目标还会被裁剪到“当前实际虚拟TCP±λ”，λ每回合16–25 mm；这是每轴目标误差限制，不是机器人实际速度或实际运动距离的保证。
阻抗控制器在240 Hz下把位姿误差变为末端力/矩，经Jᵀ映射和重力补偿输出关节力矩。转动Kp=30、Kd=2√30。夹爪独立每物理步施加每指−20 N闭合驱动力。
当前没有动作变化率惩罚、低通平滑或单独的yaw速率限制；持续±1来回跳可能造成抖动。

## 4. reward、成功、失败、done

### reward（每个15 Hz控制步）

令d为轴上0/20/40 mm三个关键点到目标关键点的平均欧氏距离（m）；目标轴底为真实孔口下47 mm。

```text
K(d;a,b) = 1 / (exp(-a*d) + b + exp(a*d))
r = K(d;50,2) + K(d;100,0)
    + place + success
    - 0.2 * max(||F_shaft_base|| - Fth, 0)
    - abs((a4+1)/2 - success)
```

- 关键点项最高0.75；位置与轴方向共同影响d。
- place=1：真实轴底横向误差≤2.5 mm、相对孔口z∈[−48,+2.5] mm、轴倾角≤5°；它覆盖对准孔口和有效深度范围，并非完整成功。
- success=1的严格几何条件见下表。place与success每个满足条件的控制步都会给分，不是首次到达的一次性奖励。
- 接触力使用无额外观测噪声的代理力；阈值不是硬上限。例如F=20 N、Fth=7.5 N，该项为−2.5。
- 最优单步奖励上界2.75；超力惩罚无单独下界。没有专门的时间惩罚、掉落惩罚或失败一次性罚分。
- 配置权重已除以policy_dt，以抵消RewardManager默认乘dt，所以这里确为每控制步奖励。
- TensorBoard的`Episode_Reward/*`为对应回合奖励和除以最大时长10 s；`Mean reward`是另一种回合回报统计。不能把日志分项直接当成单步奖励。

### 成功与终止条件

| 条件 | 当前精确定义 | 是否结束episode |
|---|---|---|
| 几何成功 | 真实轴底距真实孔中心xy≤0.5 mm；深度46–48 mm；轴正z方向与世界正z夹角≤5°；三项同时满足 | **否**，只奖励和生成预测标签；无持续时长判据 |
| `outside_workspace` | 真实轴底相对孔口xy距离>100 mm，或z<−57 mm，或z>+100 mm | 是，terminated=True |
| `invalid_grasp_width` | 实际经过≥1 s且总宽度不在30–40 mm（含边界） | 是，terminated=True；单步触发，无连续超限滤波 |
| `time_out` | 框架episode长度达到150控制步 | 是，truncated=True；不是成功标签 |
| 超力 | 超过本回合5–10 N阈值 | 否，只有负奖励 |
| 滑移/旋转/手指不对称 | 没有独立阈值 | 否；可能进一步触发越界或宽度失败 |
| 预测概率高 | a4接近1 | 否 |

`done = terminated OR truncated`。多个条件可同一步成立，termination计数不一定互斥。对超时，RSL-RL采用timeout价值补偿，与真正失败区别处理。
Gym返回`obs, reward, terminated, truncated, info`：reward/done属于刚结束的动作；done环境的obs通常已经是下一回合reset后的观测。

**特别注意**：当前几何成功没有要求夹爪仍夹住轴、持续稳定若干步或接触力低于阈值；脱手轴若恰好满足几何条件也会得到成功标签。当前也没有独立掉落检测。这里只披露现状，没有擅自替你改成功定义。
正式评估建议另外统计“回合曾成功”“回合最后成功”“连续保持≥0.5 s成功”“脱手/超力次数”，不要只用累计成功步数冒充成功率。

## 5. PPO参数与batchsize

当前默认值属于工程起点，不能说论文证明这些值最优。配置文件：[rsl_rl_ppo_cfg.py](../forge_shaft_insert/agents/rsl_rl_ppo_cfg.py)。

| 参数 | 当前默认 | 初次训练建议 |
|---|---:|---|
| num_envs | 场景默认1，命令行覆盖 | 可视化4；正式起步128，测速后再考虑256/512 |
| num_steps_per_env | 32 | 先保持32，即每轮每环境采样2.133 s |
| max_iterations | 1000 | 先20–50轮调试，再200轮观察趋势，最后1000轮起步 |
| num_mini_batches | 4 | 保持4；环境数须能被mini-batch数整除，且每组至少2环境；4环境调试覆盖为2组 |
| num_learning_epochs | 5 | 保持5，即每轮通常20次优化更新 |
| learning_rate | 3e−4 | 保持；adaptive随KL调节，不是恒定学习率 |
| desired_kl / clip_param | 0.01 / 0.2 | 保持；不要靠增大学习率补偿几何或控制问题 |
| gamma / lam | 0.99 / 0.95 | 保持 |
| entropy_coef / init_noise_std | 0.001 / 0.5 | 先保持；初始位置探索σ约12.5 mm、yaw约22.5°（裁剪前） |
| max_grad_norm | 1.0 | 保持 |
| value_loss_coef / clipped value loss | 1.0 / True | 保持 |
| policy/critic | 各自LSTM128×1层、MLP[128,128]、ELU | 保持；actor/critic观测均归一化 |
| save_interval | 50轮 | 正式训练保持；20轮调试可临时改10 |

本实现没有一个独立`batch_size`控制所有采样。每轮采样量B=N环境数×T步数；等效每mini-batch有效transition数约B/M。
当前是recurrent PPO，实际按环境轨迹分组，episode边界处分段、padding和mask，不能把它理解成任意打散的独立样本。

| N | T | B每轮新样本 | M | 每mini-batch等效样本 | 1000轮总样本 |
|---:|---:|---:|---:|---:|---:|
| 4 | 32 | 128 | 2 | 64 | 128,000 |
| 8 | 32 | 256 | 4 | 64 | 256,000 |
| 128 | 32 | 4,096 | 4 | 1,024 | 4,096,000 |
| 256 | 32 | 8,192 | 4 | 2,048 | 8,192,000 |
| 512 | 32 | 16,384 | 4 | 4,096 | 16,384,000 |

5个epoch重复使用同一批数据，不会让新样本数乘5。一次iteration不是一次episode；rollout可以包含episode的一部分或多个episode片段。LSTM能跨rollout保存状态，但本次训练序列长度仍受rollout/episode分段限制。
不要直接用默认1环境训练配4个mini-batch，本机RSL的循环batch生成器会出现每组0环境；本机版本的1环境仅用于测试/播放；循环PPO训练每组至少2环境，不能通过改为M=1规避单环境轴问题。

参考字段含义：[Isaac Lab 2.3.1官方PPO配置文档](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab_rl/isaaclab_rl.html)。项目实际行为以本机2.3.2源码为准。

## 6. 推荐顺序与可复现命令

以下都从工作区根目录 `/home/masais/Xinyu/Isaacsim` 执行，保证训练和播放日志根目录一致。python使用已安装Isaac Sim的环境。

```bash
# A. 4环境可视化保持：看持续夹紧、Base固定、初始偏差和自然滑移
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode hold --num_envs 4 --steps 160

# B. 4环境可视化零残差下插：先检查几何/控制是否能完成名义动作
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode directed --nominal --num_envs 4 --steps 145

# C. 再打开真实误差，零残差不是预期一定成功，而是看接触与误差影响
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode directed --num_envs 4 --steps 160

# D. 4环境可视化短训练，20轮只检查行为和链路
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --num_envs 4 --max_iterations 20 --run_name visual_debug agent.save_interval=10 agent.algorithm.num_mini_batches=2

# E. 128环境headless起步训练，200轮后先评估
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --headless --num_envs 128 --max_iterations 200 --run_name residual_pilot agent.num_steps_per_env=32 agent.algorithm.num_mini_batches=4

# F. 正式1000轮；这是新训练，不能自动延续E，想续训须明确指定resume和检查点
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --headless --num_envs 128 --max_iterations 1000 --run_name residual_main

# G. 替换为实际检查点绝对路径；少量环境播放，观察确定性策略
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/play.py --task FORGE-Shaft-Insert-Paper-v0 --num_envs 4 --checkpoint /absolute/path/to/model_200.pt

# H. 单独终端看训练曲线（环境中需已安装tensorboard）
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/tensorboard --logdir logs/rsl_rl/forge_shaft_insert_residual
```

本次没有运行GUI，也没有替你启动200/1000轮长期训练。B/C是下一步建议，不能当作已经通过；`--nominal`只关掉抓取/孔误差与力死区，Kp/摩擦等仍随机。
先可视化检查，再headless训练，期间每50–100轮挑检查点播放。不要一边训练一边开另一份大规模仿真测速。只播放4环境比始终渲染128环境更方便定位问题。
旧24/48维之前的checkpoint和当前动作语义不兼容；只能续训本版本兼容的模型。

## 7. 训练时重点看什么

| 优先级 | 看什么 | 怎样判断/如何处理 |
|---|---|---|
| 1 | 独立评估成功率与深度/横向误差 | 固定评估种子，使用确定性策略；至少100回合初筛，再500–1000回合；区分低误差、完整误差和边界误差 |
| 2 | 成功是否可靠 | 记录成功保持时间、终局成功、轴是否仍在夹爪、失败原因；总reward上涨但真实成功不涨，不算完成 |
| 3 | 接触与滑移 | 看力峰值/P95、超阈值时长、轴相对hand位移、姿态变化；可允许真实范围内滑移，不能忽略明显脱手 |
| 4 | 动作/运动质量 | 残差±1饱和比例、相邻动作差、TCP速度、姿态振荡；训练探索抖动和确定性播放抖动分开判断 |
| 5 | 分项reward | keypoints/place/inserted是否改善；contact惩罚是否变小；prediction变好可能只是总预测失败，并不代表插入进步 |
| 6 | 终止分布与episode长度 | 短回合且宽度/越界多通常有问题；长回合也可能只是停着不动；time_out不能直接当失败率或成功率 |
| 7 | PPO数值与速度 | value loss、policy loss、noise std、学习率、采样/学习耗时、显存；NaN/Inf/OOM必须停下排查；KL/clip fraction若未记录需另加监控 |

现成TensorBoard会有Mean reward/episode length、各reward与termination分项、noise std、loss和性能等；**原始测试时没有上述独立指标；现已新增回合成功率、回合P95力、动作饱和率和连续成功时长日志**。新增指标见NEXT_RL_CHECKS.md；历史日志无法追溯补出当时未保存的逐回合数据。
日志`Episode_Termination/*`是框架统计，不应未经确认直接解释为“全部episode的失败率”；测试脚本原始次数也可能有同一步多原因重叠。

建议检查点：20–50轮确认无明显数值问题；200轮比较固定评估集；若到500轮仍完全没有几何成功，应先检查零残差基准、成功标签是否可达、滑移和接触，而不是直接加到10000轮。
必要时再设计由低误差到完整误差的课程，但应单独标记实验；不能为了曲线上升悄悄缩小最终用户误差范围。

## 8. 时间估算

本机GPU为RTX4090 24GB。8环境短测约31.85 s/iteration，其中采集约30–32 s、更新约1 s；仅512个样本不足以证明长期稳定速度。启动/资产加载另计。
时间公式：训练墙钟时间≈启动时间+iterations×稳态秒/iteration；总新样本=N×32×iterations。
环境数增加不会按比例降低每轮时间，它增加每轮并行样本；最终是否更快应比较samples/s、显存与学习效果。

128环境实测与对应估算如下。GUI速度尚未实测，任何GUI时间都是预算范围而非测量值。训练完成指定轮数与策略收敛是两件事，现有结果不能保证多少小时必然成功。


### 本次新增128环境测速（已完成）

命令为上面的训练入口加 `--headless --num_envs 128 --max_iterations 5 --run_name timing128_review`，其余PPO参数保持默认；不是接着8环境模型训练。
日志：[ppo128_timing.log](../outputs/residual_validation/ppo128_timing.log)。5轮20,480 transitions，训练部分155.46 s；每轮9.87、32.01、36.77、37.19、38.47 s。
第一轮大多还未深入接触、初次episode也被随机截短，明显较快，不适合拿它估计长期耗时。最后3轮平均37.48 s，采集35.8–37.5 s、更新约1 s，瓶颈主要在仿真采集而不是PPO优化。
5轮仍无inserted奖励，回报下降不能据此判定训练已失败；初次回合长度逐渐恢复、接触增多也影响回报。但同样不能宣称策略正在收敛。

| 阶段 | 设置 | 墙钟预算与依据 |
|---|---|---|
| GUI场景检查 | 4环境hold/directed，每次150–160步 | 每次暂留3–8分钟，三组约10–25分钟；未实测GUI，包含启动余量 |
| GUI短训练 | 4环境20轮 | 暂留15–30分钟；按约35–75 s/轮作预算，GUI未测速 |
| 小规模正式前验证 | 128环境headless，200轮 | 实测速率外推约2.08小时；建议预留2–3小时 |
| 第一轮正式训练 | 128环境headless，1000轮，409.6万样本 | 外推10.41小时；建议预留10–15小时，不含后续调参/评估 |
| 继续训练 | 128环境headless，每增加1000轮 | 约再10–15小时；如果共3000轮，预算30–45小时 |
| 256/512环境 | 先各10–20轮测速再决定 | 当前未测，不承诺更快；按实测秒/轮重新乘轮数，同时比较同样样本预算和成功率 |

初次项目建议安排半小时至一小时检查、2–3小时pilot，再决定是否进行一晚到一天的正式训练；完整获得稳定策略可能需要多次实验，目前不能给出必然收敛时间。
不要只按平均155.46/5来估算，也不要直接把并行环境从128翻倍就声称时间减半。真实接触越来越多、失败频率变化会改变采集速度。

若接着pilot续训，可从同一工作目录运行（替换实际run目录和checkpoint文件名）：

```bash
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --headless --num_envs 128 --resume --load_run '实际pilot运行目录名' --checkpoint 'model_实际编号.pt' --max_iterations 800 --run_name residual_continue
```

本机runner的learn参数表示本次追加的循环次数；续训不要把它误解为统一的最终iteration编号。检查启动日志确认确实加载了预期检查点。以上命令中的占位符必须替换，不能原样执行。


### 20轮GUI训练后的配置修正

本机RSL-RL循环PPO在每mini-batch只有1个环境时，使用无维度参数的squeeze误删环境轴，导致概率比广播错误。4环境/4组不适用；之前的该项建议已更正为**4环境/2组**，128环境/4组保持不变。训练入口已检查整除及每组至少2环境，不修改安装库。
`2026-09-28_13-28-16_visual_debug_metrics`受此问题影响，不建议续训该检查点或用它判断PPO能否收敛。它的环境轨迹统计仍有效：17个完成回合，0成功、16次超时、1次越界、0夹宽失败；策略更新受到错误概率比影响。
