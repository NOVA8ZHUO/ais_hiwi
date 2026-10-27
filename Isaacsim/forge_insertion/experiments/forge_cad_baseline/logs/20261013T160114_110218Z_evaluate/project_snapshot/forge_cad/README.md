# 官方 FORGE + 自有 CAD

任务ID：`Isaac-Forge-ShaftLarge-CAD-Direct-v0`。环境直接使用官方 `ForgeEnv`，PPO注册指向官方YAML；不引入旧项目的控制器、奖励、松爪、观测或训练配置。

资产仅提取旧shaft_large/base的网格拓扑与SDF碰撞参数，转换成独立USD，不再依赖旧项目运行时代码。网格顶点只做刚体坐标变换，不缩放或改变孔径。源文件hash、尺寸和输出hash见 `assets/cad/manifest.json`。

允许的资产差异：
- Shaft：83 mm总长、35 mm夹持段、25 mm插入段；标称质量17.5 g（用户指定）。保留官方reset时加减5 g规则，实际采样12.5–22.5 g。
- Base：孔口原坐标(127,-0.18226,50) mm；落座目标为(127,-0.18226,3) mm，使轴肩在47 mm插深接触Base上表面。资产原点转换至此目标，孔口相对高度47 mm。
- Base保留官方固定件标称质量50 g并固定于世界；不继承旧CAD的1 kg和可动态运动状态。它是固定件，不进行真机质量匹配。
- 名义hand初始相对高度由47 mm改为80 mm，恰好补偿轴长增加33 mm，保持官方轴底离孔口的初始间隙及随机化幅度。
- 抓取直径元数据改为35 mm；官方夹爪闭合控制、力矩上限、刚度阻尼不变。

全部其他配置保持继承：128环境、200 epochs、10秒回合、actor24/critic61/action7、RL-Games PPO、奖励公式和权重、全部DR范围、Franka、控制增益与动作尺度。

成功仍按官方逻辑：横向误差<2.5 mm，轴底距落座目标高度<0.047×0.04=1.88 mm，即插深>45.12 mm。无25 mm释放、无放宽判据、无固定轴到夹爪。官方判据本身没有过深下界，本任务不擅自添加；诊断应检查非物理穿透。

验收入口（与训练完全隔离的探针，不作为策略性能）：

```bash
./run.sh validate --variant cad
./run.sh smoke --variant cad
./run.sh short --variant cad
./run.sh train --variant cad
```

`validate`对比官方配置，白名单以外任何参数变化都失败；验证资产结构、随机化质量、两次reset和夹持、官方判据，以及隔离的孔内/实体壁碰撞探针。通过后才进入训练。结果保存于 `experiments/forge_cad_baseline`，不会覆盖原 `forge_peg_baseline`。

当前几何和接口适配不保证200轮能收敛，也不沿用原模型95.31%的成功率。旧模型迁移评估若执行会单独标记；正式训练从零开始。
