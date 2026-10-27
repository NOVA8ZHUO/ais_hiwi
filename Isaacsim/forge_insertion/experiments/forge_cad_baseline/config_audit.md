# CAD任务配置审查

官方基准：IsaacLab v2.3.2 / commit 37ddf626871758333d6ed89cf64ad702aef127d0。
原任务：Isaac-Forge-PegInsert-Direct-v0；独立任务：Isaac-Forge-ShaftLarge-CAD-Direct-v0。
环境类仍为官方 ForgeEnv，训练仍为官方 RL-Games train.py 与同一个注册YAML。

以下为验收时实例化两个配置得到的全部差异（白名单以外任何差异都会中止验收）：

| 路径 | 官方 | CAD |
|---|---|---|
| `task.held_asset.spawn.mass_props.mass` | `0.019` | `0.0175` |
| `task.held_asset.spawn.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_peg_8mm.usd` | `/home/masais/Xinyu/Isaacsim/forge_insertion/assets/cad/shaft_large.usda` |
| `task.hand_init_pos` | `[0.0, 0.0, 0.047]` | `[0.0, 0.0, 0.08]` |
| `task.held_asset_cfg.mass` | `0.019` | `0.0175` |
| `task.held_asset_cfg.diameter` | `0.007986` | `0.035` |
| `task.held_asset_cfg.height` | `0.05` | `0.083` |
| `task.held_asset_cfg.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_peg_8mm.usd` | `/home/masais/Xinyu/Isaacsim/forge_insertion/assets/cad/shaft_large.usda` |
| `task.fixed_asset.spawn.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_hole_8mm.usd` | `/home/masais/Xinyu/Isaacsim/forge_insertion/assets/cad/base.usda` |
| `task.fixed_asset_cfg.diameter` | `0.0081` | `0.026` |
| `task.fixed_asset_cfg.height` | `0.025` | `0.047` |
| `task.fixed_asset_cfg.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_hole_8mm.usd` | `/home/masais/Xinyu/Isaacsim/forge_insertion/assets/cad/base.usda` |

轴质量17.5 g来自用户明确指令；质量随机化仍为标称值±5 g。
手部初始高度增加33 mm补偿轴长增加33 mm，保留初始轴底间隙和随机化范围。
固定件任务原点位于轴肩落座时的轴底位置，目标孔口相对高度47 mm。
USD拓扑差异：SDF网格；轴为浮动单刚体、无关节articulation；Base使用对世界固定关节。
原始顶点/三角面完整保留，只做坐标变换和资产物理接口包装；未继承旧动力学控制。

官方判据系数不变，因此成功插深门槛是45.12 mm、横向误差2.5 mm。不包含旧25 mm松爪流程。
公式没有额外过深下界，独立评估单独记录最大深度及>50 mm计数，不静默改变官方成功率定义。

与论文差异仍继承原官方Baseline，见 ../forge_peg_baseline/config_audit.md。本任务不声称严格论文复现。
