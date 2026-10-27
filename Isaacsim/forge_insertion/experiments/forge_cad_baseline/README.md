# 官方 FORGE + 自有 CAD 实验

实现与资产定义见 ../../forge_cad/README.md。

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion
./run.sh validate --variant cad
./run.sh smoke --variant cad
./run.sh short --variant cad
# 完整训练由train_pipeline.py --variant cad启动，勿重复开启
```

预检查验收记录：各logs子目录的validation.json、evaluation.json、manifest.json。
正式训练与自动评估状态：pipeline_status.json；终端查看 `tail -f pipeline.log`。

`config_audit.md`列出与未修改官方配置的全部差异；`EXPERIMENT_CHANGELOG.md`记录资产修复。
标称质量17.5g，官方DR后为12.5–22.5g。没有25mm松爪逻辑。
