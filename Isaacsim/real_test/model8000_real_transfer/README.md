# model8000 真机部署交接包

入口：请真机电脑上的Codex完整阅读 **DEPLOYMENT_HANDOFF.md**，再检查本机驱动并实现real_adapter。

本包包含8000原始权重、独立TorchScript actor（含归一化）、LSTM推理包装、序列测试、训练配置/源码快照和评估结论。推理可用CPU，不需要Isaac Sim。尚不含接通未知真机的驱动程序。

```bash
python3 verify_package.py
python3 runtime/test_vectors.py
```

这两条为离线校验，不发送机器人指令。真实配置从real_config.template.json另存填写。不要改历史源码来填写标定，也不要运行build_export_reference.py作为部署启动命令。

8000导出对原RSL策略batch1/4各200步（含reset）最大输出误差0；并非真机行为已验证。默认仿真20秒，25秒只经过8个历史浅插失败定向验证，详见交接文档。
