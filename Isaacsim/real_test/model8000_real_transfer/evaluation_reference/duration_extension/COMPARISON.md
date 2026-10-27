# model8000指定8次浅插失败的时长对照

固定权重；仅指定回合延长至30s，25s为同轨迹截点。不同步复现须单独解读，不能算作原失败被救回。入孔成功与完成松爪分别统计。

|seed/env/episode|前20秒匹配|20秒深度mm|25秒成功/完成松爪|30秒成功/完成松爪|
|---|---|---:|---|---|
|42/24/1|True|17.97|True/True|True/True|
|42/9/3|True|22.06|False/False|False/False|
|42/9/4|True|11.45|False/False|False/False|
|43/18/2|True|24.19|True/True|True/True|
|44/11/1|True|21.16|True/True|True/True|
|44/23/1|True|18.68|True/True|True/True|
|44/30/2|True|9.72|True/True|True/True|
|44/31/4|True|8.97|True/True|True/True|

25秒：全部重跑成功6/8；严格前缀匹配子集成功6/8。

30秒：全部重跑成功6/8；严格前缀匹配子集成功6/8。

该测试只覆盖8次历史浅插失败，不代表384回合整体新成功率。未改全局20秒配置或部署包。
