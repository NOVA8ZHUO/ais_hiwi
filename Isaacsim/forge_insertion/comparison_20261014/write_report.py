from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'evaluation_summary.json').read_text())
assert len(rows)==4,'Complete all four evaluation cases first'
labels={'random_on_random':'随机化模型 → 随机化环境','random_on_zero':'随机化模型 → 固定环境','zero_on_random':'零随机化模型 → 随机化环境','zero_on_zero':'零随机化模型 → 固定环境'}
lines=['FORGE 自有 CAD：随机化与零随机化训练结果对比','',
'协议：两版均使用最终第 200 轮 checkpoint；128 环境、128 回合、seed=0、10秒回合。沿用原辅助评估协议与随机动作采样（player.deterministic=False）。',
'原模型在随机环境的结果复用既有评估，其余三组本次补测。这里是同协议、同环境配置的比较，不宣称不同配置之间拥有逐项相同的随机数轨迹。',
'两版实际 agent.yaml 除输出目录外完全一致；CAD 内容一致，环境配置差异限于已列明的随机化项和复制后的资产路径。',
'','独立评估：']
for d in rows:
 lines.append(f"{labels[d['case']]}：最终成功 {d['success_final_count']}/128 ({d['success_final_count']/128:.2%})；曾经成功 {d['success_any_count']}/128；平均回报 {d['reward_mean']:.2f}；成功回合首次成功时间 {d['first_success_seconds_mean_success_only']:.3f} s；回合平均腕部力 {d['episode_force_mean_N']:.2f} N；平均最终插入深度 {d['final_depth_mean_mm']:.2f} mm。")
lines += ['', '训练过程：',
'两版均完成 200 轮，无非有限标量。随机化训练计时 14938.03 s（4小时8分58秒），零随机化 14721.89 s（4小时5分22秒）；机器并发负载不同，不能将这点时间差归因于算法。',
'最后20轮训练回合结束成功率均值：随机化98.67%，零随机化100%。最后20条训练奖励均值348.63与403.84；奖励日志末条为epoch199，与最终checkpoint文件名中的epoch200奖励不是同一统计时刻。',
'10条记录滑动平均首次达到95%：随机化第49轮，零随机化第12轮。这是首次达到，不是永久稳定收敛。零随机化第31轮曾回落至2.34%，第21–40轮均值83.44%，后续恢复。',
'', '解释与限制：',
'零随机化模型在固定场景学得更快、训练奖励更高，但面对完整随机化集合时泛化明显下降。原随机化模型更适合作为包含位姿、质量、摩擦、控制参数及观测扰动的基线。',
'随机环境下零随机化模型37次曾成功、30次最终成功，说明不仅有未插入情况，还有达到判据后未保持的情况。不可只报曾经成功率。',
'零随机化跨环境平均腕部力升高、平均插入深度下降，与装配更困难一致；不能仅凭这些汇总量断言具体卡在哪个接触位置。力是控制频率采样的平滑腕力模长，不是高频接触峰值。',
'同时关闭多个随机化项，不能将差异单独归因于某一项（例如质量或位置）；若要定位原因，需要逐项消融。',
'所有四组都未出现插入深度超过50mm的回合，但这不是完整碰撞真实性证明。',
'只有一个训练seed和一个评估seed；128/128不代表真实成功概率必然为100%，固定环境的128次也不是128种不同装配位姿。该结果不证明真机性能。',
'相关代码、指标、逐回合原始数据与图均保存在本目录。']
(ROOT/'analysis.txt').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
order=['random_on_random','zero_on_random','random_on_zero','zero_on_zero']
by={d['case']:d for d in rows}
fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
values=[100*by[k]['success_final_count']/128 for k in order]
bars=ax.bar(['Rand model\nRand env','Zero model\nRand env','Rand model\nFixed env','Zero model\nFixed env'],values,color=['#2678b2','#ee8c32','#2678b2','#ee8c32'])
ax.set_ylim(0,112);ax.set_ylabel('Final success (%)');ax.set_title('Final epoch-200 checkpoints | 128 episodes per condition | seed 0')
for b,v,k in zip(bars,values,order):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.2f}%\n{by[k]["success_final_count"]}/128',ha='center',va='bottom',fontsize=9)
fig.savefig(ROOT/'evaluation_comparison.png',dpi=160)
print(ROOT/'analysis.txt')
