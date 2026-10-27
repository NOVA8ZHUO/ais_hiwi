import csv,json
from pathlib import Path
import numpy as np
from design import GROUPS,LABELS
ROOT=Path(__file__).resolve().parent

def pct(v):return f'{v*100:.2f}%'
def effect(d):return f"{d['estimate']*100:.2f} [{d['ci95'][0]*100:.2f}, {d['ci95'][1]*100:.2f}]"
def table(result,label):
    lines=[f'## {label}','',f"评估 seeds：{result['seeds']}；每 seed 128 回合。固定环境成功率 {pct(result['zero_success'])}，完整随机化 {pct(result['full_success'])}。",'',
      '| 扰动 | 单独开启成功率 | 单独关闭成功率 | 开启导致下降 pp [95% CI] | 关闭带来恢复 pp [95% CI] | 敏感性分数 pp [95% CI] |',
      '|---|---:|---:|---:|---:|---:|']
    by={r['name']:r for r in result['items']}
    for name in result['ranking']:
        r=by[name];lines.append(f"| {LABELS.get(name,name)} | {pct(r['on_success'])} | {pct(r['off_success'])} | {effect(r['enable_drop'])} | {effect(r['disable_recovery'])} | {effect(r['score'])} |")
    lines+=['',f"前两名分数差：{effect(result['top_gap'])} pp；是否满足预设明确领先条件：{result['clear_leader']}。",'']
    return lines

def write_report():
    group=json.loads((ROOT/'group_screen.json').read_text());factor=json.loads((ROOT/'factor_screen.json').read_text())
    confirm=json.loads((ROOT/'confirmation.json').read_text());lock=json.loads((ROOT/'lock.json').read_text())
    lines=['# 当前零随机化模型的扰动敏感性评估','',
      '本实验仅评估固定模型，不训练或更新网络。成功指回合结束时满足官方装配判据。',
      f"模型：`{lock['checkpoint']}`；SHA256：`{lock['checkpoint_sha256']}`。",'',
      '## 结论','',f"类别初筛排序首位为 **{LABELS[group['ranking'][0]]}**；在该类别中，单项初筛排序首位为 **{LABELS.get(factor['ranking'][0],factor['ranking'][0])}**。",
      f"独立确认样本的类别排序首位为 **{LABELS[confirm['group']['ranking'][0]]}**，入围单项排序首位为 **{LABELS.get(confirm['factor']['ranking'][0],confirm['factor']['ranking'][0])}**。",
      '只有独立确认的差值区间与排序共同支持时，才能说在本测试范围内存在明确的领先扰动。若区间跨零、差距小或排序翻转，应报告未区分清楚，而不是强行给出唯一最大贡献项。',
      '单项筛选只覆盖初筛选中类别；如果类别在追加样本后仍接近，不能宣称已找出所有类别中影响最大的单项。','',
      '## 实验口径','',
      '- 类别五组、单项18项；每个条件初始5 seeds × 128回合。接近时追加至10或15 seeds；入围前两名另用100–104五个未参与选择的seed确认。',
      '- 开启影响 = 固定环境成功率 − 仅开启该项的成功率；关闭影响 = 仅关闭该项的成功率 − 完整随机化成功率。敏感性分数为两者平均，可为负，不是可相加的因果贡献比例。',
      '- 前两名分数差不足5个百分点，或配对95%区间跨零时追加样本。到15 seeds仍模糊则保留不确定性，并按点估计首位开展探索性细分。',
      '- 95%区间采用配对分层bootstrap：先重采样seed，再在seed内重采样环境编号，对所有条件共同使用索引；2000次。自适应初筛区间是探索性区间，未做全体多重比较校正，独立确认可减少选择偏差。',
      '- 环境样本按扰动独立生成并落盘，关闭扰动不消耗/移动其他样本；IK重试按attempt、env_id索引，reset观测与正式rollout使用分离时间段。策略保留随机Normal采样与原裁剪/RNN逻辑，使用预生成的独立共同epsilon。',
      '- 不改变控制/奖励/成功判据的公式；控制参数扰动是被操纵变量。接触力阈值等固定值沿用零随机化版的范围中点。',
      '- 相同输入扰动不代表不同条件下产生相同状态/动作。开启随机化所产生的物理轨迹差异正是测量目标。',
      '- 评估seed不等于训练seed重复；本实验只针对一个已经训练好的checkpoint，不代表重训消融或真机结果。','']
    for kind,title in [('group','类别'),('factor','入围单项')]:
        c=confirm[kind]
        if c['clear_leader']:
            lines.append(f"独立确认：{title}中 {LABELS.get(c['ranking'][0],c['ranking'][0])} 明确领先另一入围候选，分数差 {effect(c['top_gap'])} pp。")
        else:
            lines.append(f"独立确认：{title}前两名尚不能按预设标准明确区分，分数差 {effect(c['top_gap'])} pp；不宣称唯一最大影响。")
    if confirm['group']['ranking'][0]!=group['ranking'][0]:
        lines.append('类别排序在确认集发生变化：已细分的类别不是确认集点估计第一名，因此单项结果仅是该已选类别内的结论。')
    lines.append('')
    lines+=table(group,'类别初筛（自动追加后）')+table(factor,'选中类别的单项初筛（自动追加后）')
    lines+=table(confirm['group'],'独立确认：入围类别')+table(confirm['factor'],'独立确认：入围单项')
    records=[]
    for cond in sorted((ROOT/'results').iterdir()):
        if not cond.is_dir():continue
        for p in sorted(cond.glob('seed_*/evaluation.json')):
            d=json.loads(p.read_text());seed=d['seed']
            for row in d['rows']:records.append({'condition':cond.name,'seed':seed,**row})
    with (ROOT/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines+=['## 失败表现诊断（初筛 seeds 0–14；各条件样本数见表）','',
      '| 条件 | N | 最终成功 | 曾成功但未保持 | 最终深度均值 mm | 横向误差均值 mm | 平均腕力 N | 每回合峰值腕力P95 N |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in sorted({r['condition'] for r in records}):
        rows=[r for r in records if r['condition']==name and r['seed']<100]
        if not rows:continue
        avg=lambda key:float(np.mean([r[key] for r in rows]))
        lost=sum(r['success_any'] and not r['success_final'] for r in rows)
        lines.append(f"| {name} | {len(rows)} | {pct(avg('success_final'))} | {lost} | {avg('final_insertion_depth_m')*1000:.2f} | {avg('final_xy_error_m')*1000:.2f} | {avg('force_mean_N'):.2f} | {np.percentile([r['force_max_N'] for r in rows],95):.2f} |")
    lines+=['','腕力是控制频率下平滑后的腕部力模长，不是原始高频接触峰值。完整逐回合数据见 episodes.csv；配对样本指纹、配置、命令和初始状态在每个结果目录中。',
      '所有“最敏感”结论都受当前扰动范围、固定值选择及扰动交互影响。单独开启与单独关闭结论不一致时，应保留背景条件，不将分数解释为独立贡献之和。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
