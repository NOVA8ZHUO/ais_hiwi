"""Run grouped on/off screening, adaptive replication, item drilldown, held-out confirmation."""
import argparse,fcntl,json,time,traceback
from pathlib import Path
from design import ALL,GROUPS,INITIAL_SEEDS,MAX_SCREEN_SEEDS,CONFIRM_SEEDS
from samples import generate
from runner import ROOT,run_case,verify_lock,require_preflight
from paired_stats import compare

def state(**kwargs):
    p=ROOT/'status.json';d=json.loads(p.read_text()) if p.exists() else {}
    d.update(kwargs,updated_unix=time.time());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def batch(items,prefix,seeds):
    for seed in seeds:generate(ROOT/'samples'/f'n128_seed{seed}.npz',seed)
    conditions=[('zero',[]),('full',ALL)]
    for name,factors in items.items():
        conditions += [(prefix+'_on_'+name,factors),(prefix+'_off_'+name,[f for f in ALL if f not in factors])]
    for seed in seeds:
        for condition,active in conditions:
            state(condition=condition,seed=seed)
            run_case(condition,active,seed)

def screen(items,prefix):
    seeds=list(INITIAL_SEEDS)
    while True:
        batch(items,prefix,seeds)
        summary=compare(list(items),seeds,prefix)
        (ROOT/(prefix+'_screen.json')).write_text(json.dumps(summary,indent=2))
        if summary['clear_leader'] or len(seeds)>=MAX_SCREEN_SEEDS:return summary
        seeds=list(range(min(MAX_SCREEN_SEEDS,len(seeds)+5)))
        state(note='Top effects close; expand paired screening seeds',screen_seed_count=len(seeds))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true',help='Completed cases are always validated and reused')
    a=p.parse_args()
    lockfile=(ROOT/'pipeline.lock').open('w')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_lock();require_preflight()
    started=time.time()
    try:
        state(status='running',stage='group_screen',model='zero')
        groups=screen(GROUPS,'group');winner=groups['ranking'][0]
        state(stage='factor_screen',selected_group=winner,category_leader_clear=groups['clear_leader'])
        factors={f:[f] for f in GROUPS[winner]}
        singles=screen(factors,'factor')
        state(stage='held_out_confirmation')
        group_finalists={k:GROUPS[k] for k in groups['ranking'][:2]}
        factor_finalists={k:[k] for k in singles['ranking'][:2]}
        batch(group_finalists,'group',CONFIRM_SEEDS)
        batch(factor_finalists,'factor',CONFIRM_SEEDS)
        confirmation={'group':compare(list(group_finalists),CONFIRM_SEEDS,'group'),
                      'factor':compare(list(factor_finalists),CONFIRM_SEEDS,'factor')}
        (ROOT/'confirmation.json').write_text(json.dumps(confirmation,indent=2))
        from report import write_report
        write_report()
        from model_comparison import replay_randomized,write_comparison
        state(stage='randomized_model_replay',model='randomized')
        replay_randomized(state)
        state(stage='model_comparison_report')
        write_comparison()
        state(status='complete',stage='complete',comparison_report=str(ROOT/'MODEL_COMPARISON.md'),elapsed_this_invocation_s=time.time()-started,report=str(ROOT/'RESULTS.md'))
    except BaseException:
        state(status='failed',error=traceback.format_exc());raise
