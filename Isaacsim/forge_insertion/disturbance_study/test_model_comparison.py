import json,tempfile,unittest
from pathlib import Path
import numpy as np
import model_comparison as m
from runner import digest

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.old=m.ROOT;m.ROOT=Path(self.tmp.name)
        plan=[]
        conditions={'zero':1.,'full':0.,'group_on_pose':0.,'group_off_pose':1.}
        for condition,rate in conditions.items():
            for seed in [0,1,2,3,4,100,101,102,103,104]:
                plan.append({'condition':condition,'seed':seed,'n':4,'active':[]})
                for model,folder in [('zero','results'),('randomized','results_randomized')]:
                    run=m.ROOT/folder/condition/f'seed_{seed}';run.mkdir(parents=True)
                    rows=[{'env_id':i,'success_final':bool(i<4*rate) if model=='zero' else True,
                           'success_any':True,'final_insertion_depth_m':.047,'final_xy_error_m':.001,
                           'force_mean_N':2.,'force_max_N':3.} for i in range(4)]
                    data={'seed':seed,'sample_sha256':'paired_fixture','active_factors':[],
                          'policy_sample_count':149,'episodes':4,'rows':rows}
                    path=run/'evaluation.json';path.write_text(json.dumps(data))
                    (run/'done.json').write_text(json.dumps({'identity':{'model':model,'condition':condition,'sample_sha256':'paired_fixture'},'evaluation_sha256':digest(path)}))
        (m.ROOT/'randomized_replay_plan.json').write_text(json.dumps(plan))
        (m.ROOT/'RESULTS.md').write_text('# Fixture report\n')
    def tearDown(self):m.ROOT=self.old;self.tmp.cleanup()
    def test_known_gain_and_sensitivity(self):
        d=m.condition_stats('full',[0,1,2,3,4]);self.assertEqual(d['success_gain']['estimate'],1.)
        self.assertEqual(d['success_gain']['ci95'],[1.,1.])
        d=m.sensitivity('group','pose',[0,1,2,3,4]);np.testing.assert_allclose(d['improvement']['estimate'],[1.,1.,1.])
        # At a shared ceiling, raw benefit is zero; do not invent a benefit.
        self.assertEqual(m.condition_stats('zero',[0,1,2,3,4])['success_gain']['estimate'],0.)
    def test_reject_mismatched_samples(self):
        p=m.ROOT/'results_randomized/full/seed_0/evaluation.json';d=json.loads(p.read_text())
        d['sample_sha256']='wrong';p.write_text(json.dumps(d))
        done=json.loads(p.with_name('done.json').read_text());done['evaluation_sha256']=digest(p)
        done['identity']['sample_sha256']='wrong';p.with_name('done.json').write_text(json.dumps(done))
        with self.assertRaises(AssertionError):m.paired('full',[0])
    def test_replay_exact_first_model_job_set(self):
        from unittest.mock import patch
        lock={'models':{'zero':{'checkpoint_sha256':'zero_checkpoint'}}}
        (m.ROOT/'lock.json').write_text(json.dumps(lock))
        for path in (m.ROOT/'results').glob('*/seed_*/done.json'):
            d=json.loads(path.read_text());d['identity'].update(checkpoint_sha256='zero_checkpoint',
                lock_sha256=digest(m.ROOT/'lock.json'),seed=int(path.parent.name.split('_')[1]),num_envs=4,active=[])
            path.write_text(json.dumps(d))
        (m.ROOT/'randomized_replay_plan.json').unlink()
        with patch.object(m,'verify_lock',return_value=lock),patch.object(m,'run_case') as run:
            m.replay_randomized(lambda **kwargs:None)
            self.assertEqual(run.call_count,40)
            self.assertTrue(all(c.kwargs['model']=='randomized' and c.kwargs['n']==4 for c in run.call_args_list))
        plan=json.loads((m.ROOT/'randomized_replay_plan.json').read_text())
        self.assertEqual(len(plan),40)
        self.assertEqual({j['seed'] for j in plan},{0,1,2,3,4,100,101,102,103,104})

    def test_report_and_idempotent_append(self):
        m.write_comparison();m.write_comparison()
        self.assertTrue((m.ROOT/'MODEL_COMPARISON.md').exists())
        self.assertTrue((m.ROOT/'model_comparison_episodes.csv').exists())
        self.assertEqual((m.ROOT/'RESULTS.md').read_text().count('## 两模型抗扰能力对比'),1)

if __name__=='__main__':unittest.main()
