"""Offline checks: exact complements, paired jobs, report and lifecycle without simulation."""
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import experiment as e

class StudyTest(unittest.TestCase):
    def test_plan(self):
        cs=e.conditions();jobs=e.plan()
        self.assertEqual(len(cs),24);self.assertEqual(len(jobs),480)
        self.assertEqual(len({(j['model'],j['condition'],j['seed']) for j in jobs}),480)
        for c,a in cs.items():
            if '_on_' in c:self.assertEqual(set(cs[c.replace('_on_','_off_')]),set(e.ALL)-set(a))
        for a,b in zip(jobs[::2],jobs[1::2]):
            self.assertEqual({k:v for k,v in a.items() if k!='model'},{k:v for k,v in b.items() if k!='model'})
        old={int(p.stem.split('seed')[1]) for p in (e.PARENT/'samples').glob('n128_seed*.npz')}
        self.assertFalse(old & set(sum(e.PHASES.values(),[])))

    def test_bootstrap(self):
        import numpy as np
        x=np.zeros((5,128,2));x[:,:,0]=1
        v=e.bootstrap_interval(x)
        self.assertEqual(v['estimate'],[1.,0.]);self.assertEqual(v['interval'],[[1.,0.],[1.,0.]])

    def test_pipeline_and_report(self):
        original=e.HERE
        lock=e.verify_lock()
        with tempfile.TemporaryDirectory() as tmp, patch.object(e,'HERE',Path(tmp)), patch.object(e.comparison,'ROOT',Path(tmp)), patch.object(e,'verify'), patch('samples.generate') as generate:
            def fake_generate(path,seed,n):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(str(seed).encode())
            generate.side_effect=fake_generate
            def fake_case(condition,active,seed,model):
                j=dict(condition=condition,model=model,seed=seed);d=e.jobdir(j);d.mkdir(parents=True,exist_ok=True)
                success=not (model=='zero' and condition not in ['zero'])
                rows=[dict(env_id=i,steps=149,success_final=success,success_any=True,final_insertion_depth_m=.046,final_xy_error_m=.001,force_mean_N=3.,force_max_N=5.) for i in range(128)]
                sample=e.digest(e.HERE/'samples'/f'n128_seed{seed}.npz')
                data=dict(seed=seed,episodes=128,rows=rows,policy_sample_count=149,active_factors=sorted(active),sample_sha256=sample)
                e.save(d/'evaluation.json',data)
                identity=dict(condition=condition,model=model,seed=seed,active=sorted(active),checkpoint_sha256=lock['models'][model]['checkpoint_sha256'],lock_sha256=e.digest(e.PARENT/'lock.json'),sample_sha256=sample)
                e.save(d/'done.json',dict(identity=identity,evaluation_sha256=e.digest(d/'evaluation.json'),seconds=1.))
            with patch.object(e,'run_case',side_effect=fake_case), patch('builtins.print'):
                e.pipeline()
            self.assertEqual(json.loads((e.HERE/'status.json').read_text())['status'],'complete')
            summary=json.loads((e.HERE/'summary.json').read_text())
            self.assertEqual(len(summary['confirmation']['factors']),9)
            self.assertEqual(summary['confirmation']['factors'][0]['statistics']['estimate'][6],1.)
            self.assertIn('独立确认',(e.HERE/'RESULTS.md').read_text())
            # Tampering must prevent a report, not silently use corrupted episodes.
            path=next((e.HERE/'results').glob('*/seed_*/evaluation.json'));path.write_text('{}')
            with self.assertRaises(AssertionError):e.report()
        self.assertEqual(e.HERE,original)

if __name__=='__main__':unittest.main()
