import json,tempfile,unittest
from pathlib import Path
import numpy as np
import samples,paired_stats,report

class PairingTests(unittest.TestCase):
    def test_samples_named_and_replayable(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.npz';b=Path(folder)/'b.npz';c=Path(folder)/'c.npz'
            samples.generate(a,7,4);samples.generate(b,7,4);samples.generate(c,8,4)
            x,y,z=samples.Tape(a),samples.Tape(b),samples.Tape(c)
            for key in x.data:
                np.testing.assert_array_equal(x.data[key],y.data[key])
                self.assertFalse(np.array_equal(x.data[key],z.data[key]))
            before=x.data['mass'].copy()
            _=x.data['policy'][5];_=x.data['base_position']
            np.testing.assert_array_equal(before,x.data['mass'])
            a.write_bytes(a.read_bytes()+b'corruption')
            with self.assertRaises(AssertionError):samples.Tape(a)

    def test_paired_contrasts_and_report(self):
        old_stats,old_report=paired_stats.ROOT,report.ROOT
        try:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);paired_stats.ROOT=report.ROOT=root
                seeds=list(range(5))
                conditions={'zero':1.,'full':0.,'group_on_pose':0.,'group_off_pose':1.,
                    'group_on_physics':.5,'group_off_physics':.5}
                for name,prob in conditions.items():
                    for seed in seeds:
                        p=root/'results'/name/f'seed_{seed}';p.mkdir(parents=True)
                        rows=[{'env_id':i,'success_final':bool(i<int(4*prob)),'success_any':True,
                          'final_insertion_depth_m':.047,'final_xy_error_m':.0001,'force_mean_N':2.,'force_max_N':4.} for i in range(4)]
                        (p/'evaluation.json').write_text(json.dumps({'seed':seed,'rows':rows}))
                d=paired_stats.compare(['pose','physics'],seeds,'group',replicates=200)
                self.assertEqual(d['ranking'][0],'pose')
                self.assertEqual(d['items'][0]['score']['estimate'],1.)
                self.assertEqual(d['items'][1]['score']['estimate'],.5)
                self.assertEqual(d['items'][0]['enable_drop']['ci95'],[1.,1.])
                for name in ['group_screen','factor_screen']:(root/(name+'.json')).write_text(json.dumps(d))
                (root/'confirmation.json').write_text(json.dumps({'group':d,'factor':d}))
                (root/'lock.json').write_text(json.dumps({'checkpoint':'TEST_FIXTURE_ONLY','checkpoint_sha256':'TEST'}))
                report.write_report()
                self.assertIn('独立确认',(root/'RESULTS.md').read_text())
                self.assertTrue((root/'episodes.csv').exists())
        finally:paired_stats.ROOT=old_stats;report.ROOT=old_report

if __name__=='__main__':unittest.main()
