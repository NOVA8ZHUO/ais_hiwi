"""Build local bore variants from the existing CAD; never modify baseline assets."""
from pathlib import Path
import ast,hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
ASSETS=BASE/'assets/cad'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def points(text):
    matches=list(re.finditer(r'point3f\[\] points = (\[.*?\])',text))
    assert len(matches)==1,'Expected a single flattened mesh'
    m=matches[0];return m,np.array(ast.literal_eval(m.group(1)))

def main():
    out=ROOT/'assets';out.mkdir(exist_ok=True)
    src=ASSETS/'base.usda';text=src.read_text();m,a=points(text)
    _,shaft=points((ASSETS/'shaft_large.usda').read_text())
    r=np.linalg.norm(a[:,:2],axis=1);sr=np.linalg.norm(shaft[:,:2],axis=1)
    assert np.count_nonzero(abs(r-.013)<1e-7)>100
    assert np.count_nonzero(abs(sr-.0125)<1e-7)>100
    mask=(r<=.0150001)&(r>=.0129999)
    assert np.all(r>=.0129999), 'Unexpected central geometry'
    record={'baseline_sha256':{str(p):sha(p) for p in [src,ASSETS/'shaft_large.usda']},'shaft_diameter_m':.025,'nominal_bore_diameter_m':.026,'nominal_diametral_clearance_m':.001,'variants':[]}
    for cmm in [.25,.5,1.,2.]:
        path=out/f'base_clearance_{cmm:g}mm.usda'
        delta=(cmm/1000-.001)/2;b=a.copy()
        b[mask,:2]*=((r[mask]+delta)/r[mask])[:,None]
        if delta==0:path.write_text(text)
        else:
            string='['+', '.join('('+', '.join(format(float(v),'.10g') for v in row)+')' for row in b)+']'
            path.write_text(text[:m.start(1)]+string+text[m.end(1):])
        _,actual=points(path.read_text());rr=np.linalg.norm(actual[:,:2],axis=1)
        assert np.allclose(actual[~mask],a[~mask],atol=1e-10)
        assert np.allclose(actual[:,2],a[:,2],atol=1e-10)
        inner=abs(r-.013)<1e-7
        measured=2*np.median(rr[inner])-.025
        assert abs(measured-cmm/1000)<1e-7
        record['variants'].append({'path':str(path),'clearance_mm':cmm,'measured_clearance_mm':measured*1000,'bore_vertices_modified':int(mask.sum()) if delta else 0,'outer_vertices_unchanged':True,'z_unchanged':True,'sha256':sha(path)})
    (ROOT/'geometry_audit.json').write_text(json.dumps(record,indent=2))
    # Reuse the existing acceptance procedure, with an explicit new-config diff.
    original=(BASE/'scripts/validate_cad.py').read_text()
    original=original.replace("p.add_argument('--seed'", "p.add_argument('--clearance',type=float,required=True)\np.add_argument('--pose',type=int,choices=[0,1],required=True)\np.add_argument('--seed'")
    original=original.replace('    cfg = ForgeCadCfg()','''    cfg = ForgeCadCfg()
    baseline=cfg.to_dict()
    study_root=Path(__file__).resolve().parent
    cfg.task.fixed_asset_cfg.usd_path=str(study_root/'assets'/f'base_clearance_{a.clearance:g}mm.usda')
    cfg.task.fixed_asset.spawn.usd_path=cfg.task.fixed_asset_cfg.usd_path
    cfg.task.fixed_asset_cfg.diameter=round(.025+a.clearance/1000,9)
    cfg.obs_rand.fixed_asset_pos=[.001*a.pose,.001*a.pose,.001]
    report['pose_xy_sigma_m']=.001*a.pose
    report['clearance_mm']=a.clearance''')
    original=original.replace('    delta = differences(official, cfg.to_dict())',"    study_diff=differences(baseline,cfg.to_dict())\n    permitted={'task.fixed_asset_cfg.usd_path','task.fixed_asset.spawn.usd_path','task.fixed_asset_cfg.diameter','obs_rand.fixed_asset_pos'}\n    assert all(d['path'] in permitted for d in study_diff),study_diff\n    assert ForgeCadCfg().to_dict()==baseline,'Mutated baseline defaults'\n    report['full_randomized_baseline_diff']=study_diff\n    delta = differences(official, cfg.to_dict())")
    original=original.replace("'task.fixed_asset.spawn.usd_path','task.hand_init_pos'}", "'task.fixed_asset.spawn.usd_path','task.hand_init_pos','obs_rand.fixed_asset_pos'}")
    original=original.replace('for offset in [0.0, .025]:','for offset in [0.0, .025, a.clearance/4000, a.clearance/2000+.0005]:')
    original=original.replace("    report['status'] = 'passed'", """    report['small_clearance_contact_check']={
        'free_offset_m':probe[2]['lateral_offset_m'],
        'free_displacement_max_m':max(probe[2]['displacement_m']),
        'overlap_offset_m':probe[3]['lateral_offset_m'],
        'overlap_displacement_min_m':min(probe[3]['displacement_m']),
        'note':'20-step dynamic probe; does not certify machining clearance or SDF error bound'}
    assert max(probe[2]['displacement_m']) < .0001, ('Nominally free near-wall pose moves >0.1mm; investigate SDF/contact offsets',probe)
    assert min(probe[3]['displacement_m']) > .0001, ('Near-wall penetration not resolved >0.1mm',probe)
    report['status'] = 'passed'""")
    original=original.replace('        record = snapshot()', """        record = snapshot()
        bias=r.init_fixed_pos_obs_noise.clone()
        record['hole_estimate_bias_m']=bias.cpu().tolist()
        assert torch.isfinite(bias).all()
        if a.pose==0:assert torch.count_nonzero(bias[:,:2])==0
        else:assert torch.count_nonzero(bias[:,:2])>0
        report['checks']['pose_bias_runtime']=True""")
    (ROOT/'validate_runtime.py').write_text(original)
    # Static source checks explicitly document both consumers of the same estimate.
    forge=BASE.parent/'IsaacLab_2_3_2/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py'
    code=forge.read_text()
    checks={x:x in code for x in ['noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise','fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise']}
    assert all(checks.values())
    (ROOT/'pose_path_audit.json').write_text(json.dumps({'source':str(forge),'sha256':sha(forge),'checks':checks,'scope':'Static path check; dynamic reset samples remain native FORGE sampling.'},indent=2))
    print('Geometry variants and pose paths validated; runtime validation required.')
if __name__=='__main__':main()
