import torch

def mass_event(env,env_ids):
    asset=env.scene['held_asset'];ids=env_ids.cpu()
    masses=asset.root_physx_view.get_masses()
    default=asset.data.default_mass.cpu()
    delta=env.tape.tensor('mass')*.010-.005 if 'mass' in env.active else torch.zeros_like(default)
    masses[ids]=default[ids]+delta[ids]
    asset.root_physx_view.set_masses(masses,ids)
    inertias=asset.root_physx_view.get_inertias()
    inertias[ids]=asset.data.default_inertia.cpu()[ids]*(masses[ids]/default[ids])[...,None]
    asset.root_physx_view.set_inertias(inertias,ids)

def friction_event(env,env_ids):
    asset=env.scene['fixed_asset']
    ids=torch.arange(env.num_envs,device='cpu') if env_ids is None else env_ids.cpu()
    materials=asset.root_physx_view.get_material_properties();shapes=materials.shape[1]
    assert shapes<=16,'Expand pre-generated material shape tape'
    # Same official 128 uniform material buckets, assigned independently per shape.
    buckets=env.tape.tensor('friction_buckets').clone()
    buckets[:,0]=.25+buckets[:,0] if 'friction' in env.active else .75
    buckets[:,1]=.25;buckets[:,2]=0.
    assignments=(env.tape.tensor('friction_ids')[:,:shapes]*128).long()
    materials[ids]=buckets[assignments[ids]]
    asset.root_physx_view.set_material_properties(materials,ids)

def dead_zone_event(env,env_ids):
    # Official fixed 2-second interval, indexed by physical episode time, not RNG calls.
    slot=min(7,1+max(0,env.eval_step)//30)
    env.dead_zone_thresholds=(env.tape.tensor('dead_zone',env.device)[slot] if 'dead_zone' in env.active else .5)*env.default_dead_zone
