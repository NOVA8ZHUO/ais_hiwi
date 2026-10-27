from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
from pxr import Usd, UsdGeom, Gf
import numpy as np
stage = Usd.Stage.Open('/home/masais/Xinyu/USD_files/shaft_insertion.usd')
cache = UsdGeom.XformCache()
for path, center in [('/Parts/Shaft_large', (0.5,0,0.15)), ('/Parts/Base',(0.627,-0.00018226745,0.05))]:
    chunks=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.IsA(UsdGeom.Mesh):
            transform = cache.GetLocalToWorldTransform(prim)
            chunks.extend([tuple(transform.Transform(Gf.Vec3d(*p))) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    a=np.array(chunks)-np.array(center)
    radius=np.linalg.norm(a[:,:2],axis=1)
    levels=np.round(a[:,2],5)
    for z in sorted(set(levels)):
        r=radius[levels==z]
        if path.endswith('Base'):
            r=r[r<.025]
        if len(r):
            print('SECTION',path,'z',round(float(z),5),'r_min_max',round(float(r.min()),5),round(float(r.max()),5),'n',len(r),flush=True)
app.close()
