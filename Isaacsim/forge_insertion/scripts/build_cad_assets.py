"""Extract only CAD meshes and normalize frames for the official FORGE interface.
Run with the existing Isaac Sim USD libraries; does not import the legacy task.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, Vt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'FORGE_shaft_Insert/assets'
OUT = ROOT / 'assets/cad'
OUT.mkdir(parents=True, exist_ok=True)
manifest = {'source': str(SOURCE), 'shaft_nominal_mass_kg': 0.0175,
            'shaft_length_m': 0.083, 'grasp_diameter_m': 0.035,
            'insertion_diameter_m': 0.025, 'seating_depth_m': 0.047,
            'base_target_in_original_frame_m': [0.127, -0.00018226, 0.003], 'assets': {}}
for name, mass, shift, fixed in [
    ('shaft_large', 0.0175, [0, 0, 0], False),
    ('base', 0.05, [0.127, -0.00018226, 0.003], True),
]:
    source = Usd.Stage.Open(str(SOURCE / (name + '.usda')))
    source_root = source.GetDefaultPrim()
    mesh_prim = next(p for p in source.Traverse() if p.IsA(UsdGeom.Mesh))
    source_mesh = UsdGeom.Mesh(mesh_prim)
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(mesh_prim, source_root)
    points = np.array([transform.Transform(Gf.Vec3d(*v)) for v in source_mesh.GetPointsAttr().Get()])
    points -= np.asarray(shift)
    path = OUT / (name + '.usda')
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Asset')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if fixed:
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, '/Asset/Body')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    if not fixed:
        UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    mesh = UsdGeom.Mesh.Define(stage, '/Asset/Body/Geometry')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get())
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.58, 0.62) if not fixed else Gf.Vec3f(0.18, 0.24, 0.32)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('sdf')
    # Retain authored SDF/collision cooking parameters, not legacy body dynamics/materials.
    # Offline USD may not load PhysX plugins: preserve raw authored API tokens.
    schemas = []
    for spec in mesh_prim.GetPrimStack():
        if spec.HasInfo('apiSchemas'):
            schemas.extend(str(x) for x in spec.GetInfo('apiSchemas').GetAddedOrExplicitItems() if str(x).startswith('Physx'))
    schemas = list(dict.fromkeys(schemas + ['PhysxSDFMeshCollisionAPI']))
    mesh.GetPrim().SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(
        ['PhysicsCollisionAPI','PhysicsMeshCollisionAPI'] + schemas))
    for attr in mesh_prim.GetAttributes():
        if attr.GetName().startswith(('physxSDFMeshCollision:', 'physxCollision:', 'physxMeshCollision:')) and attr.HasAuthoredValueOpinion():
            mesh.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
    if fixed:
        joint = UsdPhysics.FixedJoint.Define(stage, '/Asset/FixedJoint')
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
    stage.GetRootLayer().Save()
    reconstructed = points + np.asarray(shift)
    originals = np.array([transform.Transform(Gf.Vec3d(*v)) for v in source_mesh.GetPointsAttr().Get()])
    assert np.allclose(reconstructed, originals, atol=1e-9)
    manifest['assets'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'vertices': len(points), 'bounds_min_m': points.min(0).tolist(), 'bounds_max_m': points.max(0).tolist(),
        'source_layers': {str(l.realPath): hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
                          for l in source.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()},
        'fixed_to_world': fixed, 'collision': 'sdf', 'mass_kg': mass}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
