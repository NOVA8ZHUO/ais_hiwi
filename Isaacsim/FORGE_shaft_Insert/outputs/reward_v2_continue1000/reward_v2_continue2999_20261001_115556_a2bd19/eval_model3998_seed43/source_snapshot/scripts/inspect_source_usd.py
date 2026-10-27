# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Print the source USD prims and geometry values used by the task configuration."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


USD_PATH = "/home/masais/Xinyu/USD_files/shaft_insertion.usd"


def main() -> None:
    stage = Usd.Stage.Open(USD_PATH)
    print(f"file={USD_PATH}")
    print(f"defaultPrim={stage.GetDefaultPrim()} upAxis={stage.GetMetadata('upAxis')}")
    print("topLevel=", [str(prim.GetPath()) for prim in stage.GetPseudoRoot().GetChildren()])
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for path in ("/Franka", "/Parts/Shaft_small", "/Parts/Base"):
        prim = stage.GetPrimAtPath(path)
        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        print(
            f"{path}: type={prim.GetTypeName()} rigid={prim.HasAPI(UsdPhysics.RigidBodyAPI)} "
            f"articulation={prim.HasAPI(UsdPhysics.ArticulationRootAPI)} "
            f"bbox_min={bbox.GetMin()} bbox_max={bbox.GetMax()}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

