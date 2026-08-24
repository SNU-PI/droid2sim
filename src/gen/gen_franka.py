"""Emit a zero-DoF Franka as static geoms, posed by forward kinematics.

The arm is scene context, not the actuator. Stage 0 found the frozen encoder
almost blind to the physics sweep, but the scene had no robot in it at all,
while V-JEPA 2-AC was post-trained on DROID where a Franka is always in frame.
That domain gap has to be ruled out before the insensitivity can be blamed on
the representation.

So the arm is added as appearance ONLY: its links are baked into fixed geoms at
a chosen joint configuration, with collision disabled. No new joints, no new
bodies with mass, nothing that touches the dynamics -- the physics must stay
bit-for-bit identical to the version that passed all eight checks, and there is
a test asserting exactly that.
"""

import sys
import numpy as np
import mujoco

MEN = "/data/pgc/simdroid/stage0/mujoco_menagerie/franka_emika_panda"
# a plausible DROID-like reaching pose; the gripper ends up near the pusher
import os as _os, json as _json
_CFG = _json.loads(_os.environ.get("FRANKA_CFG", "{}"))
QPOS = _CFG.get("qpos", [0.0, -0.35, 0.0, -2.55, 0.0, 2.25, 0.785])
BASE_POS = tuple(_CFG.get("base", (0.0, -0.62, 0.40)))
BASE_QUAT = tuple(_CFG.get("quat", (1.0, 0.0, 0.0, 0.0)))
OUTTAG = _CFG.get("tag", "")


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def main():
    m = mujoco.MjModel.from_xml_path(f"{MEN}/panda.xml")
    d = mujoco.MjData(m)
    nq = min(len(QPOS), m.nq)
    d.qpos[:nq] = QPOS[:nq]
    mujoco.mj_forward(m, d)

    meshes, geoms = {}, []
    bq = np.array(BASE_QUAT)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        # group 2 is menagerie's visual layer; group 3 is collision geometry
        # whose .stl files are not all shipped and which we do not want anyway
        if int(m.geom_group[g]) != 2:
            continue
        mid = m.geom_dataid[g]
        mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        if mname is None:
            continue
        # world pose of the geom in the panda's own frame
        p = np.array(d.geom_xpos[g])
        R = np.array(d.geom_xmat[g]).reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        # rotate/translate into the scene frame
        wp = np.array(BASE_POS) + p
        wq = quat_mul(bq, q)
        rgba = m.geom_rgba[g]
        meshes[mname] = True
        geoms.append((mname, wp, wq, rgba))

    asset = "\n".join(
        f'    <mesh name="fr_{n}" file="{MEN}/assets/{n}.obj"/>'
        for n in sorted(meshes))
    body = "\n".join(
        f'      <geom type="mesh" mesh="fr_{n}" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
        f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
        f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}" '
        f'contype="0" conaffinity="0" group="0"/>'
        for n, p, q, c in geoms)

    print(f"<!-- {len(geoms)} static Franka geoms, {len(meshes)} meshes -->",
          file=sys.stderr)
    with open(os.path.join(sys_root, "core", f"franka_asset{OUTTAG}.xml"), "w") as f:
        f.write(asset + "\n")
    with open(os.path.join(sys_root, "core", f"franka_body{OUTTAG}.xml"), "w") as f:
        f.write(body + "\n")
    print(f"wrote src/franka_asset.xml ({len(meshes)} meshes) and "
          f"src/franka_body.xml ({len(geoms)} geoms)", file=sys.stderr)


if __name__ == "__main__":
    main()
