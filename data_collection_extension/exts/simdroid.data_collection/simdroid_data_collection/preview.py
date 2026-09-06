"""Session-only visual markers and physics-free gripper mesh copies."""
import uuid
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from .stage_store import NS, pose_matrix, world_pose


class PreviewManager:
    def __init__(self, stage):
        self.stage = stage
        # Adding/removing a session sublayer while simulating can invalidate all
        # PhysX tensor views. Own only a unique prim subtree in the existing
        # session layer instead; leave every other session opinion untouched.
        self.layer = stage.GetSessionLayer()
        self.root = "/__FrankaWaypointPreview_" + uuid.uuid4().hex[:8]
        self.waypoint = ""
        self.robot = None
        self.marker_paths = {}
        self.hidden = False

    def destroy(self):
        with Usd.EditContext(self.stage, self.layer):
            self.stage.RemovePrim(self.root)
        self.robot = None
        self.waypoint = ""
        self.marker_paths.clear()

    def selection_target(self, prim):
        while prim and str(prim.GetPath()).startswith(self.root):
            rel = prim.GetRelationship(NS + "previewTarget")
            if rel and rel.GetTargets():
                return str(rel.GetTargets()[0])
            prim = prim.GetParent()
        return None

    def set_hidden(self, hidden):
        self.hidden = hidden
        with Usd.EditContext(self.stage, self.layer):
            root = UsdGeom.Xform.Define(self.stage, self.root)
            root.CreateVisibilityAttr().Set("invisible" if hidden else "inherited")

    def hide_gripper(self):
        self.waypoint = ""
        self.robot = None
        with Usd.EditContext(self.stage, self.layer):
            self.stage.RemovePrim(self.root + "/Gripper")

    def update(self, store):
        if self.hidden:
            return
        with Usd.EditContext(self.stage, self.layer):
            UsdGeom.Xform.Define(self.stage, self.root)
            paths = store.paths() if store.sequence else []
            for missing in set(self.marker_paths) - set(paths):
                self.stage.RemovePrim(self.marker_paths.pop(missing))
            for i, path in enumerate(paths):
                try:
                    pose = world_pose(store.member(path))
                except (ValueError, RuntimeError):
                    old = self.marker_paths.pop(path, None)
                    if old:
                        self.stage.RemovePrim(old)
                    continue
                marker_path = self.marker_paths.get(path)
                if marker_path is None or not self.stage.GetPrimAtPath(marker_path):
                    marker_path = self.root + "/Markers/M_" + uuid.uuid4().hex[:8]
                    xf = UsdGeom.Xform.Define(self.stage, marker_path)
                    xf.GetPrim().CreateRelationship(NS + "previewTarget").SetTargets([path])
                    units = UsdGeom.GetStageMetersPerUnit(self.stage)
                    sphere = UsdGeom.Sphere.Define(self.stage, marker_path + "/Center")
                    sphere.CreateRadiusAttr(.012 / units)
                    sphere.CreateDisplayColorAttr([Gf.Vec3f(.1, .8, .9)])
                    for axis, color in zip("XYZ", ((1., .1, .1), (.1, 1., .1), (.1, .3, 1.))):
                        arrow = UsdGeom.Cone.Define(self.stage, marker_path + "/" + axis)
                        arrow.CreateAxisAttr(axis)
                        arrow.CreateHeightAttr(.06 / units)
                        arrow.CreateRadiusAttr(.004 / units)
                        arrow.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                        v = [0., 0., 0.]
                        v["XYZ".index(axis)] = .03 / units
                        arrow.AddTranslateOp().Set(Gf.Vec3d(*v))
                    self.marker_paths[path] = marker_path
                self._place(marker_path, pose)
            if self.waypoint:
                try:
                    prim = store.member(self.waypoint)
                    if not self.stage.GetPrimAtPath(self.root + "/Gripper"):
                        raise ValueError("Preview was deleted.")
                    self._place(self.root + "/Gripper", world_pose(prim).compose(self.robot.tcp.inverse()))
                except (ValueError, RuntimeError):
                    self.hide_gripper()

    def _place(self, path, pose):
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
        matrix = pose_matrix(pose, UsdGeom.GetStageMetersPerUnit(self.stage))
        ops = xf.GetOrderedXformOps()
        if not ops:
            xf.AddTransformOp().Set(matrix)
        elif ops[0].Get() != matrix:
            ops[0].Set(matrix)

    def show_gripper(self, robot, waypoint, opening):
        """opening is keep/open/close; copy only render geometry, no schemas/relations."""
        self.hide_gripper()
        cache = UsdGeom.XformCache()
        hand_world = cache.GetLocalToWorldTransform(robot.links["panda_hand"])
        hand_inverse = hand_world.GetInverse()
        with Usd.EditContext(self.stage, self.layer):
            root = UsdGeom.Xform.Define(self.stage, self.root + "/Gripper")
            root.GetPrim().CreateRelationship(NS + "previewTarget").SetTargets([waypoint.path])
            copied = 0
            for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
                link = robot.links[name]
                shift = Gf.Vec3d(0)
                if name != "panda_hand" and opening != "keep":
                    jname = "panda_finger_joint1" if name == "panda_leftfinger" else "panda_finger_joint2"
                    joint = UsdPhysics.PrismaticJoint(robot.links[jname])
                    body0 = self.stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
                    body1 = self.stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
                    frames = []
                    for body, pos, rot in ((body0, joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                                            (body1, joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())):
                        m = Gf.Matrix4d(1)
                        m.SetRotate(Gf.Quatd(rot.Get()))
                        m.SetTranslateOnly(Gf.Vec3d(pos.Get()))
                        frames.append(m * cache.GetLocalToWorldTransform(body))
                    axis = [0., 0., 0.]
                    axis["XYZ".index(str(joint.GetAxisAttr().Get()))] = 1.
                    direction = frames[0].TransformDir(Gf.Vec3d(*axis)).GetNormalized()
                    current = Gf.Dot(frames[1].ExtractTranslation()-frames[0].ExtractTranslation(), direction)
                    target = 0. if opening == "close" else min(.04/robot.units, joint.GetUpperLimitAttr().Get())
                    shift = hand_inverse.TransformDir(direction * (target-current))
                # Instance proxies expose the render mesh without de-instancing the source robot.
                for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                    if not p.IsA(UsdGeom.Mesh):
                        continue
                    path_lower = str(p.GetPath()).lower()
                    if "collision" in path_lower or "collider" in path_lower:
                        continue
                    if UsdGeom.Imageable(p).ComputeVisibility() == "invisible":
                        continue
                    mesh = UsdGeom.Mesh.Define(self.stage, self.root + f"/Gripper/Mesh_{copied:03d}")
                    # No source references, rigid-body APIs, joints, or material bindings are copied.
                    for key in ("points", "faceVertexCounts", "faceVertexIndices", "normals",
                                "subdivisionScheme", "orientation", "doubleSided", "holeIndices",
                                "creaseIndices", "creaseLengths", "creaseSharpnesses",
                                "cornerIndices", "cornerSharpnesses"):
                        a = p.GetAttribute(key)
                        if a and a.HasValue():
                            out = mesh.GetPrim().CreateAttribute(key, a.GetTypeName())
                            out.Set(a.Get())
                            if a.HasMetadata("interpolation"):
                                out.SetMetadata("interpolation", a.GetMetadata("interpolation"))
                    mesh.CreateDisplayColorAttr([Gf.Vec3f(.95, .55, .12)])
                    local = cache.GetLocalToWorldTransform(p) * hand_inverse
                    local.SetTranslateOnly(local.ExtractTranslation()+shift)
                    mesh.AddTransformOp().Set(local)
                    copied += 1
            if copied == 0:
                self.hide_gripper()
                raise ValueError("No visible hand/finger meshes found in this Panda asset.")
            self.robot, self.waypoint = robot, waypoint.path
            self._place(self.root + "/Gripper", waypoint.pose.compose(robot.tcp.inverse()))
        return copied
