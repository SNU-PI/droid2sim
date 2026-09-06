"""USD-backed authoring. Only explicit sequence membership makes a goal runnable."""
from pathlib import Path
import uuid
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import omni.kit.commands
from .domain import Pose, Waypoint

NS = "frankaPath:"
DEFAULT_ROOT = "/World/FrankaPaths"
DEFAULT_TCP = Pose((0, 0, .107), (1, 0, 0, 0))


def attr(prim, name, default=None):
    a = prim.GetAttribute(NS + name)
    value = a.Get() if a else None
    return default if value is None else value


def author(prim, name, kind, value):
    prim.CreateAttribute(NS + name, kind, custom=True).Set(value)


def tagged(prim, kind):
    return bool(prim and attr(prim, "kind") == kind and attr(prim, "version") == 1)


def matrix_pose(matrix, meters_per_unit=1.):
    # Scaling/shearing a robot/target invalidates a rigid kinematic transform.
    linear = np.asarray(matrix)[:3, :3]
    if not np.allclose(linear @ linear.T, np.eye(3), atol=1e-4) or np.linalg.det(linear) < 0:
        raise ValueError("Robot/waypoint transforms must have unit scale and no shear/reflection.")
    q = matrix.ExtractRotationQuat()
    return Pose(np.asarray(matrix.ExtractTranslation()) * meters_per_unit,
                (q.GetReal(), *q.GetImaginary()))


def pose_matrix(pose, meters_per_unit=1.):
    q = pose.orientation
    m = Gf.Matrix4d(1)
    m.SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:])))
    m.SetTranslateOnly(Gf.Vec3d(*(np.asarray(pose.position) / meters_per_unit)))
    return m


def world_pose(prim):
    return matrix_pose(UsdGeom.XformCache().GetLocalToWorldTransform(prim),
                       UsdGeom.GetStageMetersPerUnit(prim.GetStage()))


def set_world_pose(prim, pose):
    stage = prim.GetStage()
    cache = UsdGeom.XformCache()
    xf = UsdGeom.Xformable(prim)
    reset_stack = xf.GetResetXformStack()
    parent_world = Gf.Matrix4d(1) if reset_stack else cache.GetParentToWorldTransform(prim)
    matrix_pose(parent_world)  # reject unsupported parent scale before writing
    local = pose_matrix(pose, UsdGeom.GetStageMetersPerUnit(stage)) * parent_world.GetInverse()
    op = xf.MakeMatrixXform()
    xf.SetResetXformStack(reset_stack)
    op.Set(local)


class FrankaWaypointLayerEdit(omni.kit.commands.Command):
    """Undo only the small authoring layer, never snapshot the large scene."""
    def __init__(self, store, edit):
        self.store, self.edit = store, edit
        self.before = self.after = None

    @property
    def layer(self):
        # Saving an anonymous layer replaces its identity, but not undo history.
        return self.store._layer()

    def do(self):
        if self.after is not None:
            self.layer.ImportFromString(self.after)
            return
        self.before = self.layer.ExportToString()
        try:
            # USD queries in this edit require immediately composed results.
            self.edit()
        except Exception:
            self.layer.ImportFromString(self.before)
            raise
        self.after = self.layer.ExportToString()

    def undo(self):
        self.layer.ImportFromString(self.before)


class StageStore:
    def __init__(self, stage):
        self.stage = stage
        self.layer = next((l for l in stage.GetLayerStack()
                           if l.customLayerData.get("frankaPathAuthoring") == 1), None)
        self.sequence = ""

    def sequences(self):
        root = self.stage.GetPrimAtPath(DEFAULT_ROOT)
        return [str(p.GetPath()) for p in Usd.PrimRange(root) if tagged(p, "sequence")] if root else []

    def sequence_for_waypoint(self, path):
        for sequence in self.sequences():
            rel = self.stage.GetPrimAtPath(sequence).GetRelationship(NS + "waypoints")
            if rel and Sdf.Path(path) in rel.GetTargets():
                return sequence
        return ""

    def validate_conventions(self, layer=None):
        layer = layer or self.layer
        if layer is None:
            return
        units = layer.customLayerData.get("frankaPathMetersPerUnit", 1.)
        up = layer.customLayerData.get("frankaPathUpAxis")
        current = UsdGeom.GetStageMetersPerUnit(self.stage)
        if not np.isfinite(current) or current <= 0 or not np.isclose(units, current) or (
                up and up != str(UsdGeom.GetStageUpAxis(self.stage))):
            raise ValueError("Waypoint layer and scene must have matching units and up axis.")

    def _layer(self):
        stack = self.stage.GetLayerStack()
        if self.layer not in stack:
            candidates = [layer for layer in stack if layer.customLayerData.get("frankaPathAuthoring") == 1]
            if len(candidates) > 1:
                raise ValueError("Multiple waypoint authoring layers are attached; use one per scene.")
            if candidates:
                self.layer = candidates[0]
            elif self.layer is not None:
                raise ValueError("The waypoint authoring layer was detached; reload it before editing.")
        if self.layer is None:
            self.layer = Sdf.Layer.CreateAnonymous("franka_paths.usda")
            self.layer.customLayerData = {
                "frankaPathAuthoring": 1,
                "frankaPathMetersPerUnit": UsdGeom.GetStageMetersPerUnit(self.stage),
                "frankaPathUpAxis": str(UsdGeom.GetStageUpAxis(self.stage)),
            }
            self.stage.GetRootLayer().subLayerPaths.insert(0, self.layer.identifier)
        self.validate_conventions()
        return self.layer

    def edit(self, fn):
        layer = self._layer()
        def scoped():
            with Usd.EditContext(self.stage, layer):
                fn()
        ok, _ = omni.kit.commands.execute("FrankaWaypointLayerEdit", store=self, edit=scoped)
        if not ok:
            raise RuntimeError("USD waypoint edit failed; see Isaac Sim's log.")

    def create_sequence(self, robot_path=""):
        root = self.stage.GetPrimAtPath(DEFAULT_ROOT)
        if root and not tagged(root, "collection"):
            raise ValueError(f"{DEFAULT_ROOT} is already occupied by an unrelated prim.")
        path = self.unique(DEFAULT_ROOT + "/Sequence")
        def create():
            # A saved layer must remain traversable even in a blank stage. Leave
            # existing /World's type and transform untouched in the source scene.
            Sdf.CreatePrimInLayer(self._layer(), "/World").specifier = Sdf.SpecifierDef
            root = UsdGeom.Scope.Define(self.stage, DEFAULT_ROOT).GetPrim()
            for p, kind in ((root, "collection"),
                            (UsdGeom.Xform.Define(self.stage, path).GetPrim(), "sequence")):
                author(p, "kind", Sdf.ValueTypeNames.Token, kind)
                author(p, "version", Sdf.ValueTypeNames.Int, 1)
            seq = self.stage.GetPrimAtPath(path)
            seq.CreateRelationship(NS + "waypoints", custom=True).SetTargets([])
            seq.CreateRelationship(NS + "robot", custom=True).SetTargets(
                [Sdf.Path(robot_path)] if robot_path else [])
            author(seq, "tcpPosition", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*DEFAULT_TCP.position))
            author(seq, "tcpOrientation", Sdf.ValueTypeNames.Quatd, Gf.Quatd(1))
        self.edit(create)
        self.sequence = path
        return path

    def unique(self, base):
        path, index = base, 1
        while self.stage.GetPrimAtPath(path):
            path, index = f"{base}_{index:03d}", index + 1
        return path

    def seq(self):
        self.validate_conventions()
        if not self.sequence:
            raise ValueError("Create or select a waypoint sequence first.")
        prim = self.stage.GetPrimAtPath(self.sequence)
        if not tagged(prim, "sequence") or not prim.IsActive():
            raise ValueError("Create or select a waypoint sequence first.")
        return prim

    def paths(self):
        rel = self.seq().GetRelationship(NS + "waypoints")
        return [str(p) for p in rel.GetTargets()] if rel else []

    def robot_path(self):
        rel = self.seq().GetRelationship(NS + "robot")
        targets = rel.GetTargets() if rel else []
        return str(targets[0]) if targets else ""

    def tcp(self):
        p = self.seq()
        q = attr(p, "tcpOrientation", Gf.Quatd(1))
        return Pose(attr(p, "tcpPosition", DEFAULT_TCP.position), (q.GetReal(), *q.GetImaginary()))

    def bind(self, robot_path, tcp):
        if not self.stage.GetPrimAtPath(robot_path):
            raise ValueError("Selected robot no longer exists.")
        if self.robot_path() == robot_path and self.tcp() == tcp:
            # Reinitializing physics is not a USD configuration change. Avoid
            # redundant edits triggering metrics reassembly in an active scene.
            return
        def change():
            p = self.seq()
            p.GetRelationship(NS + "robot").SetTargets([Sdf.Path(robot_path)])
            author(p, "tcpPosition", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*tcp.position))
            q = tcp.orientation
            author(p, "tcpOrientation", Sdf.ValueTypeNames.Quatd, Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        self.edit(change)

    def add(self, pose, source=None):
        self.seq()
        path = self.unique(self.sequence + "/Waypoints/Waypoint")
        wp = self.read(source) if source else Waypoint(path, pose)
        def create():
            p = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            author(p, "kind", Sdf.ValueTypeNames.Token, "waypoint")
            author(p, "version", Sdf.ValueTypeNames.Int, 1)
            author(p, "id", Sdf.ValueTypeNames.String, str(uuid.uuid4()))
            set_world_pose(p, pose)
            self._settings(p, wp)
            self.seq().GetRelationship(NS + "waypoints").SetTargets(
                [Sdf.Path(s) for s in self.paths()] + [Sdf.Path(path)])
        self.edit(create)
        return path

    def _settings(self, prim, wp):
        author(prim, "enabled", Sdf.ValueTypeNames.Bool, wp.enabled)
        author(prim, "gripper", Sdf.ValueTypeNames.Token, wp.gripper)
        for name in ("dwell", "speed", "position_tolerance", "orientation_tolerance", "timeout"):
            author(prim, name, Sdf.ValueTypeNames.Double, getattr(wp, name))

    def read(self, path):
        prim = self.member(path)
        return Waypoint(path, world_pose(prim), **{
            key: attr(prim, key, default) for key, default in {
                "enabled": True, "gripper": "keep", "dwell": .5, "speed": .3,
                "position_tolerance": .005, "orientation_tolerance": np.radians(3),
                "timeout": 30.,
            }.items()})

    def member(self, path):
        if not path:
            raise ValueError("Select a waypoint first.")
        prim = self.stage.GetPrimAtPath(path)
        if (path not in self.paths() or not tagged(prim, "waypoint") or not prim.IsActive()
                or not prim.IsA(UsdGeom.Xform)):
            raise ValueError(f"{path} is not a tagged member of the selected sequence.")
        return prim

    def update(self, wp):
        self.member(wp.path)
        def change():
            p = self.stage.GetPrimAtPath(wp.path)
            set_world_pose(p, wp.pose)
            self._settings(p, wp)
            result = self.read(wp.path)
            distance, angle = result.pose.error(wp.pose)
            if distance > 1e-8 or angle > 1e-6 or any(
                    getattr(result, key) != getattr(wp, key) for key in
                    ("enabled", "gripper", "dwell", "speed", "position_tolerance", "orientation_tolerance", "timeout")):
                raise ValueError("A stronger USD layer overrides this waypoint; remove its conflicting opinions first.")
        self.edit(change)

    def reorder(self, path, direction):
        paths = self.paths()
        i = paths.index(path)
        j = max(0, min(len(paths)-1, i+direction))
        paths[i], paths[j] = paths[j], paths[i]
        self.edit(lambda: self.seq().GetRelationship(NS + "waypoints").SetTargets(paths))

    def remove(self, path):
        if path not in self.paths():
            raise ValueError("Select a member of this sequence first.")
        remaining = [s for s in self.paths() if s != path]
        def change():
            self.seq().GetRelationship(NS + "waypoints").SetTargets(remaining)
            # Deactivate works even when opinions originate in a weaker layer.
            prim = self.stage.GetPrimAtPath(path)
            if tagged(prim, "waypoint"):
                prim.SetActive(False)
        self.edit(change)

    def snapshot(self, selected=None):
        paths = [selected] if selected else self.paths()
        if len(paths) != len(set(paths)):
            raise ValueError("Sequence contains duplicate waypoint references.")
        result = tuple(self.read(p) for p in paths if attr(self.member(p), "enabled", True))
        if not result:
            raise ValueError("The sequence has no enabled waypoints.")
        return result

    def save(self, filename):
        path = Path(filename).expanduser().resolve()
        if path.suffix not in (".usd", ".usda") or not path.parent.is_dir():
            raise ValueError("Choose an existing directory and a .usd or .usda filename.")
        layer = self._layer()
        previous_target = self.stage.GetEditTarget()
        was_editing_layer = previous_target.GetLayer() == layer
        # Prevent accidentally replacing the source scene or any unrelated layer.
        if path.exists() and str(path) != layer.realPath:
            raise ValueError("File already exists. Load it first or choose a new filename.")
        if path.exists():
            if not layer.Save():
                raise RuntimeError("Could not save waypoint layer.")
            return
        layers = list(self.stage.GetRootLayer().subLayerPaths)
        root_dir = Path(self.stage.GetRootLayer().realPath).parent
        import os
        reference = os.path.relpath(path, root_dir) if self.stage.GetRootLayer().realPath else str(path)
        index = next((i for i, ref in enumerate(layers) if ref == layer.identifier or
                      (not layer.anonymous and self.stage.GetRootLayer().realPath and
                       (root_dir / ref).resolve() == Path(layer.realPath))), None)
        if index is None:
            raise ValueError("Waypoint layer is no longer a direct scene sublayer.")
        if not layer.Export(str(path)):
            raise RuntimeError("Could not export waypoint layer.")
        layers[index] = reference
        self.stage.GetRootLayer().subLayerPaths = layers
        self.layer = Sdf.Layer.FindOrOpen(str(path))
        self.stage.SetEditTarget(self.layer if was_editing_layer else previous_target)

    def load(self, filename):
        if self.layer and self.layer.dirty:
            raise ValueError("Save the current authoring layer before loading another.")
        path = str(Path(filename).expanduser().resolve())
        layer = Sdf.Layer.FindOrOpen(path)
        if not layer or layer.customLayerData.get("frankaPathAuthoring") != 1:
            raise ValueError("This is not a Franka Waypoint Editor layer.")
        self.validate_conventions(layer)
        if self.layer and self.layer != layer:
            raise ValueError("Open another scene to switch authoring layers.")
        if layer not in self.stage.GetLayerStack():
            root = self.stage.GetPrimAtPath(DEFAULT_ROOT)
            if root and not tagged(root, "collection"):
                raise ValueError(f"{DEFAULT_ROOT} is occupied by unrelated scene content.")
            self.stage.GetRootLayer().subLayerPaths.insert(0, path)
        self.layer = layer
        choices = self.sequences()
        self.sequence = choices[0] if choices else ""


def find_robots(stage):
    """Discover Panda link/joint sets; never identify a robot by its path alone."""
    roots = []
    for p in stage.Traverse():
        if p.GetName() != "panda_hand":
            continue
        parent = p.GetParent()
        while parent and not parent.IsPseudoRoot():
            children = {q.GetName(): q for q in Usd.PrimRange(parent)}
            if ("panda_link0" in children and all(f"panda_joint{i}" in children for i in range(1, 8))
                    and all(n in children for n in ("panda_leftfinger", "panda_rightfinger"))
                    and any(q.HasAPI(UsdPhysics.ArticulationRootAPI) for q in children.values())):
                roots.append(str(parent.GetPath()))
                break
            parent = parent.GetParent()
    return sorted(set(roots))
