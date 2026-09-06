"""Keep native gizmo do/undo/redo in the waypoint layer after deselection.

Kit 107's TransformPrim commands use the current edit target even for undo.
Callbacks scope only explicitly registered waypoint prims; mixed selections
still leave unrelated prim commands in the user's original layer.
"""
import omni.kit.commands as commands
from pxr import UsdGeom
from .stage_store import tagged


class WaypointEditRouter:
    def __init__(self, owner):
        self.owner = owner
        self.pending = []
        self.callbacks = []
        for command in ("TransformPrim", "TransformPrimSRT"):
            for pre, post in ((commands.PRE_DO_CALLBACK, commands.POST_DO_CALLBACK),
                              (commands.PRE_UNDO_CALLBACK, commands.POST_UNDO_CALLBACK)):
                self.callbacks.append(commands.register_callback(command, pre, self.before))
                self.callbacks.append(commands.register_callback(command, post, self.after))

    def before(self, args):
        saved = None
        self.pending.append(saved)
        store = self.owner.store
        path = str(args.get("path", ""))
        if not store or not path or args.get("usd_context_name", ""):
            return
        prim = store.stage.GetPrimAtPath(path)
        if not tagged(prim, "waypoint") or not prim.IsA(UsdGeom.Xform):
            return
        if not store.sequence_for_waypoint(path):
            return
        saved = (store.stage, store.stage.GetEditTarget())
        store.stage.SetEditTarget(store._layer())
        self.pending[-1] = saved

    def after(self, args):
        if self.pending:
            self._restore(self.pending.pop())

    def _restore(self, saved):
        if saved:
            stage, target = saved
            if target.GetLayer() in stage.GetLayerStack():
                stage.SetEditTarget(target)

    def flush(self):
        # Native commands do not guarantee POST callbacks on exceptions.
        # A command stack is synchronous, so no command remains in flight when
        # Kit next delivers an app update or this extension is shut down.
        while self.pending:
            self._restore(self.pending.pop())

    def destroy(self):
        self.flush()
        for callback in self.callbacks:
            commands.unregister_callback(callback)
        self.callbacks.clear()
        self.owner = None
