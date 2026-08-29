"""CPU-only structural contracts for the CES Baseline FSM.

The local Windows interpreter has no torch/Isaac Lab.  These tests therefore
exercise the pure phase/command types and inspect the runtime handlers without
pretending to validate simulation physics.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CES = ROOT / "action_provider" / "ces_grasp"


def load_types_module():
    path = CES / "fsm_types.py"
    spec = importlib.util.spec_from_file_location("ces_fsm_types_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parsed(name: str) -> ast.Module:
    return ast.parse((CES / name).read_text(encoding="utf-8"))


def method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


class BaselinePhaseContractTests(unittest.TestCase):
    def test_nominal_phase_order_has_no_legacy_states(self):
        module = load_types_module()
        self.assertEqual(
            [phase.value for phase in module.BASELINE_PHASE_ORDER],
            [
                "settle",
                "goto_pick",
                "unfold",
                "descend",
                "grasp",
                "lift",
                "return_home",
                "carry",
                "goto_place",
                "place_hold",
                "place_approach",
                "release",
                "retract",
                "done",
            ],
        )
        enum_names = set(module.CesPickPlacePhase.__members__)
        self.assertTrue(
            {"APPROACH", "HOLD", "RAISE_FOR_PLACE", "PLACE_DESCEND"}.isdisjoint(
                enum_names
            )
        )

    def test_facade_dispatches_every_phase_once(self):
        tree = parsed("state_machine.py")
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        assignment = next(
            node
            for node in cls.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_PHASE_HANDLERS" for target in node.targets)
        )
        self.assertIsInstance(assignment.value, ast.Dict)
        self.assertEqual(len(assignment.value.keys), 15)
        handlers = [
            value.value
            for value in assignment.value.values
            if isinstance(value, ast.Constant)
        ]
        self.assertEqual(len(handlers), len(set(handlers)))
        defined = set()
        for file_name in (
            "state_machine.py",
            "fsm_pick.py",
            "fsm_walk.py",
            "fsm_place.py",
        ):
            defined.update(
                node.name
                for node in ast.walk(parsed(file_name))
                if isinstance(node, ast.FunctionDef)
            )
        self.assertTrue(set(handlers).issubset(defined))

    def test_facade_constructor_only_accepts_context_and_speed(self):
        tree = parsed("state_machine.py")
        init = method(tree, "CesPickPlaceStateMachine", "__init__")
        self.assertEqual(
            [argument.arg for argument in init.args.args],
            ["self", "ctx", "speed_scale"],
        )

    def test_cli_keeps_only_baseline_compatibility_values(self):
        source = (ROOT / "sim_main.py").read_text(encoding="utf-8")
        self.assertIn('choices=["walk"]', source)
        self.assertIn('choices=["place"]', source)
        self.assertIn('choices=["ces_pick_smooth_v1"]', source)

    def test_descend_emits_q_ref_without_arm_q(self):
        fn = method(parsed("fsm_pick.py"), "CesPickMixin", "_step_descend")
        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_cmd"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertIn("arm_q_ref", keywords)
        self.assertNotIn("arm_q", keywords)

    def test_carry_primes_reverse_before_walking(self):
        fn = method(parsed("fsm_walk.py"), "CesWalkMixin", "_step_carry")
        source = ast.unparse(fn)
        self.assertIn("-C.WALK_VX", source)
        self.assertIn("prime_walk_filt", source)
        self.assertIn("guide=True", source)
        self.assertLess(source.index("prime_walk_filt"), source.index("GOTO_PLACE"))

    def test_place_releases_at_15_without_cartesian_descent(self):
        tree = parsed("fsm_place.py")
        fn = method(tree, "CesPlaceMixin", "_step_place_approach")
        source = ast.unparse(fn)
        self.assertIn("CesPickPlacePhase.RELEASE", source)
        self.assertNotIn("tcp=", source)
        combined = "\n".join(
            (CES / name).read_text(encoding="utf-8")
            for name in ("state_machine.py", "fsm_pick.py", "fsm_walk.py", "fsm_place.py")
        )
        self.assertNotIn("PLACE_DESCEND", combined)
        self.assertNotIn("vertical_descend", combined)


if __name__ == "__main__":
    unittest.main()
