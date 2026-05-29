# Combo variant of 2ba387bc (index 18)
# combo_spec: D5/4+D9/3
# Name: 交错排列+等数量约束

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Input still contains isolated 4x4 objects on a plain background.
#      Frames have a 1-cell colored border and a bg-colored 2x2 center; solids
#      are fully filled 4x4 monochrome blocks.
#    - Original task:
#        1. detect all 4x4 objects;
#        2. split into frame list and solid list;
#        3. sort each list by (row, col);
#        4. stack frames in the left output column and solids in the right one.
#    - Fused task:
#        1. keep the same detection and per-type sorting;
#        2. enforce equal counts: F1..Fn and S1..Sn;
#        3. compare the first sorted frame with the first sorted solid:
#           whichever is higher in the input starts the single output column;
#        4. alternate types in one 4-wide column:
#           if F1 is higher, output F1,S1,F2,S2,...,Fn,Sn;
#           if S1 is higher, output S1,F1,S2,F2,...,Sn,Fn.
#      Example:
#        frames at rows [5, 17], solids at rows [2, 10]
#        original output: left=[F1,F2], right=[S1,S2]
#        fused output:    [S1,F1,S2,F2] in one 4-wide stack.
#      The crucial difference from the D5-only variant is that the first slot is
#      no longer hard-coded to a frame; sorting geometry decides the start type.
# b) Complexity calibration:
#    - The generator uses 2-4 pairs so alternation is visible, not trivial.
#    - Total colors stay unique across all objects, preserving object identity.
#    - Samples where the global top-to-bottom order already equals the fused
#      interleaving are rejected; otherwise a solver could ignore the per-type
#      grouping and just sort every object globally.
#    - The generator also randomizes whether a frame or a solid starts, so the
#      "sorting decides the first slot" rule is genuinely exercised.
# c) Implicit-assumption audit of the original main():
#    - `find_components(... monochromatic=False)` still assumes every object is a
#      separated 4-connected component. Generator preserves that with padding.
#    - `bbox_h == bbox_w == 4` still holds; the variant does not change shape.
#    - Frame classification via monochrome border + bg center still holds.
#    - Original `outH = 4 * max(len(frames), len(solids))` breaks because the
#      fused output gives every object its own 4-row slot in a single column.
#    - Original `outW = 8` and fixed paste columns 0 / 4 break because there is
#      now only one 4-cell-wide destination column.
#    - D5-only code assumed "start with frame". D9/3 makes equal counts available
#      for a true symmetric alternation, so that hidden assumption must be removed.
# d) Placement legality:
#    - Objects are placed with `random_free_location(... border=1, padding=1,
#      padding_connectivity=4)`, so they neither overlap nor touch.
#    - The canvas is sized large enough for 4-8 objects with separation margins.
#    - After placement, the generator re-extracts objects and validates both
#      counts and the fused output, so only legal instances are emitted.
# e) Visual observability:
#    - Across examples, humans can see that output width collapses from two
#      columns to one, and adjacent 4x4 slots alternate frame/solid.
#    - Because some samples start with a frame and others start with a solid,
#      the first slot cannot be explained by a fixed type rule; it must come from
#      whichever type's first member appears higher in the input.
# f) Visual cleanliness:
#    - The scene remains uncluttered: only isolated 4x4 objects on background,
#      no extra markers, no overlays, no partial occlusion.
# g) Color inferability:
#    - Colors are preserved and never remapped. They only serve to distinguish
#      objects, not to encode hidden ordering signals.
# h) Quantitative anti-degeneracy target:
#    - Post-generation metrics should show:
#        1. `len(frames) == len(solids)` always;
#        2. both start kinds (`frame` and `solid`) appear over a sample batch;
#        3. `global_type_order != fused_type_order` always.
#      These checks verify that equal-count alternation and dynamic start both
#      materially affect the generated tasks.
# i) New solver assumptions and matching generator guarantees:
#    - Solver assumes the start type can be inferred by comparing the earliest
#      frame and earliest solid under the same visible `(row, col)` ordering used
#      elsewhere in the task. Generator guarantees those objects exist and are
#      visually separated.
#    - Solver assumes relevant objects are exactly the detected 4x4 components.
#      Generator never adds distractors or touching artifacts that would violate
#      this extraction model.


Obj = tuple[int, int, np.ndarray]


def _extract_objects(grid: np.ndarray) -> tuple[int, list[Obj], list[Obj]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[Obj] = []
    solids: list[Obj] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = set(map(int, border.flatten()))

        if len(border_colors) == 1 and bool((sub[1:3, 1:3] == bg).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _choose_start_kind(frames: list[Obj], solids: list[Obj]) -> str:
    if frames and solids:
        frame_key = (int(frames[0][0]), int(frames[0][1]))
        solid_key = (int(solids[0][0]), int(solids[0][1]))
        return "frame" if frame_key < solid_key else "solid"
    if frames:
        return "frame"
    if solids:
        return "solid"
    return "frame"


def _interleave_objects(frames: list[Obj], solids: list[Obj], start_kind: str) -> list[Obj]:
    ordered: list[Obj] = []
    n = max(len(frames), len(solids))
    for i in range(n):
        if start_kind == "frame":
            if i < len(frames):
                ordered.append(frames[i])
            if i < len(solids):
                ordered.append(solids[i])
        else:
            if i < len(solids):
                ordered.append(solids[i])
            if i < len(frames):
                ordered.append(frames[i])
    return ordered


def _build_output(bg: int, frames: list[Obj], solids: list[Obj]) -> np.ndarray:
    ordered = _interleave_objects(frames, solids, _choose_start_kind(frames, solids))
    out = np.full((4 * len(ordered), 4), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(ordered):
        paste(out, sub, int(i * 4), 0)
    return out


def _global_type_order(frames: list[Obj], solids: list[Obj]) -> list[str]:
    tagged = [("frame", *item) for item in frames] + [("solid", *item) for item in solids]
    tagged.sort(key=lambda t: (int(t[1]), int(t[2])))
    return [str(t[0]) for t in tagged]


def _fused_type_order(frames: list[Obj], solids: list[Obj]) -> list[str]:
    start_kind = _choose_start_kind(frames, solids)
    order: list[str] = []
    n = max(len(frames), len(solids))
    for i in range(n):
        if start_kind == "frame":
            if i < len(frames):
                order.append("frame")
            if i < len(solids):
                order.append("solid")
        else:
            if i < len(solids):
                order.append("solid")
            if i < len(frames):
                order.append("frame")
    return order


def _fixed_frame_type_order(n_frames: int, n_solids: int) -> list[str]:
    order: list[str] = []
    n = max(int(n_frames), int(n_solids))
    for i in range(n):
        if i < n_frames:
            order.append("frame")
        if i < n_solids:
            order.append("solid")
    return order


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    _bg, frames, solids = _extract_objects(grid)
    start_kind = _choose_start_kind(frames, solids)
    return {
        "n_frames": len(frames),
        "n_solids": len(solids),
        "start_kind": start_kind,
        "first_rows": (
            int(frames[0][0]) if frames else None,
            int(solids[0][0]) if solids else None,
        ),
        "global_matches_fused": _global_type_order(frames, solids) == _fused_type_order(frames, solids),
        "matches_fixed_frame_d5": _fused_type_order(frames, solids)
        == _fixed_frame_type_order(len(frames), len(solids)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects by type, then alternate them in a single column.

    - Frames and solids are each sorted by visible input position (row, col).
    - The type whose first member is higher in the input starts the stack.
    - Output width is 4 and height is 4 times the total object count.
    """
    bg, frames, solids = _extract_objects(np.array(grid, dtype=int))
    return _build_output(bg, frames, solids)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _sample_pair_count() -> int:
    return int(np.random.randint(2, 5))


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_pairs = _sample_pair_count()
    total = int(2 * n_pairs)
    target_start_kind = "frame" if bool(np.random.randint(0, 2) == 0) else "solid"

    H = int(np.random.randint(24, 36))
    W = int(np.random.randint(24, 36))
    g = np.full((H, W), int(bg), dtype=int)

    colors = np.random.permutation(np.arange(1, 10))[:total]
    objects: list[tuple[str, np.ndarray]] = []
    for i in range(n_pairs):
        objects.append(("frame", _make_frame(int(colors[i]), bg)))
    for i in range(n_pairs):
        objects.append(("solid", _make_solid(int(colors[n_pairs + i]))))

    placement_order = np.random.permutation(len(objects))
    try:
        for idx in placement_order:
            _kind, sprite = objects[int(idx)]
            r0, c0 = random_free_location(
                g, sprite, transparent=bg, border=1, padding=1, padding_connectivity=4
            )
            paste(g, sprite, int(r0), int(c0), transparent=bg)
    except Exception:
        return None

    bg2, frames, solids = _extract_objects(g)
    if len(frames) != n_pairs or len(solids) != n_pairs:
        return None
    if int(frames[0][0]) == int(solids[0][0]):
        return None

    start_kind = _choose_start_kind(frames, solids)
    if start_kind != target_start_kind:
        return None

    global_types = _global_type_order(frames, solids)
    fused_types = _fused_type_order(frames, solids)
    if global_types == fused_types:
        return None

    metrics = _fusion_metrics(g)
    if metrics["n_frames"] != metrics["n_solids"]:
        return None
    if bool(metrics["global_matches_fused"]):
        return None
    if str(metrics["start_kind"]) != target_start_kind:
        return None

    expected = _build_output(bg2, frames, solids)
    actual = main(g)
    if actual.shape != (4 * total, 4):
        return None
    if not np.array_equal(actual, expected):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
