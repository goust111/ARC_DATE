# Variant of 7ed72f31
# Dimension: D8 — 多标记组的处理顺序 (multi_step)
# Change: 各标记组独立处理，无特定顺序要求，结果直接叠加... → 按标记类型分阶段处理

from __future__ import annotations

"""
Design analysis for the D8 variant
==================================

source_task:
  id: d59b0160
  solver_status: unavailable in this workspace
  note: only the evaluation json and a short textual summary are present, so the
    multi-step baseline is inferred from the variant description rather than by
    reading a reference solver.

a_concretization_and_trace:
  original_behavior:
    - Every marker group reads from the same original input grid.
    - A horizontal red line mirrors payload vertically.
    - A vertical red line mirrors payload horizontally.
    - A red point mirrors an adjacent component by point reflection.
    - The three operations are effectively parallel overlays.
  variant_behavior:
    - Stage 1: process all horizontal-axis groups on the current grid.
    - Stage 2: process all vertical-axis groups on the result of stage 1.
    - Stage 3: process all point-reflection groups on the result of stage 2.
    - Within one stage, all groups read from the same stage-input snapshot; they
      do not chain into one another until the next stage.
  concrete_chain_used_by_generator:
    - One shared-color horizontal group creates new pixels below its red row.
    - Those new pixels lie inside the row-range of a vertical-axis group with the
      same payload color, so stage 2 reflects both the original vertical payload
      and the stage-1 additions.
    - The stage-2 right-side component becomes larger than it was in the raw
      input and sits diagonally next to an isolated red point, so stage 3 point-
      reflects the enlarged component rather than the small seed alone.
  small_example_trace:
    - Input: row group at upper-left, col group in the middle, isolated red point
      near the future right-side reflection, all with the same payload color.
    - Original solver: row copy appears; column group ignores it because it still
      reads the raw input; point marker reflects only its tiny original neighbor.
    - New solver: row copy appears first; column group sees it and reflects extra
      pixels; point marker then sees the enlarged right-side component and copies
      more pixels than the original task would.

b_complexity_calibration:
  - The generator keeps exactly one explicit cascade chain so the staged rule is
    visible, but allows random bg/color/translation/flip so samples are not
    trivial clones.
  - Only one shared color is reused across the three chained groups. This is the
    minimum adaptation needed for D8 to be observable; the original "distinct
    color per group" habit would suppress all cross-stage interactions.
  - The chained sprites are asymmetric, so every reflection is visually legible.

c_implicit_assumption_audit_on_main:
  - Original code scanned the immutable input grid `g` in every loop. That is the
    exact old-property dependency that must change for D8.
  - Group detection still uses the original input. This assumes markers do not
    move and newly painted pixels are not needed to discover new markers. The
    generator guarantees that.
  - Row eligibility still depends on the marker component's column span; column
    eligibility still depends on its row span; point eligibility still depends on
    local adjacency / local bbox. The generator places the chain so stage-1
    outputs land inside the later eligible spans on purpose.
  - Point markers are applied in the final stage using one shared stage snapshot.
    This avoids ambiguous within-stage ordering between multiple point markers.

d_placement_legality:
  - The generator uses a fixed local interaction macro whose sequential output is
    known to stay inside a 9x13 bounding box even after all three stages.
  - That macro is only translated and flipped horizontally/vertically, never
    rotated 90 degrees, so "horizontal before vertical before point" remains the
    intended semantic order.
  - The macro contains no input overlaps, and its future reflected pixels also
    stay inside bounds. A self-check rejects any accidental degeneration.

e_visual_observability:
  - The final output contains pixels that cannot be explained by one-step local
    mirroring alone: some come from a vertical reflection of a horizontal copy,
    and some come from a point reflection of that enlarged vertical result.
  - The generator rejects samples where the sequential result equals the old
    parallel baseline or where the column/point cascade contributions vanish.

f_visual_cleanliness:
  - Only one chain occupies one compact region; there is no dense pile-up of many
    unrelated groups.
  - The red row, red column, and isolated red point remain clearly distinguishable
    from one another in the input.

g_color_inference:
  - No extra color remapping is introduced.
  - The only color adaptation is controlled color sharing across the three linked
    groups; that sharing is itself a clue that the same colored sprite is being
    reused across stages.

post_change_generator_guarantees:
  - At least one horizontal group, one vertical group, and one isolated point are
    present in every sample.
  - The column stage has at least one output pixel that would not appear if it
    ran on the raw input.
  - The point stage has at least one output pixel that would not appear if it ran
    before the column stage.
"""

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask

MARK = 2
PALETTE = tuple(range(10))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(grid != int(bg), connectivity=4)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(grid[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        reds = [(r, c) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "reds": reds,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
                "payload": {int(v) for v in vals if int(v) != int(MARK)},
            }
        )
    return comps


def _extract_groups(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    row_groups: list[dict] = []
    col_groups: list[dict] = []
    point_groups: list[dict] = []
    isolated_markers: list[dict] = []

    for comp in _component_records(grid, bg):
        vals = comp["vals"]
        reds = comp["reds"]
        if len(comp["coords"]) == 1 and int(vals[0]) == int(MARK):
            isolated_markers.append(comp)
            continue
        if int(MARK) not in set(map(int, vals)) or not reds or not comp["payload"]:
            continue

        row_set = {int(r) for r, _ in reds}
        col_set = {int(c) for _, c in reds}
        if len(reds) == 1:
            point_groups.append(comp)
        elif len(row_set) == 1:
            row_groups.append(comp)
        elif len(col_set) == 1:
            col_groups.append(comp)

    def _key(rec: dict) -> tuple[int, int]:
        r0, c0, _r1, _c1 = rec["bbox"]
        return int(r0), int(c0)

    row_groups.sort(key=_key)
    col_groups.sort(key=_key)
    point_groups.sort(key=_key)
    isolated_markers.sort(key=_key)
    return row_groups, col_groups, point_groups, isolated_markers


def _apply_row_stage(base: np.ndarray, groups: Iterable[dict], bg: int) -> np.ndarray:
    h, w = base.shape
    out = base.copy()
    writes: dict[tuple[int, int], int] = {}
    for rec in groups:
        _r0, c0, _r1, c1 = map(int, rec["bbox"])
        axis_r = int(rec["reds"][0][0])
        payload = {int(v) for v in rec["payload"]}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(base[r, c])
                if v == int(bg) or v == int(MARK) or v not in payload or not (int(c0) <= int(c) <= int(c1)):
                    continue
                rr, cc = _refl_row(int(axis_r), int(r), int(c))
                if _in_bounds(base.shape, rr, cc):
                    writes[(int(rr), int(cc))] = int(v)
    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _apply_col_stage(base: np.ndarray, groups: Iterable[dict], bg: int) -> np.ndarray:
    h, w = base.shape
    out = base.copy()
    writes: dict[tuple[int, int], int] = {}
    for rec in groups:
        r0, _c0, r1, _c1 = map(int, rec["bbox"])
        axis_c = int(rec["reds"][0][1])
        payload = {int(v) for v in rec["payload"]}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(base[r, c])
                if v == int(bg) or v == int(MARK) or v not in payload or not (int(r0) <= int(r) <= int(r1)):
                    continue
                rr, cc = _refl_col(int(axis_c), int(r), int(c))
                if _in_bounds(base.shape, rr, cc):
                    writes[(int(rr), int(cc))] = int(v)
    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _apply_point_stage(base: np.ndarray, mixed_groups: Iterable[dict], isolated_markers: Iterable[dict], bg: int) -> np.ndarray:
    h, w = base.shape
    out = base.copy()
    writes: dict[tuple[int, int], int] = {}

    for rec in mixed_groups:
        reds = rec["reds"]
        if len(reds) != 1:
            continue
        pr, pc = map(int, reds[0])
        r0, c0, r1, c1 = map(int, rec["bbox"])
        rr0 = max(0, int(r0 - 1))
        rr1 = min(int(h - 1), int(r1 + 1))
        cc0 = max(0, int(c0 - 1))
        cc1 = min(int(w - 1), int(c1 + 1))
        payload = {int(v) for v in rec["payload"]}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(base[r, c])
                if v == int(bg) or v == int(MARK) or v not in payload:
                    continue
                if not (int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)):
                    continue
                rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                if _in_bounds(base.shape, rr, cc):
                    writes[(int(rr), int(cc))] = int(v)

    labels_by_color: dict[int, np.ndarray] = {}
    for rec in isolated_markers:
        pr, pc = map(int, rec["coords"][0])
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _in_bounds(base.shape, rr, cc):
                    continue
                v = int(base[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        for v in neigh_colors:
            if int(v) not in labels_by_color:
                labels_by_color[int(v)], _ = label_mask(base == int(v), connectivity=8)
            labv = labels_by_color[int(v)]
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if _in_bounds(base.shape, rr, cc):
                        writes[(int(rr), int(cc))] = int(v)

    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _trace_execution(grid: np.ndarray) -> dict[str, np.ndarray | list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    row_groups, col_groups, point_groups, isolated_markers = _extract_groups(g, bg)
    after_row = _apply_row_stage(g, row_groups, bg)
    after_col = _apply_col_stage(after_row, col_groups, bg)
    after_point = _apply_point_stage(after_col, point_groups, isolated_markers, bg)
    return {
        "input": g,
        "after_row": after_row,
        "after_col": after_col,
        "after_point": after_point,
        "row_groups": row_groups,
        "col_groups": col_groups,
        "point_groups": point_groups,
        "isolated_markers": isolated_markers,
    }


def _parallel_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    row_groups, col_groups, point_groups, isolated_markers = _extract_groups(g, bg)
    out = g.copy()
    row_part = _apply_row_stage(g, row_groups, bg)
    col_part = _apply_col_stage(g, col_groups, bg)
    point_part = _apply_point_stage(g, point_groups, isolated_markers, bg)
    out[row_part != g] = row_part[row_part != g]
    out[col_part != g] = col_part[col_part != g]
    out[point_part != g] = point_part[point_part != g]
    return out


def _delta_coords(before: np.ndarray, after: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(np.array(before, dtype=int) != np.array(after, dtype=int))}


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    trace = _trace_execution(grid)
    g = np.array(trace["input"], dtype=int)
    after_row = np.array(trace["after_row"], dtype=int)
    after_col = np.array(trace["after_col"], dtype=int)
    after_point = np.array(trace["after_point"], dtype=int)
    bg = int(get_bg(g))

    row_groups = trace["row_groups"]
    col_groups = trace["col_groups"]
    point_groups = trace["point_groups"]
    isolated_markers = trace["isolated_markers"]

    col_on_raw = _apply_col_stage(g, col_groups, bg)
    point_after_row = _apply_point_stage(after_row, point_groups, isolated_markers, bg)
    parallel = _parallel_baseline(g)

    return {
        "n_row": int(len(row_groups)),
        "n_col": int(len(col_groups)),
        "n_point_mixed": int(len(point_groups)),
        "n_point_isolated": int(len(isolated_markers)),
        "row_added": int(len(_delta_coords(g, after_row))),
        "col_added": int(len(_delta_coords(after_row, after_col))),
        "point_added": int(len(_delta_coords(after_col, after_point))),
        "col_cascade_pixels": int(len(_delta_coords(after_row, after_col) - _delta_coords(g, col_on_raw))),
        "point_cascade_pixels": int(len(_delta_coords(after_col, after_point) - _delta_coords(after_row, point_after_row))),
        "seq_parallel_delta": int(np.count_nonzero(after_point != parallel)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant of 7ed72f31.

    Marker groups are still recognized exactly as in the original task, but they
    are executed in three stages:
    1. all horizontal-axis reflections
    2. all vertical-axis reflections
    3. all point reflections

    Each stage reads from the previous stage's result, so later marker types can
    act on pixels created earlier.
    """
    return np.array(_trace_execution(np.array(grid, dtype=int))["after_point"], dtype=int)


_CHAIN_CANVAS_SHAPE = (9, 13)
_CHAIN_TEMPLATE = {
    "row_red": [(2, 0), (2, 1), (2, 2), (2, 3)],
    "row_payload": [(1, 0), (0, 1), (1, 1), (0, 2)],
    "col_red": [(3, 6), (4, 6), (5, 6), (6, 6), (7, 6)],
    "col_payload": [(4, 5), (5, 4), (5, 5), (6, 5)],
    "point_red": [(6, 10)],
    "point_seed": [(5, 11)],
}


def _flip_coords(coords: Iterable[tuple[int, int]], *, height: int, width: int, flip_ud: bool, flip_lr: bool) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(height - 1 - r) if flip_ud else int(r)
        cc = int(width - 1 - c) if flip_lr else int(c)
        out.append((int(rr), int(cc)))
    return out


def _stamp(grid: np.ndarray, coords: Iterable[tuple[int, int]], color: int, r0: int, c0: int) -> None:
    for r, c in coords:
        grid[int(r0 + r), int(c0 + c)] = int(color)


def generate_input() -> np.ndarray:
    """
    Generate a legal D8 sample with a guaranteed visible row->col->point cascade.

    The construction is input-first with solver-backed rejection:
    we place a compact interaction macro, then reject any sample whose measured
    behavior collapses back to the original parallel task.
    """
    rng = random.Random()
    chain_h, chain_w = _CHAIN_CANVAS_SHAPE

    for _attempt in range(200):
        h = rng.randint(18, 24)
        w = rng.randint(18, 24)
        bg = int(rng.choice([c for c in PALETTE if c != MARK]))
        chain_color = int(rng.choice([c for c in PALETTE if c not in (bg, MARK)]))
        grid = np.full((int(h), int(w)), int(bg), dtype=int)

        flip_ud = bool(rng.randint(0, 1))
        flip_lr = bool(rng.randint(0, 1))
        r0 = rng.randint(0, int(h - chain_h))
        c0 = rng.randint(0, int(w - chain_w))

        row_red = _flip_coords(_CHAIN_TEMPLATE["row_red"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        row_payload = _flip_coords(_CHAIN_TEMPLATE["row_payload"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        col_red = _flip_coords(_CHAIN_TEMPLATE["col_red"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        col_payload = _flip_coords(_CHAIN_TEMPLATE["col_payload"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        point_red = _flip_coords(_CHAIN_TEMPLATE["point_red"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        point_seed = _flip_coords(_CHAIN_TEMPLATE["point_seed"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)

        _stamp(grid, row_payload, chain_color, r0, c0)
        _stamp(grid, row_red, MARK, r0, c0)
        _stamp(grid, col_payload, chain_color, r0, c0)
        _stamp(grid, col_red, MARK, r0, c0)
        _stamp(grid, point_red, MARK, r0, c0)
        _stamp(grid, point_seed, chain_color, r0, c0)

        out = main(grid)
        metrics = variant_metrics(grid)

        if (
            out.shape == grid.shape
            and metrics["n_row"] >= 1
            and metrics["n_col"] >= 1
            and metrics["n_point_isolated"] >= 1
            and metrics["col_cascade_pixels"] >= 1
            and metrics["point_cascade_pixels"] >= 1
            and metrics["seq_parallel_delta"] >= 4
        ):
            return grid

    raise RuntimeError("failed to generate a non-degenerate D8 sample")
