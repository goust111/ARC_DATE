# Combo variant of 7ed72f31 (index 12)
# combo_spec: D4/2+D8/3
# Name: L形标记分阶段处理

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   - The original task treated the three marker types independently:
       row line -> reflect across that row,
       column line -> reflect across that column,
       single point -> point reflection.
     Every group read from the same immutable input grid, so the order did not
     matter.
   - This combo keeps the same visual language but adds a fourth marker shape:
       an L-shaped red marker. Its corner is the anchor, and the payload reflects
       across the diagonal that bisects the L interior:
       right+down or left+up -> "\" diagonal,
       right+up or left+down -> "/" diagonal.
   - The key fusion is staged execution:
       1. all horizontal markers,
       2. then all vertical markers,
       3. then all L/diagonal markers,
       4. then all point markers.
     Later stages read the previous stage result, not the raw input.
   - Concrete local trace used by the generator:
       * A horizontal marker copies a few same-color pixels downward.
       * A vertical marker sees those new pixels in its row band and mirrors them
         to the right.
       * An L marker sees that enlarged right-side cluster inside its padded bbox
         and reflects part of it across a diagonal.
       * A single red point finally reflects the now-larger 8-connected color
         component, so the point output is bigger than it would be on the raw input.
   - Difference from the old logic:
       the old solver would ignore the L marker entirely and would apply the point
       reflection before the diagonal-enlarged component existed.

b) Complexity calibration
   - Only four discrete marker classes are used, matching the fusion spec exactly.
   - Every generated sample contains one row marker, one column marker, one
     diagonal L marker, and one point marker, so the solver must infer both
     geometry classification and the fixed stage order.
   - The generator reuses exactly one payload color across the whole chain. This
     is deliberate: without color sharing, the staged interaction would collapse.
     The freedom stays limited to bg color, chain color, translation, and flips.

c) Audit of old implicit assumptions
   - Original assumption: any non-collinear red set is ambiguous and should be
     ignored. This breaks once L markers become meaningful.
   - Original assumption: each reflection scans only the raw input grid `g`.
     This breaks once later markers must see earlier results.
   - Original assumption: groups can be processed in any order. This breaks
     because the combo spec imposes row -> col -> diag -> point.
   - Original bbox heuristics are preserved:
       row markers use the component column span,
       col markers use the component row span,
       diag/point-in-mixed markers use a 1-cell padded bbox.
     The generator guarantees spacing so these coarse filters stay unambiguous.

d) Placement legality
   - The generator uses one fixed local macro whose full staged output fits inside
     an 8x13 canvas. The macro is only translated and flipped, never stretched.
   - Input cells do not overlap, and the final staged output is solver-checked to
     remain inside bounds.
   - The isolated point marker has no 4-neighbor non-bg pixels in the input.
   - The L marker is always a true one-corner union of one horizontal arm and one
     vertical arm; no plus/T/box red shapes are generated.

e) Visual observability
   - The input already shows all four red geometries in one compact region.
   - The output contains pixels that cannot be explained by a single independent
     pass: some right-side pixels depend on the row stage, some diagonal pixels
     depend on the column stage, and some point-reflected pixels depend on the
     diagonal stage.
   - Because the chain uses one shared color, a human can visually follow how the
     same sprite grows from stage to stage.

f) Visual cleanliness
   - Only one compact interaction macro is generated. This avoids clutter from
     multiple overlapping chains.
   - Padded empty space around the placed macro keeps the four markers visually
     separable even though they intentionally interact through later reflections.

g) Color inference
   - Colors are not remapped. The non-red chain color is preserved through every
     reflection. The task is purely about geometry class + stage order.

New solver assumptions introduced by this combo
-----------------------------------------------
1. Marker discovery is frozen from the original input; newly painted pixels never
   create new red markers. `generate_input()` guarantees that.
2. Every generated L marker has a unique valid corner, contiguous one-sided arms,
   and payload cells strictly off the diagonal so the mirror adds visible pixels.
3. The point-stage cascade relies on an isolated red point staying adjacent to one
   same-color 8-connected component whose size can increase after earlier stages.
   The generator guarantees that chain.
4. Coarse row/col/bbox eligibility is intentionally exploited by the macro; the
   generator rejects any sample where those interactions fail to produce staged
   deltas.
"""

MARK = 2
PALETTE = tuple(range(10))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - int(r)), int(2 * pc - int(c))


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _refl_diag(kind: str, corner: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, corner)
    dr = int(r) - int(pr)
    dc = int(c) - int(pc)
    if str(kind) == "diag_main":
        return int(pr + dc), int(pc + dr)
    if str(kind) == "diag_anti":
        return int(pr - dc), int(pc - dr)
    raise ValueError(f"unknown diagonal kind: {kind}")


def _classify_marker(reds: list[tuple[int, int]]) -> tuple[str, object] | None:
    reds = [(int(r), int(c)) for r, c in reds]
    if not reds:
        return None

    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}
    if len(reds) == 1:
        return ("point", (int(reds[0][0]), int(reds[0][1])))
    if len(rs) == 1:
        return ("row", int(next(iter(rs))))
    if len(cs) == 1:
        return ("col", int(next(iter(cs))))

    red_set = set(reds)
    for pr, pc in reds:
        row_cols = sorted(int(c) for r, c in reds if int(r) == int(pr))
        col_rows = sorted(int(r) for r, c in reds if int(c) == int(pc))
        if len(row_cols) < 2 or len(col_rows) < 2:
            continue
        if any((int(r), int(c)) not in red_set or (int(r) != int(pr) and int(c) != int(pc)) for r, c in reds):
            continue

        h_signs = {int(np.sign(int(c) - int(pc))) for c in row_cols if int(c) != int(pc)}
        v_signs = {int(np.sign(int(r) - int(pr))) for r in col_rows if int(r) != int(pr)}
        if len(h_signs) != 1 or len(v_signs) != 1:
            continue

        h_dir = int(next(iter(h_signs)))
        v_dir = int(next(iter(v_signs)))
        if h_dir == 0 or v_dir == 0:
            continue

        row_extent = max(abs(int(c) - int(pc)) for c in row_cols)
        col_extent = max(abs(int(r) - int(pr)) for r in col_rows)
        expect_row = {int(pc + h_dir * k) for k in range(0, int(row_extent) + 1)}
        expect_col = {int(pr + v_dir * k) for k in range(0, int(col_extent) + 1)}
        if set(row_cols) != expect_row or set(col_rows) != expect_col:
            continue
        if len(red_set) != len(expect_row) + len(expect_col) - 1:
            continue

        diag_kind = "diag_main" if int(h_dir) == int(v_dir) else "diag_anti"
        return ("diag", (int(pr), int(pc), str(diag_kind)))
    return None


def _component_records(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab, n = label_mask(grid != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(grid[r, c]) for r, c in coords]
        rs = [int(r) for r, _c in coords]
        cs = [int(c) for _r, c in coords]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
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


def _extract_groups(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    row_groups: list[dict[str, object]] = []
    col_groups: list[dict[str, object]] = []
    diag_groups: list[dict[str, object]] = []
    point_groups: list[dict[str, object]] = []
    isolated_markers: list[dict[str, object]] = []

    for comp in _component_records(grid, int(bg)):
        vals = comp["vals"]
        reds = comp["reds"]
        if len(comp["coords"]) == 1 and int(vals[0]) == int(MARK):
            isolated_markers.append(comp)
            continue
        if int(MARK) not in set(map(int, vals)) or not reds or not comp["payload"]:
            continue
        axis = _classify_marker(list(reds))
        if axis is None:
            continue
        rec = dict(comp)
        rec["axis"] = axis
        if axis[0] == "row":
            row_groups.append(rec)
        elif axis[0] == "col":
            col_groups.append(rec)
        elif axis[0] == "diag":
            diag_groups.append(rec)
        elif axis[0] == "point":
            point_groups.append(rec)

    def _key(rec: dict[str, object]) -> tuple[int, int]:
        r0, c0, _r1, _c1 = map(int, rec["bbox"])
        return int(r0), int(c0)

    row_groups.sort(key=_key)
    col_groups.sort(key=_key)
    diag_groups.sort(key=_key)
    point_groups.sort(key=_key)
    isolated_markers.sort(key=_key)
    return row_groups, col_groups, diag_groups, point_groups, isolated_markers


def _apply_row_stage(base: np.ndarray, groups: Iterable[dict[str, object]], bg: int) -> np.ndarray:
    h, w = base.shape
    out = np.array(base, dtype=int, copy=True)
    writes: dict[tuple[int, int], int] = {}
    for rec in groups:
        _r0, c0, _r1, c1 = map(int, rec["bbox"])
        axis_r = int(rec["axis"][1])
        payload = {int(v) for v in rec["payload"]}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(base[r, c])
                if v == int(bg) or v == int(MARK) or v not in payload or not (int(c0) <= int(c) <= int(c1)):
                    continue
                rr, cc = _refl_row(int(axis_r), int(r), int(c))
                if _in_bounds(base.shape, int(rr), int(cc)):
                    writes[(int(rr), int(cc))] = int(v)
    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _apply_col_stage(base: np.ndarray, groups: Iterable[dict[str, object]], bg: int) -> np.ndarray:
    h, w = base.shape
    out = np.array(base, dtype=int, copy=True)
    writes: dict[tuple[int, int], int] = {}
    for rec in groups:
        r0, _c0, r1, _c1 = map(int, rec["bbox"])
        axis_c = int(rec["axis"][1])
        payload = {int(v) for v in rec["payload"]}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(base[r, c])
                if v == int(bg) or v == int(MARK) or v not in payload or not (int(r0) <= int(r) <= int(r1)):
                    continue
                rr, cc = _refl_col(int(axis_c), int(r), int(c))
                if _in_bounds(base.shape, int(rr), int(cc)):
                    writes[(int(rr), int(cc))] = int(v)
    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _apply_diag_stage(base: np.ndarray, groups: Iterable[dict[str, object]], bg: int) -> np.ndarray:
    h, w = base.shape
    out = np.array(base, dtype=int, copy=True)
    writes: dict[tuple[int, int], int] = {}
    for rec in groups:
        r0, c0, r1, c1 = map(int, rec["bbox"])
        rr0 = max(0, int(r0 - 1))
        rr1 = min(int(h - 1), int(r1 + 1))
        cc0 = max(0, int(c0 - 1))
        cc1 = min(int(w - 1), int(c1 + 1))
        payload = {int(v) for v in rec["payload"]}
        pr, pc, diag_kind = rec["axis"][1]
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(base[r, c])
                if v == int(bg) or v == int(MARK) or v not in payload:
                    continue
                if not (int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)):
                    continue
                rr, cc = _refl_diag(str(diag_kind), (int(pr), int(pc)), int(r), int(c))
                if _in_bounds(base.shape, int(rr), int(cc)):
                    writes[(int(rr), int(cc))] = int(v)
    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _apply_point_stage(
    base: np.ndarray,
    mixed_groups: Iterable[dict[str, object]],
    isolated_markers: Iterable[dict[str, object]],
    bg: int,
) -> np.ndarray:
    h, w = base.shape
    out = np.array(base, dtype=int, copy=True)
    writes: dict[tuple[int, int], int] = {}

    for rec in mixed_groups:
        pr, pc = map(int, rec["axis"][1])
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
                if _in_bounds(base.shape, int(rr), int(cc)):
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
                if not _in_bounds(base.shape, int(rr), int(cc)):
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
                    if _in_bounds(base.shape, int(rr), int(cc)):
                        writes[(int(rr), int(cc))] = int(v)

    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _trace_execution(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    row_groups, col_groups, diag_groups, point_groups, isolated_markers = _extract_groups(g, int(bg))
    after_row = _apply_row_stage(g, row_groups, int(bg))
    after_col = _apply_col_stage(after_row, col_groups, int(bg))
    after_diag = _apply_diag_stage(after_col, diag_groups, int(bg))
    after_point = _apply_point_stage(after_diag, point_groups, isolated_markers, int(bg))
    return {
        "input": g,
        "after_row": after_row,
        "after_col": after_col,
        "after_diag": after_diag,
        "after_point": after_point,
        "row_groups": row_groups,
        "col_groups": col_groups,
        "diag_groups": diag_groups,
        "point_groups": point_groups,
        "isolated_markers": isolated_markers,
    }


def _parallel_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    row_groups, col_groups, diag_groups, point_groups, isolated_markers = _extract_groups(g, int(bg))
    row_part = _apply_row_stage(g, row_groups, int(bg))
    col_part = _apply_col_stage(g, col_groups, int(bg))
    diag_part = _apply_diag_stage(g, diag_groups, int(bg))
    point_part = _apply_point_stage(g, point_groups, isolated_markers, int(bg))
    out = np.array(g, dtype=int, copy=True)
    for arr in (row_part, col_part, diag_part, point_part):
        out[arr != g] = arr[arr != g]
    return out


def _delta_coords(before: np.ndarray, after: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(np.array(before, dtype=int) != np.array(after, dtype=int))}


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    trace = _trace_execution(np.array(grid, dtype=int))
    g = np.array(trace["input"], dtype=int)
    after_row = np.array(trace["after_row"], dtype=int)
    after_col = np.array(trace["after_col"], dtype=int)
    after_diag = np.array(trace["after_diag"], dtype=int)
    after_point = np.array(trace["after_point"], dtype=int)
    bg = int(get_bg(g))

    row_groups = trace["row_groups"]
    col_groups = trace["col_groups"]
    diag_groups = trace["diag_groups"]
    point_groups = trace["point_groups"]
    isolated_markers = trace["isolated_markers"]

    col_on_raw = _apply_col_stage(g, col_groups, int(bg))
    diag_on_after_row = _apply_diag_stage(after_row, diag_groups, int(bg))
    point_on_after_col = _apply_point_stage(after_col, point_groups, isolated_markers, int(bg))
    parallel = _parallel_baseline(g)

    return {
        "n_row": int(len(row_groups)),
        "n_col": int(len(col_groups)),
        "n_diag": int(len(diag_groups)),
        "n_point_mixed": int(len(point_groups)),
        "n_point_isolated": int(len(isolated_markers)),
        "row_added": int(len(_delta_coords(g, after_row))),
        "col_added": int(len(_delta_coords(after_row, after_col))),
        "diag_added": int(len(_delta_coords(after_col, after_diag))),
        "point_added": int(len(_delta_coords(after_diag, after_point))),
        "col_cascade_pixels": int(len(_delta_coords(after_row, after_col) - _delta_coords(g, col_on_raw))),
        "diag_cascade_pixels": int(len(_delta_coords(after_col, after_diag) - _delta_coords(after_row, diag_on_after_row))),
        "point_cascade_pixels": int(len(_delta_coords(after_diag, after_point) - _delta_coords(after_col, point_on_after_col))),
        "seq_parallel_delta": int(np.count_nonzero(after_point != parallel)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo variant of 7ed72f31.

    Marker geometry now has four classes:
    - horizontal red line -> row reflection
    - vertical red line -> column reflection
    - L-shaped red marker -> diagonal reflection
    - single red point -> point reflection

    Execution is staged:
    row -> col -> diag -> point

    Later stages read the previous stage result, so earlier copies can be
    re-reflected by later marker types.
    """
    return np.array(_trace_execution(np.array(grid, dtype=int))["after_point"], dtype=int)


_CHAIN_CANVAS_SHAPE = (8, 13)
_CHAIN_TEMPLATE = {
    "row_red": [(2, 0), (2, 1), (2, 2), (2, 3)],
    "row_payload": [(1, 0), (0, 1), (1, 1), (0, 2)],
    "col_red": [(3, 6), (4, 6), (5, 6), (6, 6), (7, 6)],
    "col_payload": [(4, 5), (5, 4), (5, 5), (6, 5)],
    "diag_red": [(1, 8), (1, 9), (1, 10), (1, 11), (2, 8), (3, 8), (4, 8)],
    "diag_payload": [(2, 10), (2, 11), (3, 11)],
    "point_red": [(4, 10)],
    "point_seed": [(5, 9)],
}


def _flip_coords(
    coords: Iterable[tuple[int, int]],
    *,
    height: int,
    width: int,
    flip_ud: bool,
    flip_lr: bool,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(height - 1 - int(r)) if bool(flip_ud) else int(r)
        cc = int(width - 1 - int(c)) if bool(flip_lr) else int(c)
        out.append((int(rr), int(cc)))
    return out


def _stamp(grid: np.ndarray, coords: Iterable[tuple[int, int]], color: int, r0: int, c0: int) -> None:
    for r, c in coords:
        grid[int(r0 + int(r)), int(c0 + int(c))] = int(color)


def generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Generate one fused sample with a guaranteed visible row->col->diag->point
    cascade. The macro is input-first and solver-backed:
    we stamp a compact chain, run the solver, and reject any placement that
    degenerates to the old parallel behavior.
    """
    rng = random.Random()
    chain_h, chain_w = _CHAIN_CANVAS_SHAPE

    for _attempt in range(300):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        if int(h) < int(chain_h) or int(w) < int(chain_w):
            continue

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
        diag_red = _flip_coords(_CHAIN_TEMPLATE["diag_red"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        diag_payload = _flip_coords(_CHAIN_TEMPLATE["diag_payload"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        point_red = _flip_coords(_CHAIN_TEMPLATE["point_red"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)
        point_seed = _flip_coords(_CHAIN_TEMPLATE["point_seed"], height=chain_h, width=chain_w, flip_ud=flip_ud, flip_lr=flip_lr)

        _stamp(grid, row_payload, chain_color, int(r0), int(c0))
        _stamp(grid, row_red, MARK, int(r0), int(c0))
        _stamp(grid, col_payload, chain_color, int(r0), int(c0))
        _stamp(grid, col_red, MARK, int(r0), int(c0))
        _stamp(grid, diag_payload, chain_color, int(r0), int(c0))
        _stamp(grid, diag_red, MARK, int(r0), int(c0))
        _stamp(grid, point_seed, chain_color, int(r0), int(c0))
        _stamp(grid, point_red, MARK, int(r0), int(c0))

        out = main(grid)
        metrics = variant_metrics(grid)

        if (
            metrics["n_row"] >= 1
            and metrics["n_col"] >= 1
            and metrics["n_diag"] >= 1
            and metrics["n_point_isolated"] >= 1
            and metrics["row_added"] >= 1
            and metrics["col_cascade_pixels"] >= 1
            and metrics["diag_cascade_pixels"] >= 1
            and metrics["point_cascade_pixels"] >= 1
            and metrics["seq_parallel_delta"] >= 4
        ):
            return np.array(grid, dtype=int), np.array(out, dtype=int), metrics

    raise RuntimeError("failed to generate a non-degenerate combo sample")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return np.array(inp, dtype=int)
