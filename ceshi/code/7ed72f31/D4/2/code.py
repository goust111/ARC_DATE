# Variant of 7ed72f31
# Dimension: D4 — 标记类型的判定逻辑 (transform)
# Change: 红像素数=1→点反射；同行多个→水平轴反射；同列多个→竖直轴反射... → 标记形状决定变换：L形标记引入对角反射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask

"""
Design analysis for the D4 variant
==================================

a) Concrete behavior and trace
   - Keep the original visual language: a monochrome payload sprite is attached to
     a red marker, and the output adds a reflected copy while preserving the original.
   - Preserve three old marker shapes:
       1. a single red point -> point reflection
       2. a horizontal red line -> reflection across that row
       3. a vertical red line -> reflection across that column
   - Add one new marker shape:
       4. an L made of one horizontal arm and one vertical arm meeting at a corner.
          The corner is the reflection centerline anchor, and the reflection axis is
          the diagonal that bisects the L's interior quadrant.
          * arms right+down or left+up -> "\" diagonal
          * arms right+up or left+down -> "/" diagonal
   - Small trace:
       Original logic sees red cells {(5,5),(5,6),(6,5)}. Because they are neither a
       single point, nor all on one row, nor all on one column, the old solver hits
       `continue` and does nothing.
       Variant logic detects that every red cell lies on row 5 or column 5, that both
       arms are one-sided from corner (5,5), and classifies this as an L with "\" axis.
       A payload pixel at (6,8) has local offset (+1,+3), so diagonal reflection swaps
       the offset to (+3,+1), producing (8,6).
   - Net effect in IO: the new samples contain a fourth kind of red geometry whose
     output copy is diagonal rather than horizontal/vertical/central.

b) Complexity calibration
   - The new branch adds only one discrete marker category. That keeps the hypothesis
     space small enough for ARC while making the difference visible.
   - Each generated grid contains at least one row marker, one column marker, one
     single-point marker, and one L marker. This gives repeated support for the
     shape->transform mapping instead of leaving L as a rare edge case.
   - Payloads stay monochrome and connected. The novelty is in marker geometry, not
     in color mapping or object decomposition.

c) Audit of old implicit assumptions
   - Old classification:
       `len(reds) == 1` -> point, `len(rs) == 1` -> row, `len(cs) == 1` -> col,
       otherwise ambiguous. The variant breaks the last assumption because some
       non-collinear red sets are now meaningful.
   - Old reflection code assumed every valid marker could be represented by one row,
     one column, or one point. The variant adds two diagonal formulas.
   - Old bbox-based eligibility can accidentally pull in detached same-color pixels.
     To avoid changing that machinery more than needed, the generator enforces unique
     payload colors per group and leaves at least one-cell gaps between groups, so the
     solver's coarse eligibility remains harmless.
   - Old isolated-red branch remains valid and is preserved. The generator does not
     rely on it, which keeps the visual rule focused on marker shape inside mixed
     components.

d) Placement legality
   - Generator is output-first: build a fully legal reflected group, then derive the
     input by erasing only the mirrored payload pixels.
   - Each group's full output bbox, padded by one background cell on all sides, must
     fit in the canvas and not overlap previously placed groups.
   - L markers are restricted to true one-corner unions of two one-sided arms; plus
     signs, T shapes, and 2x2 blocks are never generated.
   - Payload source cells are chosen strictly on one side of the corresponding axis so
     source and mirror never overlap.

e) Visual observability
   - L groups use payload cells on one side of the diagonal within the L's interior
     quadrant, so the mirrored copy lands on the opposite diagonal side in the same
     quadrant. This makes the diagonal relation visible without introducing extra
     marker colors or legends.
   - Each sample mixes old and new marker types, so the observer can compare them and
     infer that the red shape, not the payload color, controls the transform.

f) Visual cleanliness
   - No decoy sprites are generated. The original task sometimes tolerated them, but
     they would add clutter without helping the D4 change.
   - Unique payload colors and padded placement prevent local ambiguity between groups.

g) Color inference
   - Colors do not change under the transform. Each group uses one non-red, non-bg
     color. Therefore the human must infer only geometry, not any new color rule.

New solver assumptions introduced by this variant
-------------------------------------------------
1. If a mixed component contains an L marker, all of its red cells lie on the corner's
   row or column, and each arm extends in only one direction from that corner.
   `generate_case()` guarantees exactly that.
2. The bbox-based eligibility for point/L groups assumes no detached same-color pixels
   inside or immediately around the component bbox. The generator guarantees unique
   colors and padded separation.
3. Diagonal reflection assumes source payload cells are on one side of the diagonal,
   not on the diagonal itself, so the mirrored copy is a new visible addition. The
   generator enforces strict inequalities in its local coordinate sampling.
"""


MARK = 2


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _refl_diag(kind: str, corner: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, corner)
    dr = int(r) - pr
    dc = int(c) - pc
    if kind == "diag_main":
        return int(pr + dc), int(pc + dr)
    if kind == "diag_anti":
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

    for pr, pc in reds:
        row_arm = [int(c) for r, c in reds if int(r) == int(pr)]
        col_arm = [int(r) for r, c in reds if int(c) == int(pc)]
        if len(row_arm) < 2 or len(col_arm) < 2:
            continue
        if not all((int(r) == int(pr) or int(c) == int(pc)) for r, c in reds):
            continue
        h_signs = {int(np.sign(int(c) - int(pc))) for c in row_arm if int(c) != int(pc)}
        v_signs = {int(np.sign(int(r) - int(pr))) for r in col_arm if int(r) != int(pr)}
        if len(h_signs) != 1 or len(v_signs) != 1:
            continue
        h_dir = int(next(iter(h_signs)))
        v_dir = int(next(iter(v_signs)))
        if h_dir == 0 or v_dir == 0:
            continue
        diag_kind = "diag_main" if h_dir == v_dir else "diag_anti"
        return ("diag", (int(pr), int(pc), diag_kind))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [int(r) for r, _c in coords]
        cs = [int(c) for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        axis = _classify_marker(reds)
        if axis is None:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(h - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(w - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                elif axis[0] == "point":
                    rr, cc = _refl_point(axis[1], int(r), int(c))
                else:
                    pr, pc, diag_kind = axis[1]
                    rr, cc = _refl_diag(str(diag_kind), (int(pr), int(pc)), int(r), int(c))

                if _inb((h, w), int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    marker_pts: list[tuple[int, int]] = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb((h, w), rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if _inb((h, w), int(rr), int(cc)):
                        out[int(rr), int(cc)] = int(v)

    return out


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _normalize_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(int(r) for r, _c, _v in cells)
    min_c = min(int(c) for _r, c, _v in cells)
    return [(int(r - min_r), int(c - min_c), int(v)) for r, c, v in cells]


def _random_connected_shape(
    candidates: set[tuple[int, int]],
    size_range: tuple[int, int],
    *,
    seed_filter,
    reflect_fn,
) -> set[tuple[int, int]]:
    all_candidates = {tuple(map(int, p)) for p in candidates}
    seeds = [p for p in all_candidates if seed_filter(p)]
    if not seeds:
        raise ValueError("no legal seeds")

    lo, hi = map(int, size_range)
    for _ in range(300):
        size = random.randint(int(lo), int(min(hi, len(all_candidates))))
        shape = {random.choice(seeds)}
        while len(shape) < size:
            frontier = []
            for cell in shape:
                for nxt in _neighbors4(cell):
                    if nxt in all_candidates and nxt not in shape:
                        frontier.append(nxt)
            if not frontier:
                break
            shape.add(random.choice(frontier))
        if len(shape) != size:
            continue
        if any(tuple(map(int, reflect_fn(r, c))) in shape for r, c in shape):
            continue
        return shape
    raise RuntimeError("failed to grow a connected asymmetric shape")


def _rotate_local(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    turns = int(turns) % 4
    out = [(int(r), int(c)) for r, c in coords]
    for _ in range(turns):
        out = [(int(c), int(-r)) for r, c in out]
    return out


def _build_group_row(color: int) -> dict[str, object]:
    line_len = random.randint(3, 6)
    depth = random.randint(3, 5)
    candidates = {(int(r), int(c)) for r in range(1, depth + 1) for c in range(line_len)}
    shape = _random_connected_shape(
        candidates,
        (3, min(8, len(candidates))),
        seed_filter=lambda p: int(p[0]) == 1,
        reflect_fn=lambda r, c: _refl_row(0, int(r), int(c)),
    )
    reds = [(0, c, MARK) for c in range(line_len)]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [(_refl_row(0, int(r), int(c))[0], _refl_row(0, int(r), int(c))[1], int(color)) for r, c in shape]
    if random.choice([False, True]):
        reds = [(-int(r), int(c), int(v)) for r, c, v in reds]
        payload = [(-int(r), int(c), int(v)) for r, c, v in payload]
        mirror = [(-int(r), int(c), int(v)) for r, c, v in mirror]
    in_cells = _normalize_cells(reds + payload)
    out_cells = _normalize_cells(reds + payload + mirror)
    return {"type": "row", "in_cells": in_cells, "out_cells": out_cells}


def _build_group_col(color: int) -> dict[str, object]:
    line_len = random.randint(3, 6)
    width = random.randint(3, 5)
    candidates = {(int(r), int(c)) for r in range(line_len) for c in range(1, width + 1)}
    shape = _random_connected_shape(
        candidates,
        (3, min(8, len(candidates))),
        seed_filter=lambda p: int(p[1]) == 1,
        reflect_fn=lambda r, c: _refl_col(0, int(r), int(c)),
    )
    reds = [(r, 0, MARK) for r in range(line_len)]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [(_refl_col(0, int(r), int(c))[0], _refl_col(0, int(r), int(c))[1], int(color)) for r, c in shape]
    if random.choice([False, True]):
        reds = [(int(r), -int(c), int(v)) for r, c, v in reds]
        payload = [(int(r), -int(c), int(v)) for r, c, v in payload]
        mirror = [(int(r), -int(c), int(v)) for r, c, v in mirror]
    in_cells = _normalize_cells(reds + payload)
    out_cells = _normalize_cells(reds + payload + mirror)
    return {"type": "col", "in_cells": in_cells, "out_cells": out_cells}


def _build_group_point(color: int) -> dict[str, object]:
    candidates = {(int(r), int(c)) for r in range(-2, 3) for c in range(1, 5)}
    shape = _random_connected_shape(
        candidates,
        (3, min(7, len(candidates))),
        seed_filter=lambda p: int(p[1]) == 1,
        reflect_fn=lambda r, c: _refl_point((0, 0), int(r), int(c)),
    )
    turns = random.randint(0, 3)
    shape = set(_rotate_local(list(shape), turns))
    reds = [(0, 0, MARK)]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [(_refl_point((0, 0), int(r), int(c))[0], _refl_point((0, 0), int(r), int(c))[1], int(color)) for r, c in shape]
    in_cells = _normalize_cells(reds + payload)
    out_cells = _normalize_cells(reds + payload + mirror)
    return {"type": "point", "in_cells": in_cells, "out_cells": out_cells}


def _build_group_diag(color: int) -> dict[str, object]:
    arm_len_h = random.randint(3, 5)
    arm_len_v = random.randint(3, 5)
    span = max(int(arm_len_h), int(arm_len_v), 4) + 1
    candidates = {(int(r), int(c)) for r in range(1, span + 1) for c in range(1, span + 1) if int(c) > int(r)}
    shape = _random_connected_shape(
        candidates,
        (4, min(7, len(candidates))),
        seed_filter=lambda p: int(p[0]) == 1,
        reflect_fn=lambda r, c: _refl_diag("diag_main", (0, 0), int(r), int(c)),
    )
    turns = random.randint(0, 3)
    shape = set(_rotate_local(list(shape), turns))
    reds = [(0, c, MARK) for c in range(arm_len_h)] + [(r, 0, MARK) for r in range(1, arm_len_v)]
    reds = _rotate_local([(r, c) for r, c, _v in [(r, c, MARK) for r, c, _v in reds]], turns)
    reds = [(int(r), int(c), MARK) for r, c in reds]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    reds_rc = [(int(r), int(c)) for r, c, _v in reds]
    axis = _classify_marker(reds_rc)
    if axis is None or axis[0] != "diag":
        raise RuntimeError("generated L marker failed classification")
    pr, pc, diag_kind = axis[1]
    mirror = [(_refl_diag(str(diag_kind), (int(pr), int(pc)), int(r), int(c))[0], _refl_diag(str(diag_kind), (int(pr), int(pc)), int(r), int(c))[1], int(color)) for r, c in shape]
    all_coords = {(int(r), int(c)) for r, c, _v in reds + payload}
    if any((int(r), int(c)) in all_coords for r, c, _v in mirror):
        raise RuntimeError("diagonal mirror overlapped source")
    in_cells = _normalize_cells(reds + payload)
    out_cells = _normalize_cells(reds + payload + mirror)
    return {"type": "diag", "in_cells": in_cells, "out_cells": out_cells}


def _place_group(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    group: dict[str, object],
) -> None:
    out_cells = list(group["out_cells"])
    in_cells = list(group["in_cells"])
    gh = 1 + max(int(r) for r, _c, _v in out_cells)
    gw = 1 + max(int(c) for _r, c, _v in out_cells)
    h, w = inp.shape

    for _ in range(500):
        top = random.randint(0, int(h - gh))
        left = random.randint(0, int(w - gw))
        pad_r0 = max(0, int(top - 1))
        pad_c0 = max(0, int(left - 1))
        pad_r1 = min(int(h), int(top + gh + 1))
        pad_c1 = min(int(w), int(left + gw + 1))
        if np.any(occupied[pad_r0:pad_r1, pad_c0:pad_c1]):
            continue
        for r, c, v in out_cells:
            out[int(top + r), int(left + c)] = int(v)
            occupied[int(top + r), int(left + c)] = True
        for r, c, v in in_cells:
            inp[int(top + r), int(left + c)] = int(v)
        return

    raise RuntimeError(f"failed to place {group['type']} group")


def generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    for _attempt in range(600):
        h = random.randint(20, 28)
        w = random.randint(20, 28)
        bg = random.choice([c for c in range(10) if c != MARK])
        palette = [c for c in range(10) if c not in (bg, MARK)]
        random.shuffle(palette)

        group_types = ["row", "col", "point", "diag"]
        extras = random.choices(group_types, k=random.randint(0, 2))
        kinds = group_types + extras
        random.shuffle(kinds)
        if len(kinds) > len(palette):
            continue

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        meta_groups: list[dict[str, object]] = []
        ok = True

        try:
            for idx, kind in enumerate(kinds):
                color = int(palette[idx])
                if kind == "row":
                    group = _build_group_row(color)
                elif kind == "col":
                    group = _build_group_col(color)
                elif kind == "point":
                    group = _build_group_point(color)
                else:
                    group = _build_group_diag(color)
                _place_group(inp, out, occupied, group)
                meta_groups.append({"type": kind, "color": color})
        except RuntimeError:
            ok = False

        if not ok:
            continue
        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        diag_count = sum(1 for g in meta_groups if g["type"] == "diag")
        point_count = sum(1 for g in meta_groups if g["type"] == "point")
        meta = {
            "shape": (h, w),
            "bg": bg,
            "groups": meta_groups,
            "diag_groups": diag_count,
            "point_groups": point_count,
            "total_groups": len(meta_groups),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid variant sample")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
