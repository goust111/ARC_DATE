# Combo variant of 7ed72f31 (index 22)
# combo_spec: D1/2+D4/2
# Name: 对角反射与L形标记

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


"""
Design analysis for combo D1/2 + D4/2
=====================================

a) Concrete behavior and trace
   - Keep the original visual language: red marks define a reflection rule, colored
     monochrome payload cells are copied to the mirrored side, and the original
     payload stays in place.
   - Preserve three original marker families:
       1. one red point inside a mixed component -> point reflection
       2. one horizontal red segment -> reflection across that row
       3. one vertical red segment -> reflection across that column
   - Add the fused behavior:
       4. a red L marker (one horizontal arm + one vertical arm sharing one
          corner) means diagonal reflection through that corner.
          * right+down or left+up => "\" / y=x style reflection
          * right+up or left+down => "/" / y=-x style reflection
   - Small trace:
       Reds at {(5,5),(5,6),(6,5)} form an L with corner (5,5). The old solver
       rejected it as "ambiguous". The combo solver classifies it as a "\" axis.
       A payload pixel at (6,8) has offset (+1,+3) from the corner; diagonal
       reflection swaps offsets to (+3,+1), so the copy appears at (8,6).
   - Difference from D1-only:
       D1 used red diagonal lines as the visible axis. In this combo the visible
       cue is the L shape, while the transformation is still diagonal.

b) Complexity calibration
   - Only four discrete marker categories exist: point, row, column, L-diagonal.
     This is small enough for ARC but visually richer than the base task.
   - Each generated sample contains at least one row marker, one column marker,
     one point marker, and one L marker. This gives repeated evidence that marker
     geometry, not color, chooses the transform.
   - Generator uses 0-2 extra groups, usually including another L, so diagonal
     behavior is not a rare edge case.

c) Implicit assumptions broken by the original code
   - Original `else: continue` after row/col tests assumed every non-collinear red
     set was invalid. That fails once L shapes become meaningful.
   - Original bbox eligibility scanned the whole grid for same-color pixels near a
     marker. With diagonal reflections that becomes unnecessarily fragile, so the
     combo solver mirrors only payload pixels inside the same mixed component.
   - Original 4-connected component labeling is still valid here because the new
     diagonal axis is encoded by a 4-connected L, not by a diagonal red line.

d) Placement legality
   - Generator is output-first: build a legal full group (marker + payload +
     mirrored payload), then erase only the mirrored copy to form the input.
   - Every group is normalized to a local patch and placed with a 1-cell moat
     around its output footprint, preventing overlap and accidental component
     merging.
   - For L groups, payload cells are sampled strictly on one side of the implied
     diagonal, so the mirrored copy is a new visible addition instead of landing
     back on the source.

e) Visual observability
   - Samples mix orthogonal and diagonal markers in one grid, so a human can
     compare "row/col line -> orthogonal mirror" against "L shape -> diagonal
     mirror" directly.
   - L markers are generated in all four corner orientations, which makes the
     y=x vs y=-x distinction inferable from a few examples.

f) Visual cleanliness
   - No decoy sprites are added. Unique payload colors and padded placement keep
     each marker-payload group locally readable.
   - Payloads are asymmetric connected shapes, so their reflected copy is visually
     informative instead of collapsing into a trivial symmetric blob.

g) New solver assumptions and generator guarantees
   - Assumption: an L marker is a true one-corner union of two contiguous one-sided
     arms. Generator builds exactly that and never emits T/+/2x2 red shapes.
   - Assumption: every mixed payload belongs to the same 4-connected component as
     its red marker. Generator seeds payload growth from cells adjacent to the
     marker.
   - Assumption: no two groups share a payload color. Generator uses distinct
     colors per group, so component-local reflection is unambiguous.
"""


MARK = 2


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


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
    red_set = {(int(r), int(c)) for r, c in reds}
    if not red_set:
        return None
    if len(red_set) == 1:
        pr, pc = next(iter(red_set))
        return ("point", (int(pr), int(pc)))

    rs = {int(r) for r, _c in red_set}
    cs = {int(c) for _r, c in red_set}

    if len(rs) == 1:
        row = int(next(iter(rs)))
        cols = sorted(int(c) for _r, c in red_set)
        if cols == list(range(cols[0], cols[0] + len(cols))):
            return ("row", row)

    if len(cs) == 1:
        col = int(next(iter(cs)))
        rows = sorted(int(r) for r, _c in red_set)
        if rows == list(range(rows[0], rows[0] + len(rows))):
            return ("col", col)

    for pr, pc in sorted(red_set):
        row_arm = sorted(int(c) for r, c in red_set if int(r) == int(pr))
        col_arm = sorted(int(r) for r, c in red_set if int(c) == int(pc))
        if len(row_arm) < 2 or len(col_arm) < 2:
            continue
        if not all(int(r) == int(pr) or int(c) == int(pc) for r, c in red_set):
            continue
        h_signs = {int(np.sign(int(c) - int(pc))) for c in row_arm if int(c) != int(pc)}
        v_signs = {int(np.sign(int(r) - int(pr))) for r in col_arm if int(r) != int(pr)}
        if len(h_signs) != 1 or len(v_signs) != 1:
            continue
        h_dir = int(next(iter(h_signs)))
        v_dir = int(next(iter(v_signs)))
        if h_dir == 0 or v_dir == 0:
            continue
        expected_row = {(int(pr), int(pc + h_dir * k)) for k in range(len(row_arm))}
        expected_col = {(int(pr + v_dir * k), int(pc)) for k in range(len(col_arm))}
        expected = expected_row | expected_col
        if expected != red_set:
            continue
        diag_kind = "diag_main" if h_dir == v_dir else "diag_anti"
        return ("diag", (int(pr), int(pc), str(diag_kind)))

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
        comps.append({"coords": coords, "vals": vals})

    out = g.copy()

    # Mixed marker components are handled component-locally. This is stricter and
    # more robust than the original bbox scan, and the generator guarantees that a
    # payload truly belongs to exactly one marker component.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        axis = _classify_marker(reds)
        if axis is None:
            continue

        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not payload:
            continue

        for r, c, v in payload:
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

    # Preserve the original "isolated red pixel mirrors an adjacent 8-connected
    # monochrome component" rule for robustness, though this generator does not
    # rely on it.
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


def _normalize_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(int(r) for r, _c, _v in cells)
    min_c = min(int(c) for _r, c, _v in cells)
    return [(int(r - min_r), int(c - min_c), int(v)) for r, c, v in cells]


def _rotate_local(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    turns = int(turns) % 4
    out = [(int(r), int(c)) for r, c in coords]
    for _ in range(turns):
        out = [(int(c), int(-r)) for r, c in out]
    return out


def _random_connected_shape(
    rng: np.random.Generator,
    candidates: set[tuple[int, int]],
    size_range: tuple[int, int],
    *,
    seed_filter,
    reflect_fn,
) -> set[tuple[int, int]]:
    all_candidates = {tuple(map(int, p)) for p in candidates}
    seeds = [p for p in sorted(all_candidates) if seed_filter(p)]
    if not seeds:
        raise ValueError("no legal seeds")

    lo, hi = map(int, size_range)
    hi = min(int(hi), len(all_candidates))
    if lo > hi:
        raise ValueError("invalid size range")

    for _ in range(300):
        size = int(rng.integers(lo, hi + 1))
        shape = {seeds[int(rng.integers(0, len(seeds)))]}
        while len(shape) < size:
            frontier = set()
            for cell in shape:
                for nxt in _neighbors4(cell):
                    if nxt in all_candidates and nxt not in shape:
                        frontier.add(nxt)
            if not frontier:
                break
            options = sorted(frontier)
            shape.add(options[int(rng.integers(0, len(options)))])
        if len(shape) != size:
            continue
        if any(tuple(map(int, reflect_fn(r, c))) in shape for r, c in shape):
            continue
        rows = [int(r) for r, _c in shape]
        cols = [int(c) for _r, c in shape]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue
        return shape

    raise RuntimeError("failed to grow a connected asymmetric shape")


def _build_group_row(rng: np.random.Generator, color: int) -> dict[str, object]:
    line_len = int(rng.integers(3, 7))
    depth = int(rng.integers(3, 6))
    candidates = {(int(r), int(c)) for r in range(1, depth + 1) for c in range(line_len)}
    shape = _random_connected_shape(
        rng,
        candidates,
        (3, min(8, len(candidates))),
        seed_filter=lambda p: int(p[0]) == 1,
        reflect_fn=lambda r, c: _refl_row(0, int(r), int(c)),
    )
    reds = [(0, c, MARK) for c in range(line_len)]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [(_refl_row(0, int(r), int(c))[0], _refl_row(0, int(r), int(c))[1], int(color)) for r, c in shape]
    if bool(rng.integers(0, 2)):
        reds = [(-int(r), int(c), int(v)) for r, c, v in reds]
        payload = [(-int(r), int(c), int(v)) for r, c, v in payload]
        mirror = [(-int(r), int(c), int(v)) for r, c, v in mirror]
    return {
        "type": "row",
        "in_cells": _normalize_cells(reds + payload),
        "out_cells": _normalize_cells(reds + payload + mirror),
    }


def _build_group_col(rng: np.random.Generator, color: int) -> dict[str, object]:
    line_len = int(rng.integers(3, 7))
    width = int(rng.integers(3, 6))
    candidates = {(int(r), int(c)) for r in range(line_len) for c in range(1, width + 1)}
    shape = _random_connected_shape(
        rng,
        candidates,
        (3, min(8, len(candidates))),
        seed_filter=lambda p: int(p[1]) == 1,
        reflect_fn=lambda r, c: _refl_col(0, int(r), int(c)),
    )
    reds = [(r, 0, MARK) for r in range(line_len)]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [(_refl_col(0, int(r), int(c))[0], _refl_col(0, int(r), int(c))[1], int(color)) for r, c in shape]
    if bool(rng.integers(0, 2)):
        reds = [(int(r), -int(c), int(v)) for r, c, v in reds]
        payload = [(int(r), -int(c), int(v)) for r, c, v in payload]
        mirror = [(int(r), -int(c), int(v)) for r, c, v in mirror]
    return {
        "type": "col",
        "in_cells": _normalize_cells(reds + payload),
        "out_cells": _normalize_cells(reds + payload + mirror),
    }


def _build_group_point(rng: np.random.Generator, color: int) -> dict[str, object]:
    candidates = {(int(r), int(c)) for r in range(-2, 3) for c in range(1, 5)}
    shape = _random_connected_shape(
        rng,
        candidates,
        (3, min(7, len(candidates))),
        seed_filter=lambda p: int(p[1]) == 1,
        reflect_fn=lambda r, c: _refl_point((0, 0), int(r), int(c)),
    )
    turns = int(rng.integers(0, 4))
    shape = set(_rotate_local(list(shape), turns))
    reds = [(0, 0, MARK)]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [(_refl_point((0, 0), int(r), int(c))[0], _refl_point((0, 0), int(r), int(c))[1], int(color)) for r, c in shape]
    return {
        "type": "point",
        "in_cells": _normalize_cells(reds + payload),
        "out_cells": _normalize_cells(reds + payload + mirror),
    }


def _build_group_diag(rng: np.random.Generator, color: int) -> dict[str, object]:
    arm_len_h = int(rng.integers(3, 6))
    arm_len_v = int(rng.integers(3, 6))
    span = max(int(arm_len_h), int(arm_len_v), 4) + 1
    candidates = {(int(r), int(c)) for r in range(1, span + 1) for c in range(1, span + 1) if int(c) > int(r)}
    shape = _random_connected_shape(
        rng,
        candidates,
        (4, min(8, len(candidates))),
        seed_filter=lambda p: int(p[0]) == 1,
        reflect_fn=lambda r, c: _refl_diag("diag_main", (0, 0), int(r), int(c)),
    )
    turns = int(rng.integers(0, 4))
    shape = set(_rotate_local(list(shape), turns))

    reds_local = [(0, c) for c in range(arm_len_h)] + [(r, 0) for r in range(1, arm_len_v)]
    reds_rot = _rotate_local(reds_local, turns)
    reds = [(int(r), int(c), MARK) for r, c in reds_rot]
    reds_rc = [(int(r), int(c)) for r, c, _v in reds]
    axis = _classify_marker(reds_rc)
    if axis is None or axis[0] != "diag":
        raise RuntimeError("generated L marker failed classification")

    pr, pc, diag_kind = axis[1]
    payload = [(int(r), int(c), int(color)) for r, c in shape]
    mirror = [
        (_refl_diag(str(diag_kind), (int(pr), int(pc)), int(r), int(c))[0],
         _refl_diag(str(diag_kind), (int(pr), int(pc)), int(r), int(c))[1],
         int(color))
        for r, c in shape
    ]
    occupied = {(int(r), int(c)) for r, c, _v in reds + payload}
    if any((int(r), int(c)) in occupied for r, c, _v in mirror):
        raise RuntimeError("diagonal mirror overlapped source")
    return {
        "type": "diag",
        "diag_kind": str(diag_kind),
        "in_cells": _normalize_cells(reds + payload),
        "out_cells": _normalize_cells(reds + payload + mirror),
    }


def _place_group(
    rng: np.random.Generator,
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    group: dict[str, object],
) -> bool:
    out_cells = list(group["out_cells"])
    in_cells = list(group["in_cells"])
    gh = 1 + max(int(r) for r, _c, _v in out_cells)
    gw = 1 + max(int(c) for _r, c, _v in out_cells)
    h, w = inp.shape

    for _ in range(500):
        top = int(rng.integers(0, h - gh + 1))
        left = int(rng.integers(0, w - gw + 1))
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
        return True

    return False


def generate_case(
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng() if rng is None else rng

    for _attempt in range(600):
        h = int(rng.integers(20, 29))
        w = int(rng.integers(20, 29))
        bg_choices = [c for c in range(10) if c != MARK]
        bg = int(bg_choices[int(rng.integers(0, len(bg_choices)))])
        palette = [c for c in range(10) if c not in (bg, MARK)]

        base_kinds = ["row", "col", "point", "diag"]
        extra_pool = ["diag", "diag", "row", "col", "point"]
        extra_count = int(rng.integers(0, 3))
        extras = [str(extra_pool[int(rng.integers(0, len(extra_pool)))]) for _ in range(extra_count)]
        kinds = base_kinds + extras
        if len(kinds) > len(palette):
            continue

        perm = rng.permutation(len(palette))
        chosen_colors = [int(palette[int(i)]) for i in perm[: len(kinds)]]

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        meta_groups: list[dict[str, object]] = []
        ok = True

        for idx, kind in enumerate(kinds):
            color = int(chosen_colors[idx])
            try:
                if kind == "row":
                    group = _build_group_row(rng, color)
                elif kind == "col":
                    group = _build_group_col(rng, color)
                elif kind == "point":
                    group = _build_group_point(rng, color)
                else:
                    group = _build_group_diag(rng, color)
            except RuntimeError:
                ok = False
                break

            if not _place_group(rng, inp, out, occupied, group):
                ok = False
                break

            meta = {"type": str(kind), "color": int(color)}
            if "diag_kind" in group:
                meta["diag_kind"] = str(group["diag_kind"])
            meta_groups.append(meta)

        if not ok:
            continue

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        diag_groups = [g for g in meta_groups if g["type"] == "diag"]
        meta = {
            "shape": (int(h), int(w)),
            "bg": int(bg),
            "groups": meta_groups,
            "diag_groups": len(diag_groups),
            "diag_main_groups": sum(1 for g in diag_groups if g.get("diag_kind") == "diag_main"),
            "diag_anti_groups": sum(1 for g in diag_groups if g.get("diag_kind") == "diag_anti"),
            "total_groups": len(meta_groups),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid combo sample")


def generate_input(
    rng: np.random.Generator | None = None,
    *,
    return_output: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    inp, out, _meta = generate_case(rng)
    if return_output:
        return inp, out
    return inp
