# Variant of 7ed72f31
# Dimension: D4 — 标记类型的判定逻辑 (transform)
# Change: 红像素数=1→点反射；同行多个→水平轴反射；同列多个→竖直轴反射... → 反转判定逻辑：线→点反射，点→轴反射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask

"""
design_analysis:
  variant_core:
    original:
      - single red pixel inside a mixed component => point reflection around that pixel
      - horizontal / vertical red line => mirror across that row / column
      - isolated single red pixel => point-reflect its adjacent 8-connected sprite
    variant:
      - single red pixel => axis reflection instead of point reflection
      - red line => point reflection instead of axis reflection
      - for a single red pixel, axis direction is inferred from where the payload sits
      - for a red line, the point-reflection center is the midpoint of the line segment
  concretization_and_trace:
    single_red_axis:
      input_shape:
        - one red pixel touching a monochrome payload by 4-neighborhood
        - payload is entirely on one cardinal side of the red pixel
      output_shape:
        - original payload remains
        - a reflected copy appears across the row through the red pixel if payload is above/below
        - a reflected copy appears across the column through the red pixel if payload is left/right
      trace:
        old_logic: a payload cell at relative offset (dr, dc) becomes (-dr, -dc)
        new_logic_row_axis: the same cell becomes (-dr, dc)
        difference: horizontal displacement is no longer negated, so the copy keeps its left/right offsets
    red_line_point:
      input_shape:
        - a horizontal or vertical contiguous red line, 4-connected to a monochrome payload on one side
      output_shape:
        - original payload remains
        - a 180-degree rotated copy appears around the midpoint of the line
      trace:
        horizontal_example:
          line: row r, cols c0..c1
          old_logic: (rr, cc) -> (2*r-rr, cc)
          new_logic: (rr, cc) -> (2*r-rr, c0+c1-cc)
          difference: the reflected copy moves to the opposite side and also flips along the line direction
  complexity_calibration:
    - the generator always includes at least one single-red group and at least one red-line group
    - shapes are forced to have span in the dimension that distinguishes new behavior from old behavior
    - colors are unique per group so solver bbox scanning cannot accidentally merge groups by color
  implicit_assumptions_review:
    - original `len(reds)==1 => point` is invalid; now it must infer `row` or `col`
    - original `len(rs)==1 => row axis` and `len(cs)==1 => col axis` are invalid; now those cases are point centers
    - original isolated-marker pass assumed point reflection; it now reuses the same axis inference as the mixed single-red case
    - single-red axis inference is only reliable when the payload is clearly on one side; generator guarantees that
    - line midpoint may fall on half-integer coordinates for even-length lines, so solver uses integer numerators `(num_r, num_c)` instead of floats
  placement_legality:
    - output-first generation places full output groups first, then erases mirrored copies to derive the input
    - group footprints are kept disjoint with a one-cell buffer to avoid overlaps and visual ambiguity
    - mirrored copies are checked to stay in-bounds by construction because the whole output footprint is placed as a unit
  visual_observability:
    - single-red groups are generated with width>1 for row-axis or height>1 for col-axis so axis reflection is visibly different from old point reflection
    - red-line groups are generated so point reflection is visibly different from old line-axis mirroring
    - diagonal-only isolated single-red groups are omitted in generation because under the inverted rule their axis direction would be ambiguous to a human
  cleanliness:
    - no extra distractor sprites are generated; this keeps the D4 inversion visually legible in 2-4 examples
  new_main_assumptions_and_generator_support:
    - mixed single-red components need a cardinally attached payload; generator always seeds one anchor cell adjacent to the red pixel
    - line markers need to be contiguous horizontal or vertical segments; generator always builds contiguous lines
    - same-color clutter near a marker would trigger original-style bbox capture; generator avoids this by assigning unique payload colors
"""

MARK = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _inb(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _refl_point_num(num_r: int, num_c: int, r: int, c: int) -> tuple[int, int]:
    return int(num_r) - int(r), int(num_c) - int(c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return 2 * int(r0) - int(r), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), 2 * int(c0) - int(c)


def _infer_single_red_axis(pr: int, pc: int, coords: list[tuple[int, int]]) -> tuple[str, int] | None:
    if not coords:
        return None
    above = any(int(r) < int(pr) for r, _c in coords)
    below = any(int(r) > int(pr) for r, _c in coords)
    left = any(int(c) < int(pc) for _r, c in coords)
    right = any(int(c) > int(pc) for _r, c in coords)

    vert_one_side = bool(above) ^ bool(below)
    horiz_one_side = bool(left) ^ bool(right)

    if vert_one_side and not horiz_one_side:
        return ("row", int(pr))
    if horiz_one_side and not vert_one_side:
        return ("col", int(pc))

    has_cardinal_vertical = any(int(c) == int(pc) and abs(int(r) - int(pr)) == 1 for r, c in coords)
    has_cardinal_horizontal = any(int(r) == int(pr) and abs(int(c) - int(pc)) == 1 for r, c in coords)

    if has_cardinal_vertical and not has_cardinal_horizontal:
        return ("row", int(pr))
    if has_cardinal_horizontal and not has_cardinal_vertical:
        return ("col", int(pc))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    # --- 1) Marker-containing mixed components ---
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_colors = {int(v) for v in vals if int(v) != int(MARK)}
        payload_coords = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload_colors or not payload_coords:
            continue

        mode: tuple[str, int, int] | tuple[str, int] | None
        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            inferred = _infer_single_red_axis(int(reds[0][0]), int(reds[0][1]), payload_coords)
            mode = inferred
        elif len(rs) == 1 and len(reds) == max(cs) - min(cs) + 1:
            mode = ("point", 2 * int(next(iter(rs))), int(min(cs) + max(cs)))
        elif len(cs) == 1 and len(reds) == max(rs) - min(rs) + 1:
            mode = ("point", int(min(rs) + max(rs)), 2 * int(next(iter(cs))))
        else:
            mode = None
        if mode is None:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        if mode[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload_colors and int(c0) <= int(c) <= int(c1)

        elif mode[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload_colors and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload_colors and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[r, c])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if mode[0] == "row":
                    rr, cc = _refl_row(int(mode[1]), int(r), int(c))
                elif mode[0] == "col":
                    rr, cc = _refl_col(int(mode[1]), int(r), int(c))
                else:
                    rr, cc = _refl_point_num(int(mode[1]), int(mode[2]), int(r), int(c))

                if _inb(H, W, int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    # --- 2) Isolated single red pixels: reflect the adjacent color component across an inferred axis ---
    marker_pts = []
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
                if not _inb(H, W, rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                comp_coords = [(int(r), int(c)) for r, c in np.argwhere(labv == int(lbl))]
                axis = _infer_single_red_axis(int(pr), int(pc), comp_coords)
                if axis is None:
                    continue
                for r, c in comp_coords:
                    if axis[0] == "row":
                        rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                    else:
                        rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                    if _inb(H, W, int(rr), int(cc)):
                        out[int(rr), int(cc)] = int(v)

    return out


def _grow_connected_shape(
    anchor: tuple[int, int],
    size: int,
    allowed,
    *,
    max_tries: int = 400,
) -> set[tuple[int, int]] | None:
    cells = {tuple(map(int, anchor))}
    tries = 0
    while len(cells) < int(size) and tries < int(max_tries):
        base_r, base_c = random.choice(tuple(cells))
        dr, dc = random.choice(DIR4)
        cand = (int(base_r + dr), int(base_c + dc))
        if cand not in cells and allowed(int(cand[0]), int(cand[1])):
            cells.add(cand)
        tries += 1
    if len(cells) != int(size):
        return None
    return cells


def _bbox_of_cells(cells: dict[tuple[int, int], int]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _shift_cells(cells: dict[tuple[int, int], int], dr: int, dc: int) -> dict[tuple[int, int], int]:
    return {(int(r) + int(dr), int(c) + int(dc)): int(v) for (r, c), v in cells.items()}


def _build_dot_axis_group(color: int) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[str, object]]:
    for _ in range(200):
        orient = random.choice(["row", "col"])
        side = random.choice([-1, 1])
        size = random.randint(3, 8)
        span = random.randint(2, 4)
        depth = random.randint(2, 4)
        if orient == "row":
            anchor = (int(side), 0)

            def allowed(r: int, c: int) -> bool:
                return 1 <= int(side) * int(r) <= int(depth) and -int(span) <= int(c) <= int(span)

            shape = _grow_connected_shape(anchor, size, allowed)
            if shape is None:
                continue
            width = max(c for _r, c in shape) - min(c for _r, c in shape) + 1
            if width < 2:
                continue
            new_mirror = {(-int(r), int(c)) for r, c in shape}
            old_mirror = {(-int(r), -int(c)) for r, c in shape}
        else:
            anchor = (0, int(side))

            def allowed(r: int, c: int) -> bool:
                return 1 <= int(side) * int(c) <= int(depth) and -int(span) <= int(r) <= int(span)

            shape = _grow_connected_shape(anchor, size, allowed)
            if shape is None:
                continue
            height = max(r for r, _c in shape) - min(r for r, _c in shape) + 1
            if height < 2:
                continue
            new_mirror = {(int(r), -int(c)) for r, c in shape}
            old_mirror = {(-int(r), -int(c)) for r, c in shape}

        if new_mirror == old_mirror:
            continue

        inp = {(0, 0): MARK}
        out = {(0, 0): MARK}
        for cell in shape:
            inp[tuple(map(int, cell))] = int(color)
            out[tuple(map(int, cell))] = int(color)
        for cell in new_mirror:
            out[tuple(map(int, cell))] = int(color)
        meta = {"group_type": "dot_axis", "orientation": orient, "differs_from_original": True}
        return inp, out, meta
    raise RuntimeError("failed to build dot-axis group")


def _build_line_point_group(color: int) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[str, object]]:
    for _ in range(200):
        orient = random.choice(["hline", "vline"])
        side = random.choice([-1, 1])
        line_len = random.randint(2, 6)
        size = random.randint(3, 9)
        span = random.randint(1, 3)
        depth = random.randint(2, 4)

        if orient == "hline":
            anchor = (int(side), random.randint(0, int(line_len) - 1))

            def allowed(r: int, c: int) -> bool:
                return 1 <= int(side) * int(r) <= int(depth) and -int(span) <= int(c) <= int(line_len - 1 + span)

            shape = _grow_connected_shape(anchor, size, allowed)
            if shape is None:
                continue
            new_mirror = {(-int(r), int(line_len - 1 - c)) for r, c in shape}
            old_mirror = {(-int(r), int(c)) for r, c in shape}
            reds = {(0, c) for c in range(int(line_len))}
        else:
            anchor = (random.randint(0, int(line_len) - 1), int(side))

            def allowed(r: int, c: int) -> bool:
                return 1 <= int(side) * int(c) <= int(depth) and -int(span) <= int(r) <= int(line_len - 1 + span)

            shape = _grow_connected_shape(anchor, size, allowed)
            if shape is None:
                continue
            new_mirror = {(int(line_len - 1 - r), -int(c)) for r, c in shape}
            old_mirror = {(int(r), -int(c)) for r, c in shape}
            reds = {(r, 0) for r in range(int(line_len))}

        if new_mirror == old_mirror:
            continue

        inp = {tuple(map(int, cell)): MARK for cell in reds}
        out = {tuple(map(int, cell)): MARK for cell in reds}
        for cell in shape:
            inp[tuple(map(int, cell))] = int(color)
            out[tuple(map(int, cell))] = int(color)
        for cell in new_mirror:
            out[tuple(map(int, cell))] = int(color)
        meta = {
            "group_type": "line_point",
            "orientation": orient,
            "line_len": int(line_len),
            "differs_from_original": True,
        }
        return inp, out, meta
    raise RuntimeError("failed to build line-point group")


def _can_place(coords: list[tuple[int, int]], occupied: set[tuple[int, int]], H: int, W: int) -> bool:
    for r, c in coords:
        if not _inb(H, W, int(r), int(c)):
            return False
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if (int(rr), int(cc)) in occupied:
                    return False
    return True


def _paint(grid: np.ndarray, cells: dict[tuple[int, int], int]) -> None:
    for (r, c), v in cells.items():
        grid[int(r), int(c)] = int(v)


def _generate_case(return_meta: bool = False):
    colors_all = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    for _case_try in range(200):
        H = random.randint(18, 28)
        W = random.randint(18, 28)
        bg = random.choice(colors_all)
        usable_colors = [c for c in colors_all if c != int(bg)]
        random.shuffle(usable_colors)
        n_groups = random.randint(3, min(6, len(usable_colors)))
        kinds = ["dot_axis", "line_point"]
        while len(kinds) < int(n_groups):
            kinds.append(random.choice(["dot_axis", "line_point"]))
        random.shuffle(kinds)

        inp = np.full((H, W), int(bg), dtype=int)
        out = np.full((H, W), int(bg), dtype=int)
        occupied: set[tuple[int, int]] = set()
        meta: list[dict[str, object]] = []
        success = True

        for idx, kind in enumerate(kinds):
            color = int(usable_colors[idx])
            if kind == "dot_axis":
                local_inp, local_out, info = _build_dot_axis_group(color)
            else:
                local_inp, local_out, info = _build_line_point_group(color)

            min_r, min_c, max_r, max_c = _bbox_of_cells(local_out)
            row_lo, row_hi = -int(min_r), int(H - 1 - max_r)
            col_lo, col_hi = -int(min_c), int(W - 1 - max_c)
            if row_lo > row_hi or col_lo > col_hi:
                success = False
                break

            placed = False
            for _place_try in range(300):
                off_r = random.randint(int(row_lo), int(row_hi))
                off_c = random.randint(int(col_lo), int(col_hi))
                shifted_out = _shift_cells(local_out, off_r, off_c)
                coords = list(shifted_out.keys())
                if not _can_place(coords, occupied, H, W):
                    continue
                shifted_inp = _shift_cells(local_inp, off_r, off_c)
                _paint(inp, shifted_inp)
                _paint(out, shifted_out)
                occupied.update((int(r), int(c)) for r, c in coords)
                info = dict(info)
                info["offset"] = (int(off_r), int(off_c))
                meta.append(info)
                placed = True
                break

            if not placed:
                success = False
                break

        if success and np.array_equal(main(inp), out):
            if return_meta:
                return inp, out, meta
            return inp

    raise RuntimeError("failed to generate a valid D4 variant sample")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
