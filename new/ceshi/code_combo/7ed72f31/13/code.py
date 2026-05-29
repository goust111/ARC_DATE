# Combo variant of 7ed72f31 (index 13)
# combo_spec: D4/0+D12/2
# Name: 反直觉判定+虚假标记

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "solvers" / "arc_utils.py").exists():
    if ROOT == ROOT.parent:
        raise RuntimeError("could not locate repo root")
    ROOT = ROOT.parent
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import get_bg, label_mask

"""
design_analysis:
  fusion_summary:
    - D4/0 flips the marker interpretation: a single red pixel means axis reflection, while a red line means point reflection.
    - D12/2 adds fake red marker groups with no payload; these markers must be ignored.
    - The fused task therefore has two steps: first decide whether a red group is valid at all, then apply the inverted reflection rule.
  concretization_and_trace:
    valid_single_red:
      input_shape:
        - one red pixel is 4-connected to a monochrome payload
        - the payload lies entirely on one cardinal side of the red pixel
      fused_behavior:
        - if the payload is above or below the red pixel, reflect across the row through the red pixel
        - if the payload is left or right of the red pixel, reflect across the column through the red pixel
      contrast_with_original:
        - original task would point-reflect around the red pixel
        - fused task keeps the parallel coordinate unchanged, so the mirrored copy sits in a visibly different place
    valid_red_line:
      input_shape:
        - a contiguous horizontal or vertical red line is 4-connected to a payload on one side
      fused_behavior:
        - reflect the payload by 180 degrees around the midpoint of the line segment
      contrast_with_original:
        - original task would mirror only across the line axis
        - fused task also flips along the line direction, so the copy is not just a simple mirror
    fake_markers:
      input_shape:
        - red-only dot or red-only line with no payload attached
      fused_behavior:
        - ignored entirely; input and output are identical around that marker
      contrast_with_original:
        - these groups are pure distractors and exist only to force validity checking
  complexity_calibration:
    - every generated sample contains at least one valid single-red group, at least one valid red-line group, and at least one fake red marker
    - valid groups use distinct payload colors so color cannot be the hidden cue for deciding which marker is active
    - shapes are forced to span the axis-distinguishing dimension so inverted logic is visually observable in 2-4 examples
    - fake markers can be either dots or lines, so the solver cannot classify validity by geometry alone
  implicit_assumptions_review:
    - original assumption `single red => point reflection` is invalid; solver must infer row-vs-column axis from payload placement
    - original assumption `red line => axis reflection` is invalid; solver must treat the line midpoint as a point-reflection center
    - original assumption `every red marker participates` is invalid; payload-free red groups must be skipped
    - isolated single-red handling remains possible, but now uses axis inference too; generator intentionally omits valid isolated dots to keep the dataset legible
    - line midpoints can be half-integer when line length is even, so the solver stores reflection centers as doubled numerators `(num_r, num_c)` instead of floats
  placement_legality:
    - generation is output-first per group: build the full transformed footprint, place it on the board, then erase mirrored-only pixels to form the input footprint
    - groups are placed with a one-cell buffer, which prevents overlap, accidental adjacency, and fake-dot misfires in the isolated-marker pass
    - fake dots are only placed where all 8 neighbors are background in the final input
  visual_observability:
    - single-red valid groups always have width or height greater than 1 in the direction that separates axis reflection from point reflection
    - line groups always differ from the original solver's output because the mirrored copy is flipped along the line direction as well
    - fake markers are sparse and separated, so a human can see that some red cues cause duplication while others do nothing
  cleanliness:
    - no extra same-color clutter is generated beyond the intended reflected groups
    - valid and fake marker groups are spatially separated by construction, so the board remains readable
  quantifiable_combo_checks:
    - metric_1_fake_markers: number of payload-free red groups; should be >= 1 in every generated sample
    - metric_2_logic_gap_pixels: Hamming distance between this solver's output and the original task solver on the same input; should be > 0 in every generated sample
  new_main_assumptions_and_generator_support:
    - for valid single-red groups, payload cells are on one cardinal side of the dot; `_build_dot_axis_group` guarantees this
    - for valid line groups, red cells form a contiguous straight segment; `_build_line_point_group` guarantees this
    - fake markers have no payload in the same 4-connected component, and fake dots have no adjacent 8-connected payload; `_build_fake_marker` plus buffered placement guarantee this
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


def _classify_valid_marker(
    reds: list[tuple[int, int]],
    payload_coords: list[tuple[int, int]],
) -> tuple[str, int, int] | tuple[str, int] | None:
    if not reds or not payload_coords:
        return None

    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}
    if len(reds) == 1:
        return _infer_single_red_axis(int(reds[0][0]), int(reds[0][1]), payload_coords)
    if len(rs) == 1 and len(reds) == max(cs) - min(cs) + 1:
        return ("point", 2 * int(next(iter(rs))), int(min(cs) + max(cs)))
    if len(cs) == 1 and len(reds) == max(rs) - min(rs) + 1:
        return ("point", int(min(rs) + max(rs)), 2 * int(next(iter(cs))))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    # Pass 1: only mixed components with both red marker and payload are valid.
    # Payload-free red components are fake markers and are ignored here.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_cells = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        payload_coords = [(int(r), int(c)) for r, c, _v in payload_cells]
        mode = _classify_valid_marker(reds, payload_coords)
        if mode is None:
            continue

        for r, c, v in payload_cells:
            if mode[0] == "row":
                rr, cc = _refl_row(int(mode[1]), int(r), int(c))
            elif mode[0] == "col":
                rr, cc = _refl_col(int(mode[1]), int(r), int(c))
            else:
                rr, cc = _refl_point_num(int(mode[1]), int(mode[2]), int(r), int(c))
            if _inb(H, W, int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    # Pass 2: isolated single red pixels can still be valid, but only if they have
    # an adjacent 8-connected payload component. Fake dots remain inert.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue

        pr, pc = map(int, coords[0])
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
            if max(c for _r, c in shape) - min(c for _r, c in shape) + 1 < 2:
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
            if max(r for r, _c in shape) - min(r for r, _c in shape) + 1 < 2:
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
        return inp, out, {"group_type": "dot_axis", "orientation": orient}
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
        return inp, out, {"group_type": "line_point", "orientation": orient, "line_len": int(line_len)}
    raise RuntimeError("failed to build line-point group")


def _build_fake_marker() -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[str, object]]:
    kind = random.choice(["fake_dot", "fake_hline", "fake_vline"])
    if kind == "fake_dot":
        cells = {(0, 0): MARK}
    elif kind == "fake_hline":
        line_len = random.randint(2, 6)
        cells = {(0, c): MARK for c in range(int(line_len))}
    else:
        line_len = random.randint(2, 6)
        cells = {(r, 0): MARK for r in range(int(line_len))}
    return dict(cells), dict(cells), {"group_type": kind, "is_fake": True}


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
    for _case_try in range(300):
        H = random.randint(18, 28)
        W = random.randint(18, 28)
        bg = random.choice(colors_all)
        usable_colors = [c for c in colors_all if c != int(bg)]
        random.shuffle(usable_colors)

        n_valid = random.randint(3, min(5, len(usable_colors)))
        valid_kinds = ["dot_axis", "line_point"]
        while len(valid_kinds) < int(n_valid):
            valid_kinds.append(random.choice(["dot_axis", "line_point"]))
        random.shuffle(valid_kinds)
        n_fake = random.randint(1, 3)

        inp = np.full((H, W), int(bg), dtype=int)
        out = np.full((H, W), int(bg), dtype=int)
        occupied: set[tuple[int, int]] = set()
        meta: list[dict[str, object]] = []
        success = True

        for idx, kind in enumerate(valid_kinds):
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
                if not _can_place(list(shifted_out.keys()), occupied, H, W):
                    continue
                shifted_inp = _shift_cells(local_inp, off_r, off_c)
                _paint(inp, shifted_inp)
                _paint(out, shifted_out)
                occupied.update((int(r), int(c)) for r, c in shifted_out)
                info = dict(info)
                info["offset"] = (int(off_r), int(off_c))
                info["color"] = int(color)
                meta.append(info)
                placed = True
                break
            if not placed:
                success = False
                break

        if not success:
            continue

        for _fake_idx in range(int(n_fake)):
            local_inp, local_out, info = _build_fake_marker()
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
                if not _can_place(list(shifted_out.keys()), occupied, H, W):
                    continue
                shifted_inp = _shift_cells(local_inp, off_r, off_c)
                _paint(inp, shifted_inp)
                _paint(out, shifted_out)
                occupied.update((int(r), int(c)) for r, c in shifted_out)
                info = dict(info)
                info["offset"] = (int(off_r), int(off_c))
                meta.append(info)
                placed = True
                break
            if not placed:
                success = False
                break

        if not success:
            continue

        solved = main(inp)
        if np.array_equal(solved, out):
            if return_meta:
                return inp, out, meta
            return inp

    raise RuntimeError("failed to generate a valid combo sample")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
