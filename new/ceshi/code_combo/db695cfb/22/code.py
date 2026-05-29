# Combo variant of db695cfb (index 22)
# combo_spec: D7/1+D8/2
# Name: 色块触发+成对品红

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The carrier structure from the seed task stays the same: blue endpoint pairs define
#      45-degree diagonal segments that must be completed in the output.
#    - The trigger is no longer a single magenta pixel. It is always a 2x2 magenta square.
#      A square only matters if the completed blue diagonal passes through exactly its two
#      diagonal corner cells.
#    - Activated squares are not processed independently. Along each blue segment, activated
#      squares are ordered by their touched diagonal interval and paired greedily by the
#      smallest adjacent gap. Any leftover square is inert.
#    - For a matched pair, every touched cell of both squares emits perpendicular diagonal
#      rays. The ray length is the number of empty diagonal cells strictly between the two
#      squares.
#    - Example on one backslash segment:
#      square A touches path indices [2,3], square B touches [7,8], so there are indices
#      [4,5,6] between them. Seed task: each touched magenta cell would shoot to the border.
#      D7-only: each square could fire by shape alone. D8-only on raw pixels could pair the
#      two cells inside one square. This combo instead first groups pixels into squares, then
#      pairs A with B, then uses the block-to-block gap 3 as the ray length.
#
# b) Complexity calibration:
#    - Only one trigger shape class is used for activation: 2x2 squares. That keeps the
#      object-recognition burden small while still forcing component parsing.
#    - Pair gaps are limited to 2..4 cells, producing a small repeated length vocabulary.
#    - Each segment carries at most one matched pair or one lone square. This avoids the
#      longest-diagonal color heuristic being hijacked by magenta and keeps pairing local.
#
# c) Implicit-assumption audit of the seed solver:
#    - Seed assumption "every non-blue diagonal pixel is an independent trigger" fails.
#      We must collect 4-connected components and keep only 2x2 squares.
#    - Seed assumption "a trigger fires as soon as the draw loop touches it" also fails.
#      Here activation depends on the relative arrangement of multiple blocks, so the solver
#      records all activated squares on a segment first and pairs them afterwards.
#    - Seed assumption "all rays go to the boundary" fails. The combo uses bounded rays whose
#      length depends on the block-to-block gap.
#    - New assumption introduced by the combo: an activating square contributes a contiguous
#      two-cell touched interval on the carrier diagonal. The generator guarantees that.
#
# d) Placement legality:
#    - Magenta squares must stay 4-disconnected so components never merge.
#    - Rays are planned output-first; any candidate whose rays hit existing input pixels,
#      reserved output cells, or untouched blue-path cells is rejected.
#    - Cross-segment magenta diagonal alignments are forbidden, so line-color detection still
#      prefers the longer blue endpoint pair rather than accidental magenta alignments.
#
# e) Visual observability:
#    - The input visibly contains 2x2 magenta blocks rather than single pixels.
#    - The output visibly distinguishes matched versus lone squares: matched squares grow
#      perpendicular stubs, lone squares remain unchanged.
#    - Gap size is inferable by counting the empty diagonal spacing between the two squares,
#      then checking the perpendicular stub length.
#
# f) Visual cleanliness:
#    - Segment count stays modest and each segment uses at most one block group.
#    - Full ray lengths must fit in bounds, avoiding confusing truncation.
#    - Off-diagonal distractors use the same 2x2 shape but never intersect a carrier segment.
#
# g) Color reasoning:
#    - No hidden color ordering is used. Blue remains the carrier color, magenta remains the
#      reactive object color, and background is still the most frequent color.
#
# h) Generator guarantees for new solver assumptions:
#    - Every activating magenta component is an isolated 2x2 square touched in exactly two
#      consecutive diagonal cells.
#    - On a single segment there is at most one matched pair of squares, so greedy adjacent
#      pairing has an unambiguous intended outcome.
#    - At least one actual matched pair exists somewhere in every generated instance.


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _diag_aligned(p: tuple[int, int], q: tuple[int, int]) -> bool:
    return p[0] != q[0] and abs(p[0] - q[0]) == abs(p[1] - q[1])


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _is_square_component(cells: set[tuple[int, int]]) -> bool:
    if len(cells) != 4:
        return False
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1


def _collect_square_components(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, object]] = []
    cell_to_component: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, line_color) or (r, c) in seen:
                continue

            stack = [(r, c)]
            seen.add((r, c))
            cells: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                cells.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))

            idx = len(components)
            is_square = _is_square_component(cells)
            components.append({"cells": cells, "color": color, "is_square": is_square})
            if is_square:
                for cell in cells:
                    cell_to_component[cell] = idx

    return components, cell_to_component


def _greedy_adjacent_block_pairs(
    entries: list[dict[str, object]]
) -> list[tuple[dict[str, object], dict[str, object], int]]:
    remaining = sorted(entries, key=lambda item: int(item["start"]))
    pairs: list[tuple[dict[str, object], dict[str, object], int]] = []
    while len(remaining) >= 2:
        best_i = min(
            range(len(remaining) - 1),
            key=lambda i: int(remaining[i + 1]["start"]) - int(remaining[i]["end"]) - 1,
        )
        left = remaining[best_i]
        right = remaining[best_i + 1]
        gap = int(right["start"]) - int(left["end"]) - 1
        if gap > 0:
            pairs.append((left, right, gap))
        del remaining[best_i : best_i + 2]
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete blue diagonals, then let matched 2x2 magenta squares emit bounded perpendicular rays.

    Input structure:
    - The background is the most common color. One non-background color forms endpoint pairs on shared 45-degree diagonals.
    - Reactive magenta objects are 2x2 squares, not single pixels. Only squares intersected by a completed blue diagonal can participate.

    Rule:
    - Connect each detected blue endpoint pair along its diagonal.
    - On each completed diagonal, identify the activated 2x2 squares in order, greedily pair the closest adjacent squares, and let every touched cell of each matched square emit perpendicular diagonal rays whose length equals the empty diagonal gap between the two squares. Unpaired squares do nothing.

    Key insight:
    - The trigger logic is two-stage: first parse pixels into whole 2x2 blocks, then reason about pair spacing between blocks. The ray length comes from the block pair, not from any single pixel or from the block shape alone.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]
    components, cell_to_component = _collect_square_components(grid, bg, line_color)

    emissions: list[tuple[int, int, int, tuple[tuple[int, int], tuple[int, int]], int]] = []

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = ((dr, -dc), (-dr, dc))

        touched_by_component: dict[int, dict[str, object]] = {}
        for idx, (r, c) in enumerate(path):
            if int(out[r, c]) == bg:
                out[r, c] = line_color

            original = int(grid[r, c])
            if original in (bg, line_color):
                continue
            comp_idx = cell_to_component.get((r, c))
            if comp_idx is None:
                continue
            entry = touched_by_component.setdefault(
                comp_idx,
                {"indices": [], "cells": set(), "color": int(original)},
            )
            entry["indices"].append(idx)
            entry["cells"].add((r, c))

        ordered_blocks: list[dict[str, object]] = []
        for comp_idx, info in touched_by_component.items():
            indices = sorted(set(int(v) for v in info["indices"]))
            if not indices:
                continue
            ordered_blocks.append(
                {
                    "component": comp_idx,
                    "start": indices[0],
                    "end": indices[-1],
                    "cells": sorted(info["cells"]),
                    "color": int(info["color"]),
                    "component_cells": components[comp_idx]["cells"],
                }
            )

        ordered_blocks.sort(key=lambda item: int(item["start"]))
        for left, right, gap in _greedy_adjacent_block_pairs(ordered_blocks):
            if gap <= 0:
                continue
            for block in (left, right):
                for r0, c0 in block["cells"]:
                    emissions.append((r0, c0, int(block["color"]), perp_dirs, gap))

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for r0, c0, tcol, dirs, ray_len in emissions:
        for ddr, ddc in dirs:
            for step in range(1, ray_len + 1):
                r = r0 + ddr * step
                c = c0 + ddc * step
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) == bg:
                    out[r, c] = tcol

    return out


# --------------- generator ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (10, 25)
_W_RANGE = (10, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (9, 16)
_PAIR_GAP_RANGE = (2, 4)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_BLOCKS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_BLOCK_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 100
_GEN_RETRIES = 500


def _touches_existing_magenta(
    new_cells: set[tuple[int, int]], existing_magenta: set[tuple[int, int]]
) -> bool:
    for r, c in new_cells:
        if (r, c) in existing_magenta:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in existing_magenta:
                return True
    return False


def _diag_hits_existing_magenta(
    new_cells: set[tuple[int, int]], existing_magenta: set[tuple[int, int]]
) -> bool:
    return any(_diag_aligned(cell, old) for cell in new_cells for old in existing_magenta)


def _make_square_block(path: list[tuple[int, int]], start_idx: int, dc: int) -> dict[str, object]:
    r, c = path[start_idx]
    if dc == 1:
        cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        touched = [path[start_idx], path[start_idx + 1]]
    else:
        cells = {(r, c - 1), (r, c), (r + 1, c - 1), (r + 1, c)}
        touched = [path[start_idx], path[start_idx + 1]]
    return {"cells": cells, "touched": touched, "start": start_idx, "end": start_idx + 1}


def _compute_pair_ray_cells(
    blocks: list[dict[str, object]], gap: int, perp_dirs: tuple[tuple[int, int], tuple[int, int]], H: int, W: int
) -> set[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = set()
    for block in blocks:
        for tr, tc in block["touched"]:
            for ddr, ddc in perp_dirs:
                for step in range(1, gap + 1):
                    r = tr + ddr * step
                    c = tc + ddc * step
                    if not (0 <= r < H and 0 <= c < W):
                        return None
                    cells.add((r, c))
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        pair = seg.get("pair")
        if not pair:
            continue
        gap = int(pair["gap"])
        for block in pair["blocks"]:
            for tr, tc in block["touched"]:
                for ddr, ddc in seg["perp_dirs"]:
                    for step in range(1, gap + 1):
                        r = tr + ddr * step
                        c = tc + ddc * step
                        if 0 <= r < H and 0 <= c < W and expected[r, c] == bg:
                            expected[r, c] = _TRIGGER_COLOR
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    used_a: set[int],
    used_b: set[int],
    existing_line_pixels: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
            diag_id = r0 - c0
            if diag_id in used_a:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1
            diag_id = r0 + c0
            if diag_id in used_b:
                continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue

        ep0, ep1 = path[0], path[-1]
        if any(_diag_aligned(ep0, p) or _diag_aligned(ep1, p) for p in existing_line_pixels):
            continue
        return path, dc, diag_id
    return None


def _try_assign_segment_blocks(
    path: list[tuple[int, int]],
    dc: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    global_magenta: set[tuple[int, int]],
    require_pair: bool,
) -> dict[str, object] | None:
    perp_dirs = ((1, -dc), (-1, dc))
    path_set = set(path)
    path_fillable = set(path[1:-1])
    max_start = len(path) - 3

    choices = ["pair", "lone", "none"] if not require_pair else ["pair"]
    for _ in range(_BLOCK_PLACE_RETRIES):
        layout = random.choice(choices)
        if layout == "none":
            return {"blocks": [], "pair": None, "ray_cells": set()}

        if layout == "lone":
            if max_start < 1:
                continue
            start_idx = random.randint(1, max_start)
            block = _make_square_block(path, start_idx, dc)
            cells = set(block["cells"])
            if cells & occupied or cells & drawn:
                continue
            if _touches_existing_magenta(cells, global_magenta):
                continue
            if _diag_hits_existing_magenta(cells, global_magenta):
                continue
            return {"blocks": [block], "pair": None, "ray_cells": set()}

        gap = random.randint(*_PAIR_GAP_RANGE)
        max_left_start = len(path) - gap - 5
        if max_left_start < 1:
            continue
        left_start = random.randint(1, max_left_start)
        right_start = left_start + gap + 2
        left_block = _make_square_block(path, left_start, dc)
        right_block = _make_square_block(path, right_start, dc)
        blocks = [left_block, right_block]
        block_cells = set().union(*(block["cells"] for block in blocks))
        touched_cells = set().union(*(set(block["touched"]) for block in blocks))
        ray_cells = _compute_pair_ray_cells(blocks, gap, perp_dirs, H, W)
        if ray_cells is None:
            continue
        if block_cells & occupied or block_cells & drawn:
            continue
        if _touches_existing_magenta(block_cells, global_magenta):
            continue
        if _diag_hits_existing_magenta(block_cells, global_magenta):
            continue
        if ray_cells & occupied or ray_cells & drawn:
            continue
        if ray_cells & block_cells:
            continue
        if ray_cells & (path_fillable - touched_cells):
            continue
        if any(cell not in path_set for cell in touched_cells):
            continue
        return {"blocks": blocks, "pair": {"blocks": blocks, "gap": gap}, "ray_cells": ray_cells}

    return None


def _try_place_offdiag_block(
    H: int,
    W: int,
    path_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    magenta_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for _ in range(_DISTRACTOR_RETRIES):
        r = random.randint(0, H - 2)
        c = random.randint(0, W - 2)
        cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        if cells & path_cells or cells & occupied or cells & drawn:
            continue
        if _touches_existing_magenta(cells, magenta_cells):
            continue
        if _diag_hits_existing_magenta(cells, magenta_cells):
            continue
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    magenta_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    segments: list[dict[str, object]] = []

    used_a: set[int] = set()
    used_b: set[int] = set()
    all_line_pixels: list[tuple[int, int]] = []
    need_pair = True

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(
            slope_type,
            seg_len,
            H,
            W,
            occupied,
            drawn,
            used_a,
            used_b,
            all_line_pixels,
        )
        if result is None:
            continue
        path, dc, diag_id = result

        block_plan = _try_assign_segment_blocks(path, dc, H, W, occupied, drawn, magenta_cells, need_pair)
        if block_plan is None:
            continue

        blocks = list(block_plan["blocks"])
        pair = block_plan["pair"]
        ray_cells = set(block_plan["ray_cells"])
        touched_cells = set().union(*(set(block["touched"]) for block in blocks)) if blocks else set()
        blue_fill = set(path[1:-1]) - touched_cells
        if blue_fill & ray_cells:
            continue

        segments.append(
            {
                "endpoints": (path[0], path[-1]),
                "path": path,
                "perp_dirs": ((1, -dc), (-1, dc)),
                "blocks": blocks,
                "pair": pair,
            }
        )
        occupied.update([path[0], path[-1]])
        for block in blocks:
            occupied.update(block["cells"])
            magenta_cells.update(block["cells"])
        drawn.update(blue_fill)
        drawn.update(ray_cells)
        path_cells.update(path)
        all_line_pixels.extend([path[0], path[-1]])
        if slope_type == "backslash":
            used_a.add(diag_id)
        else:
            used_b.add(diag_id)
        if pair is not None:
            need_pair = False

    if not segments or need_pair:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        ep0, ep1 = seg["endpoints"]
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for block in seg["blocks"]:
            for cell in block["cells"]:
                inp[cell] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            cand = (r, c)
            if cand in occupied or cand in drawn:
                continue
            if any(_diag_aligned(cand, lp) for lp in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add(cand)
            all_line_positions.append(cand)
            break

    for _ in range(random.randint(*_OFF_DIAG_BLOCKS)):
        cells = _try_place_offdiag_block(H, W, path_cells, occupied, drawn, magenta_cells)
        if cells is None:
            continue
        for cell in cells:
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            magenta_cells.add(cell)

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
