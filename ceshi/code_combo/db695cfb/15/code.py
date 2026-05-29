# Combo variant of db695cfb (index 15)
# combo_spec: D6/2+D11/2+D12/0
# Name: 箭头源+标记配对+X形双线

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete representation and behavior trace
#    - Original task: two isolated blue pixels define one diagonal segment; magenta cells on
#      that segment emit rays on the perpendicular diagonal; only the direct segment is drawn.
#    - Combo task: each endpoint is replaced by a 3-cell blue L-arrow. The degree-2 corner is
#      the tip, and the missing diagonal quadrant is the arrow's forward direction.
#    - Each arrow tip has exactly one orthogonally adjacent marker pixel in a non-blue,
#      non-magenta color. Two arrows with the same marker color are one pair.
#    - For one pair, the two tips lie on a common 45-degree diagonal and the arrows face each
#      other. Their tips define the direct diagonal of a square, and the output fills both
#      diagonals of that square, forming an X.
#    - Magenta triggers may lie on the direct branch or on the crossing branch. A trigger on one
#      branch emits magenta rays along the other diagonal orientation to the grid edge.
#    - Small example trace:
#      old logic: two blue endpoints at (2,2),(6,6) -> draw only "\"; magenta at (4,4) emits "/".
#      combo logic: blue arrows with tips at (2,2),(6,6), same red marker color, facing toward
#      each other -> draw "\" plus the square's "/" branch. A magenta at (3,5) on the hidden "/"
#      branch now emits "\" rays, which the original task would never create.
#
# b) Complexity calibration
#    - Arrow geometry has only four orientations, so direction remains visually stable.
#    - Marker colors come from a small palette and every active color appears exactly twice, so
#      pairing is repeated and observable rather than one-off.
#    - We keep 2-4 pairs and at most one direct + one cross trigger per pair. This is enough to
#      make marker pairing and cross-branch triggering visible without turning the grid noisy.
#
# c) Implicit-assumption audit
#    - Original blue-pixel grouping by diagonal is invalid: each arrow contains an incidental
#      blue diagonal pair inside the sprite, so the solver must parse 4-connected blue components.
#    - Original "all blue pixels belong to one pool" is invalid: pairing moves to adjacent marker
#      colors attached to arrow tips.
#    - Original trigger collection from "any non-blue on the path" is invalid because marker
#      colors exist; the solver must only treat magenta as a trigger.
#    - Original single-branch drawing is invalid because each paired tip-set now defines two
#      branches with different ray directions.
#    - New solver assumptions introduced here:
#      1. every active blue object is a 3-cell L component;
#      2. every arrow tip has exactly one adjacent non-bg, non-blue, non-magenta marker;
#      3. every marker color labels exactly two arrows;
#      4. paired arrows lie on one diagonal and face each other;
#      5. markers are kept off the X paths, and rays never need to overwrite input pixels.
#      The generator explicitly enforces all five.
#
# d) Placement legality
#    - Both arrow bodies, both X branches, markers, triggers, and emitted rays must stay in-bounds
#      and pairwise disjoint except where a trigger intentionally sits on its own X branch.
#    - Arrow legs extend outward from the square, so the generator checks both endpoint sprites
#      before accepting a pair.
#    - Marker cells are chosen only from the two free orthogonal neighbors in front of each tip;
#      they may touch exactly one blue cell: their own tip.
#    - Rays are rejected if they hit any reserved input cell or any blue branch cell that is not
#      itself a trigger.
#
# e) Visual observability
#    - A human can see three stable cues across examples:
#      1. blue L-shapes identify arrow tips and directions;
#      2. repeated marker colors identify which distant arrows pair up;
#      3. outputs form X shapes, and some magenta triggers activate only because they sit on the
#         previously hidden crossing branch.
#    - At least one cross-branch trigger is forced in every generated sample so the X logic is not
#      visually reducible to "old task plus decorative markers".
#
# f) Visual cleanliness
#    - We keep branches and rays disjoint across objects, avoid center-cell dual triggers, and do
#      not place blue distractor components. This prevents merged blue components and keeps the
#      marker-tip relation readable.
#
# g) Color-role inferability
#    - Blue still means source/drawn scaffold and magenta still means trigger/ray color.
#    - Marker colors are structural labels only; they never transform in the output.
#    - The palette of marker colors is small and repeated, so the solver can infer "same marker
#      color means same pair" from 2-3 examples without relying on numeric color ids.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_MARKER_COLORS = [2, 3, 4, 5, 7, 8, 9]
_DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_PAIR_COUNT = (2, 4)
_SEG_LEN_RANGE = (3, 16)
_DIRECT_TRIGGERS_PER_PAIR = (0, 1)
_CROSS_TRIGGERS_PER_PAIR = (0, 1)

# === CFG: distractor counts ===
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_PAIR_PLACE_RETRIES = 180
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _cross_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "tips must form a 45-degree diagonal"

    r_lo, r_hi = sorted((r0, r1))
    c_lo, c_hi = sorted((c0, c1))
    if dr * dc > 0:
        return [(r_lo + k, c_hi - k) for k in range(abs(dr) + 1)]
    return [(r_lo + k, c_lo + k) for k in range(abs(dr) + 1)]


def _segment_dirs(p: tuple[int, int], q: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dr = 1 if q[0] > p[0] else -1
    dc = 1 if q[1] > p[1] else -1
    direct_dirs = [(dr, dc), (-dr, -dc)]
    cross_dirs = [(dr, -dc), (-dr, dc)]
    return direct_dirs, cross_dirs


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _forward_marker_candidates(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    return [(r + dr, c), (r, c + dc)]


def _parse_arrow_component(comp: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None

    deg = {cell: sum((nbr in comp) for nbr in _neighbors4(*cell)) for cell in comp}
    if sorted(deg.values()) != [1, 1, 2]:
        return None

    tip = next(cell for cell, d in deg.items() if d == 2)
    rels = [(rr - tip[0], cc - tip[1]) for rr, cc in comp if (rr, cc) != tip]
    vertical = [d for d in rels if d[1] == 0 and abs(d[0]) == 1]
    horizontal = [d for d in rels if d[0] == 0 and abs(d[1]) == 1]
    if len(vertical) != 1 or len(horizontal) != 1:
        return None

    dr = -vertical[0][0]
    dc = -horizontal[0][1]
    if dr not in (-1, 1) or dc not in (-1, 1):
        return None
    return tip, (dr, dc)


def _find_arrow_sources(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    arrows: list[dict] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != _LINE_COLOR or (r, c) in visited:
                continue

            stack = [(r, c)]
            comp: set[tuple[int, int]] = set()
            visited.add((r, c))
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and int(grid[nr, nc]) == _LINE_COLOR:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

            parsed = _parse_arrow_component(comp)
            if parsed is None:
                continue
            tip, direction = parsed
            arrows.append({"tip": tip, "direction": direction, "cells": comp})

    arrows.sort(key=lambda item: item["tip"])
    return arrows


def _exclusive_interior(path: list[tuple[int, int]], other_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    other = set(other_path)
    return [cell for cell in path[1:-1] if cell not in other]


def _compute_ray_cells(trigger_dir_map: dict[tuple[int, int], set[tuple[int, int]]], h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (tr, tc), dirs in trigger_dir_map.items():
        for ddr, ddc in dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _ordered_pair(ar0: dict, ar1: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    p = ar0["tip"]
    q = ar1["tip"]
    dr = q[0] - p[0]
    dc = q[1] - p[1]
    assert dr != 0 and abs(dr) == abs(dc), "paired arrow tips must share a diagonal"

    step = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
    if ar0["direction"] == step and ar1["direction"] == (-step[0], -step[1]):
        return p, q
    if ar1["direction"] == step and ar0["direction"] == (-step[0], -step[1]):
        return q, p
    raise AssertionError("paired arrows must face each other along their tip diagonal")


def _build_expected(inp: np.ndarray, bg: int, pairs: list[dict]) -> np.ndarray:
    expected = inp.copy()
    h, w = inp.shape
    for pair in pairs:
        for r, c in pair["path_cells"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for (tr, tc), dirs in pair["trigger_dir_map"].items():
            for ddr, ddc in dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _no_cross_output(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    arrows = _find_arrow_sources(grid)
    marker_groups: dict[int, list[dict]] = {}
    for arrow in arrows:
        tip = arrow["tip"]
        markers = []
        for nr, nc in _neighbors4(*tip):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(grid[nr, nc])
                if v not in (bg, _LINE_COLOR, _TRIGGER_COLOR):
                    markers.append(v)
        if len(markers) != 1:
            return out
        marker_groups.setdefault(markers[0], []).append(arrow)

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}

    for group in marker_groups.values():
        if len(group) != 2:
            return out
        p, q = _ordered_pair(group[0], group[1])
        _, cross_dirs = _segment_dirs(p, q)
        for r, c in _diag_path(p, q):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = _LINE_COLOR
            elif v == _TRIGGER_COLOR:
                triggers.setdefault((r, c), set()).update(cross_dirs)

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc
    return out


def _original_output(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dr = pts[j][0] - pts[i][0]
                dc = pts[j][1] - pts[i][1]
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    if best is None:
        return out

    line_color = best[1]
    pts = by_color[line_color]
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])
    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read marked blue arrow pairs, expand each pair into an X, and emit magenta rays from triggers touched by either branch.

    Input structure:
    - The background is the most common color. Active blue objects are 3-cell L-shaped arrows; the degree-2 corner is the tip and the missing diagonal quadrant gives the arrow's facing direction.
    - Each arrow tip has exactly one orthogonally adjacent marker pixel in a non-blue, non-magenta color. Two arrows with the same marker color belong together.
    - Magenta trigger pixels may sit on the tip-to-tip diagonal or on the crossing diagonal of that paired square. Extra magenta distractors may appear elsewhere.

    Rule:
    - For every marker-matched arrow pair, use the two tips as the endpoints of a direct diagonal and verify that the arrows face each other along it.
    - Fill both diagonals of the square spanned by those tips, producing a blue X.
    - Whenever a magenta trigger lies on one branch of that X, keep the trigger and launch magenta rays along the other diagonal orientation until the grid boundary, painting only background cells.

    Key insight:
    - The solution must combine three cues at once: arrow geometry tells you where the real endpoints are and which way they face, repeated marker colors tell you which two arrows form a pair, and each pair generates an entire X so triggers can activate from either branch.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    arrows = _find_arrow_sources(grid)
    assert arrows, "grid must contain at least one valid blue arrow source"

    marker_groups: dict[int, list[dict]] = {}
    for arrow in arrows:
        tip = arrow["tip"]
        markers = []
        for nr, nc in _neighbors4(*tip):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            v = int(grid[nr, nc])
            if v in (bg, _LINE_COLOR, _TRIGGER_COLOR):
                continue
            markers.append((nr, nc, v))
        assert len(markers) == 1, "each arrow tip must have exactly one adjacent marker"
        marker_groups.setdefault(markers[0][2], []).append(arrow)

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def draw_path(path: list[tuple[int, int]], ray_dirs: list[tuple[int, int]]) -> None:
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = _LINE_COLOR
            elif v == _TRIGGER_COLOR:
                triggers.setdefault((r, c), set()).update(ray_dirs)

    for marker_color in sorted(marker_groups):
        group = marker_groups[marker_color]
        assert len(group) == 2, "each marker color must label exactly two arrows"
        p, q = _ordered_pair(group[0], group[1])
        direct_path = _diag_path(p, q)
        cross_path = _cross_path(p, q)
        direct_dirs, cross_dirs = _segment_dirs(p, q)
        draw_path(direct_path, cross_dirs)
        draw_path(cross_path, direct_dirs)

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return out


def _find_marker_cell(
    tip: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = _forward_marker_candidates(tip, direction)
    random.shuffle(candidates)
    for mr, mc in candidates:
        cell = (mr, mc)
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        if cell in forbidden:
            continue
        adj_blue = {nbr for nbr in _neighbors4(mr, mc) if nbr in blue_cells}
        if adj_blue != {tip}:
            continue
        return cell
    return None


def _choose_triggers(
    direct_path: list[tuple[int, int]],
    cross_path: list[tuple[int, int]],
    need_cross: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    direct_candidates = _exclusive_interior(direct_path, cross_path)
    cross_candidates = _exclusive_interior(cross_path, direct_path)

    direct_max = min(_DIRECT_TRIGGERS_PER_PAIR[1], len(direct_candidates))
    cross_max = min(_CROSS_TRIGGERS_PER_PAIR[1], len(cross_candidates))

    if need_cross and cross_max == 0:
        return None

    direct_count = random.randint(_DIRECT_TRIGGERS_PER_PAIR[0], direct_max) if direct_max >= _DIRECT_TRIGGERS_PER_PAIR[0] else 0
    cross_min = 1 if need_cross else _CROSS_TRIGGERS_PER_PAIR[0]
    if cross_max < cross_min:
        return None
    cross_count = random.randint(cross_min, cross_max) if cross_max >= cross_min else 0

    direct_triggers = random.sample(direct_candidates, direct_count) if direct_count else []
    cross_triggers = random.sample(cross_candidates, cross_count) if cross_count else []
    return direct_triggers, cross_triggers


def _try_place_pair(
    marker_color: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    blue_occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    need_cross: bool,
) -> dict | None:
    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    for _ in range(_PAIR_PLACE_RETRIES):
        direction = random.choice(_DIAG_DIRS)
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        dr, dc = direction

        p = (random.randint(0, h - 1), random.randint(0, w - 1))
        q = (p[0] + seg_len * dr, p[1] + seg_len * dc)
        if not (0 <= q[0] < h and 0 <= q[1] < w):
            continue

        p_cells = _arrow_cells(p, direction)
        q_cells = _arrow_cells(q, (-dr, -dc))
        pair_arrow_cells = p_cells | q_cells
        if any(not (0 <= r < h and 0 <= c < w) for r, c in pair_arrow_cells):
            continue
        if any(cell in occupied or cell in drawn for cell in pair_arrow_cells):
            continue

        # Keep blue components separate and avoid incidental 4-neighbor merges.
        blocked = False
        for cell in pair_arrow_cells:
            if cell in blue_occupied:
                blocked = True
                break
            if any(nbr in blue_occupied for nbr in _neighbors4(*cell)):
                blocked = True
                break
        if blocked:
            continue

        direct_path = _diag_path(p, q)
        cross_path = _cross_path(p, q)
        path_cells = set(direct_path) | set(cross_path)
        if any(cell in occupied or cell in drawn for cell in path_cells - {p, q}):
            continue

        trigger_choice = _choose_triggers(direct_path, cross_path, need_cross)
        if trigger_choice is None:
            continue
        direct_triggers, cross_triggers = trigger_choice

        direct_dirs, cross_dirs = _segment_dirs(p, q)
        trigger_dir_map: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for pos in direct_triggers:
            trigger_dir_map.setdefault(pos, set()).update(cross_dirs)
        for pos in cross_triggers:
            trigger_dir_map.setdefault(pos, set()).update(direct_dirs)

        trigger_cells = set(trigger_dir_map)
        ray_cells = _compute_ray_cells(trigger_dir_map, h, w)

        if any(cell in occupied or cell in drawn or cell in pair_arrow_cells for cell in ray_cells):
            continue
        if any(cell in (path_cells - trigger_cells) for cell in ray_cells):
            continue

        pair_blue_cells = blue_occupied | pair_arrow_cells
        forbidden = occupied | drawn | path_cells | ray_cells | pair_arrow_cells
        p_marker = _find_marker_cell(p, direction, h, w, forbidden, pair_blue_cells)
        if p_marker is None:
            continue
        forbidden = forbidden | {p_marker}
        q_marker = _find_marker_cell(q, (-dr, -dc), h, w, forbidden, pair_blue_cells)
        if q_marker is None:
            continue

        return {
            "marker_color": marker_color,
            "tips": (p, q),
            "directions": (direction, (-dr, -dc)),
            "arrow_cells": (p_cells, q_cells),
            "marker_cells": (p_marker, q_marker),
            "path_cells": path_cells,
            "trigger_dir_map": trigger_dir_map,
            "has_cross_trigger": bool(cross_triggers),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    available_markers = [c for c in _MARKER_COLORS if c != bg]
    max_pairs = min(_PAIR_COUNT[1], len(available_markers))
    if max_pairs < _PAIR_COUNT[0]:
        return None

    n_pairs = random.randint(_PAIR_COUNT[0], max_pairs)
    marker_colors = random.sample(available_markers, n_pairs)

    occupied: set[tuple[int, int]] = set()
    blue_occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    pairs: list[dict] = []
    has_cross_trigger = False

    for marker_color in marker_colors:
        pair = _try_place_pair(marker_color, h, w, occupied, blue_occupied, drawn, need_cross=not has_cross_trigger)
        if pair is None:
            return None

        p_cells, q_cells = pair["arrow_cells"]
        occupied.update(p_cells)
        occupied.update(q_cells)
        blue_occupied.update(p_cells)
        blue_occupied.update(q_cells)

        for cell in pair["marker_cells"]:
            occupied.add(cell)
        for cell in pair["trigger_dir_map"]:
            occupied.add(cell)

        p, q = pair["tips"]
        drawn.update(pair["path_cells"] - {p, q} - set(pair["trigger_dir_map"]))
        drawn.update(_compute_ray_cells(pair["trigger_dir_map"], h, w))
        pairs.append(pair)
        has_cross_trigger = has_cross_trigger or pair["has_cross_trigger"]

    if not pairs or not has_cross_trigger:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for pair in pairs:
        p_cells, q_cells = pair["arrow_cells"]
        for cell in p_cells | q_cells:
            inp[cell] = _LINE_COLOR
        for cell in pair["marker_cells"]:
            inp[cell] = pair["marker_color"]
        for cell in pair["trigger_dir_map"]:
            inp[cell] = _TRIGGER_COLOR

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            cell = (r, c)
            if cell in occupied or cell in drawn:
                continue
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, pairs)
    if not np.array_equal(result, expected):
        return None

    # Enforce that the X-branch logic is actually active.
    if np.array_equal(_no_cross_output(inp), expected):
        return None
    if np.array_equal(_original_output(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
