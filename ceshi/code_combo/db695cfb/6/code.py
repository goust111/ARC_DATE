# Combo variant of db695cfb (index 6)
# combo_spec: D6/2+D11/2
# Name: 箭头源+标记配对

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete representation and behavior trace
#    - The original task used two isolated blue pixels as a naturally paired diagonal segment.
#      The solver could collect all blue pixels, find diagonal pairs, then use magenta cells on
#      those diagonals as ray emitters.
#    - In this combo, each active source endpoint becomes a 3-cell blue L-arrow. The degree-2
#      corner is the tip; the missing quadrant is the arrow's forward diagonal direction.
#    - Each arrow also has exactly one orthogonally adjacent marker cell in a non-blue,
#      non-magenta color. Marker colors are the pairing labels: two arrows with the same marker
#      color form one active pair. The segment is drawn between their tips, and both arrows must
#      face toward that segment.
#    - Example: suppose arrows A and B have red markers. A's tip is at (3,3) and points SE;
#      B's tip is at (8,8) and points NW. Old logic would incorrectly read many blue cell pairs
#      inside the arrow bodies and could not tell which blue cells are endpoints. New logic first
#      parses each 4-connected 3-cell L, extracts its tip/direction, groups arrows by marker
#      color, then fills the diagonal from (3,3) to (8,8). Any magenta cell on that path emits
#      magenta rays on the perpendicular diagonal as before.
#    - To make the pairing label visibly necessary, the generator may also add a uniquely marked
#      arrow distractor on the same diagonal extension as a real pair. Geometry-only pairing would
#      be tempted to connect to that farther distractor, but the true rule ignores singleton
#      marker colors and only uses colors that occur exactly twice.
#
# b) Complexity calibration
#    - Arrow orientation has only four possible values: NE, NW, SE, SW. That keeps the shape cue
#      small and repeatable.
#    - Marker colors come from a fixed small palette, and active colors appear exactly twice.
#      Generated samples use 2-4 active pairs so repeated labels are visually obvious.
#    - At most one uniquely marked distractor arrow is added. That is enough to make marker-based
#      pairing operational without flooding the board with exceptional cases.
#
# c) Implicit-assumption audit of the original solver
#    - The old `by_color[line_color]` pool is invalid because a single arrow contains three blue
#      cells, including incidental diagonal pairs that are not endpoints.
#    - The old "connect extremes on each blue diagonal" assumption is invalid because active
#      pairing now depends on adjacent marker colors, not on blue geometry alone.
#    - The old trigger rule "any non-blue color on the line is a trigger" is unsafe once marker
#      cells exist. The new solver only treats magenta (6) cells on active paths as triggers.
#    - The new solver introduces extra assumptions: every meaningful blue component must be a
#      3-cell L arrow, each such arrow must have exactly one adjacent marker, active marker colors
#      must appear on exactly two arrows, and those two arrows must face each other along a shared
#      diagonal. The generator explicitly constructs only inputs that satisfy these assumptions.
#
# d) Placement legality enforced by the generator
#    - Arrow bodies, marker cells, trigger cells, and output-only fills/rays are all placed on
#      distinct cells.
#    - No active blue fill or magenta ray overwrites any input pixel. This includes markers and
#      distractor arrows.
#    - Marker cells are attached to the tip side of an arrow, stay off all active diagonals and
#      future rays, and touch only their own arrow in 4-neighborhood.
#    - Blue components are kept 4-disconnected from one another so the solver can recover each
#      arrow unambiguously.
#
# e) Visual observability
#    - Humans can directly see the L-arrow cue and infer that the diagonal leaves the tip through
#      the missing quadrant.
#    - Repeated marker colors across distant arrows make the pairing label visible in a few
#      examples; singleton colors act as visibly unmatched distractors rather than hidden state.
#    - Because every sample has multiple active pairs, the rule "pair by repeated marker color,
#      then connect arrow tips" is more plausible than any accidental one-off geometry heuristic.
#
# f) Visual cleanliness
#    - Active diagonals do not overlap each other, and rays do not pass through any pre-placed
#      input object.
#    - Markers sit beside arrow tips instead of on the diagonal itself, so arrows, labels, path
#      fills, and magenta triggers remain easy to distinguish.
#    - Distractor arrows, when present, are placed on diagonal extensions outside the active
#      segment, so they create a clean ambiguity about pairing without cluttering the active path.
#
# g) Change inferability
#    - Marker colors do not transform into output colors; they only label which arrows belong
#      together. Blue still draws the main diagonal and magenta still draws the perpendicular
#      rays. This keeps the added symbolic load small and structurally supported.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_MARKER_COLORS = [2, 3, 4, 5, 7, 8, 9]

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_ACTIVE_PAIR_COUNT = (2, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)
_DECOY_ARROW_COUNT = (1, 1)

# === CFG: distractors ===
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 140
_MARKER_PLACE_RETRIES = 12
_DECOY_PLACE_RETRIES = 80
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


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _in_bounds(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _tip_marker_candidates(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    return [(r + dr, c), (r, c + dc)]


def _compute_ray_cells(
    trigger_positions: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc in trigger_positions:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    arrows: list[dict] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != _LINE_COLOR or (r, c) in visited:
                continue

            stack = [(r, c)]
            comp = set()
            visited.add((r, c))
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and int(grid[nr, nc]) == _LINE_COLOR:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

            if len(comp) != 3:
                continue

            deg = {cell: sum((nbr in comp) for nbr in _neighbors4(*cell)) for cell in comp}
            if sorted(deg.values()) != [1, 1, 2]:
                continue

            tip = next(cell for cell, d in deg.items() if d == 2)
            rels = [(rr - tip[0], cc - tip[1]) for rr, cc in comp if (rr, cc) != tip]
            vertical = [d for d in rels if d[1] == 0 and abs(d[0]) == 1]
            horizontal = [d for d in rels if d[0] == 0 and abs(d[1]) == 1]
            if len(vertical) != 1 or len(horizontal) != 1:
                continue

            direction = (-vertical[0][0], -horizontal[0][1])
            if direction[0] not in (-1, 1) or direction[1] not in (-1, 1):
                continue

            marker_cells: dict[tuple[int, int], int] = {}
            for ar, ac in comp:
                for nr, nc in _neighbors4(ar, ac):
                    if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in comp:
                        continue
                    value = int(grid[nr, nc])
                    if value in (bg, _LINE_COLOR, _TRIGGER_COLOR):
                        continue
                    marker_cells[(nr, nc)] = value

            assert len(marker_cells) == 1, "each arrow must have exactly one adjacent marker"
            (marker_cell, marker_color), = marker_cells.items()
            arrows.append(
                {
                    "tip": tip,
                    "direction": direction,
                    "cells": comp,
                    "marker_cell": marker_cell,
                    "marker_color": marker_color,
                }
            )

    arrows.sort(key=lambda a: (a["tip"], a["marker_color"]))
    return arrows


def _ordered_pair(left: dict, right: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    p = left["tip"]
    q = right["tip"]
    dr = q[0] - p[0]
    dc = q[1] - p[1]
    if dr == 0 or abs(dr) != abs(dc):
        return None
    step = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
    if left["direction"] == step and right["direction"] == (-step[0], -step[1]):
        return p, q, step
    return None


def _active_pairs_from_arrows(arrows: list[dict]) -> list[dict]:
    marker_groups: dict[int, list[dict]] = {}
    for arrow in arrows:
        marker_groups.setdefault(arrow["marker_color"], []).append(arrow)

    pairs: list[dict] = []
    for marker_color, group in marker_groups.items():
        if len(group) != 2:
            continue
        a, b = group
        ordered = _ordered_pair(a, b)
        if ordered is None:
            ordered = _ordered_pair(b, a)
        assert ordered is not None, "same-color arrow pair must be diagonal and face inward"
        p, q, step = ordered
        pairs.append(
            {
                "marker_color": marker_color,
                "start": p,
                "end": q,
                "path": _diag_path(p, q),
                "perp_dirs": [(step[0], -step[1]), (-step[0], step[1])],
            }
        )
    pairs.sort(key=lambda seg: (seg["start"], seg["end"], seg["marker_color"]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read marked blue arrow sprites as diagonal endpoints, connect same-marked arrow pairs, then emit rays from magenta cells on those diagonals.

    Input structure:
    - bg = most common color. Blue source objects are 3-cell L-shaped arrows whose corner cell is the tip and whose missing quadrant gives the forward diagonal.
    - Each arrow has one orthogonally adjacent marker cell in a non-blue, non-magenta color. Marker colors that appear twice define the active arrow pairs.
    - Magenta trigger cells may lie on the interior of active diagonals; uniquely marked arrows or off-diagonal magenta cells can appear as distractors.

    Rule:
    - Parse every valid blue arrow, group arrows by marker color, and for each marker color that labels exactly two arrows, draw the blue diagonal segment between their tips.
    - While tracing that segment, only magenta cells act as triggers: from each such cell, launch magenta rays in both directions along the perpendicular diagonal, filling background cells until the boundary.

    Key insight:
    - Neither raw blue geometry nor marker colors alone are sufficient: the L-shape identifies the true tip and diagonal direction, while the repeated marker color decides which two arrows should actually be connected.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    arrows = _find_arrows(grid, bg)
    assert arrows, "grid must contain at least one valid blue arrow"

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for pair in _active_pairs_from_arrows(arrows):
        for r, c in pair["path"]:
            value = int(out[r, c])
            if value == bg:
                out[r, c] = _LINE_COLOR
            elif value == _TRIGGER_COLOR:
                triggers.setdefault((r, c), set()).update(pair["perp_dirs"])

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return out


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    expected = inp.copy()
    h, w = inp.shape
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _legacy_output(grid: np.ndarray) -> np.ndarray:
    """Original db695cfb behavior, used only to reject degenerate samples."""
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
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    if best is None:
        return out

    _, line_color = best
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
            value = int(out[r, c])
            if value == bg:
                out[r, c] = line_color
            elif value != line_color:
                triggers.setdefault((r, c, value), set()).update(perp_dirs)

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


def _markerless_output(grid: np.ndarray) -> np.ndarray:
    """
    Ablation: parse arrows but ignore marker colors, pairing each arrow with its farthest
    mutually facing compatible partner. Used only to reject samples where labels do no work.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)
    arrows = _find_arrows(grid, bg)

    pair_keys: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    arrows_by_tip = {arrow["tip"]: arrow for arrow in arrows}

    for arrow in arrows:
        best_dist = -1
        best_tip = None
        for other in arrows:
            if other is arrow:
                continue
            ordered = _ordered_pair(arrow, other)
            if ordered is None:
                continue
            dist = abs(other["tip"][0] - arrow["tip"][0])
            if dist > best_dist:
                best_dist = dist
                best_tip = other["tip"]
        if best_tip is not None:
            pair = tuple(sorted((arrow["tip"], best_tip)))
            pair_keys.add(pair)

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for p, q in sorted(pair_keys):
        left = arrows_by_tip[p]
        right = arrows_by_tip[q]
        ordered = _ordered_pair(left, right)
        if ordered is None:
            ordered = _ordered_pair(right, left)
        if ordered is None:
            continue
        start, end, step = ordered
        perp_dirs = [(step[0], -step[1]), (-step[0], step[1])]
        for r, c in _diag_path(start, end):
            value = int(out[r, c])
            if value == bg:
                out[r, c] = _LINE_COLOR
            elif value == _TRIGGER_COLOR:
                triggers.setdefault((r, c), set()).update(perp_dirs)

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return out


def _try_place_active_segment(
    slope_type: str,
    seg_len: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    blue_occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> dict | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            step = (1, 1)
        else:
            if h - 1 - seg_len < 0 or seg_len > w - 1:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            step = (1, -1)

        path = [(r0 + k * step[0], c0 + k * step[1]) for k in range(seg_len + 1)]
        start = path[0]
        end = path[-1]
        start_dir = step
        end_dir = (-step[0], -step[1])
        start_cells = _arrow_cells(start, start_dir)
        end_cells = _arrow_cells(end, end_dir)
        all_arrow_cells = start_cells | end_cells
        path_interior = set(path[1:-1])

        if any(not _in_bounds(h, w, cell) for cell in all_arrow_cells):
            continue
        if any(cell in occupied or cell in drawn for cell in all_arrow_cells | path_interior):
            continue
        if any(cell in blue_occupied for cell in all_arrow_cells):
            continue
        if any(nbr in blue_occupied for cell in all_arrow_cells for nbr in _neighbors4(*cell)):
            continue

        return {
            "start": start,
            "end": end,
            "step": step,
            "path": path,
            "arrows": [
                {"tip": start, "direction": start_dir, "cells": start_cells},
                {"tip": end, "direction": end_dir, "cells": end_cells},
            ],
        }
    return None


def _find_marker_cell(
    tip: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    blue_occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = _tip_marker_candidates(tip, direction)
    random.shuffle(candidates)
    for mr, mc in candidates[:]:
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        cell = (mr, mc)
        if cell in occupied or cell in drawn:
            continue
        if any((nr, nc) in blue_occupied and (nr, nc) != tip for nr, nc in _neighbors4(mr, mc)):
            continue
        return cell
    return None


def _try_place_decoy_arrow(
    segment: dict,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    blue_occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> dict | None:
    start = segment["start"]
    end = segment["end"]
    step = segment["step"]

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for k in range(1, max(h, w)):
        tip = (end[0] + k * step[0], end[1] + k * step[1])
        if not _in_bounds(h, w, tip):
            break
        candidates.append((tip, (-step[0], -step[1])))
    for k in range(1, max(h, w)):
        tip = (start[0] - k * step[0], start[1] - k * step[1])
        if not _in_bounds(h, w, tip):
            break
        candidates.append((tip, step))

    random.shuffle(candidates)
    for tip, direction in candidates[:_DECOY_PLACE_RETRIES]:
        cells = _arrow_cells(tip, direction)
        if any(not _in_bounds(h, w, cell) for cell in cells):
            continue
        if any(cell in occupied or cell in drawn for cell in cells):
            continue
        if any(cell in blue_occupied for cell in cells):
            continue
        if any(nbr in blue_occupied for cell in cells for nbr in _neighbors4(*cell)):
            continue
        return {"tip": tip, "direction": direction, "cells": cells}
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    available_marker_colors = [c for c in _MARKER_COLORS if c != bg]
    max_pairs = min(_ACTIVE_PAIR_COUNT[1], len(available_marker_colors) - _DECOY_ARROW_COUNT[1])
    if max_pairs < _ACTIVE_PAIR_COUNT[0]:
        return None
    pair_count = random.randint(_ACTIVE_PAIR_COUNT[0], max_pairs)

    occupied: set[tuple[int, int]] = set()
    blue_occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[dict] = []

    for marker_color in random.sample(available_marker_colors, pair_count):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        placed = _try_place_active_segment(slope_type, seg_len, h, w, occupied, blue_occupied, drawn)
        if placed is None:
            continue

        path = placed["path"]
        path_interior = path[1:-1]
        trigger_count = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, trigger_count) if trigger_count > 0 else []
        perp_dirs = [(placed["step"][0], -placed["step"][1]), (-placed["step"][0], placed["step"][1])]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, h, w)

        future_input = occupied | set(trigger_positions)
        future_drawn = drawn | (set(path[1:-1]) - set(trigger_positions))
        if any(cell in future_input or cell in future_drawn for cell in ray_cells):
            continue

        marker_cells = []
        temp_occupied = occupied | set(trigger_positions)
        temp_blue = blue_occupied | placed["arrows"][0]["cells"] | placed["arrows"][1]["cells"]
        for arrow in placed["arrows"]:
            marker_cell = _find_marker_cell(arrow["tip"], arrow["direction"], h, w, temp_occupied, drawn | ray_cells | future_drawn, temp_blue)
            if marker_cell is None:
                break
            marker_cells.append(marker_cell)
            temp_occupied.add(marker_cell)
        if len(marker_cells) != 2:
            continue

        placed["marker_color"] = marker_color
        placed["marker_cells"] = marker_cells
        placed["trigger_positions"] = trigger_positions
        placed["perp_dirs"] = perp_dirs
        segments.append(placed)

        for arrow in placed["arrows"]:
            occupied.update(arrow["cells"])
            blue_occupied.update(arrow["cells"])
        occupied.update(marker_cells)
        occupied.update(trigger_positions)
        drawn.update(set(path[1:-1]) - set(trigger_positions))
        drawn.update(ray_cells)

    if len(segments) < _ACTIVE_PAIR_COUNT[0]:
        return None

    remaining_colors = [c for c in available_marker_colors if c not in {seg["marker_color"] for seg in segments}]
    decoys: list[dict] = []
    decoy_target = random.randint(*_DECOY_ARROW_COUNT)
    if len(remaining_colors) < decoy_target:
        return None

    shuffled_segments = segments[:]
    random.shuffle(shuffled_segments)
    for marker_color, segment in zip(remaining_colors, shuffled_segments):
        placed = _try_place_decoy_arrow(segment, h, w, occupied, blue_occupied, drawn)
        if placed is None:
            continue
        marker_cell = _find_marker_cell(placed["tip"], placed["direction"], h, w, occupied, drawn, blue_occupied | placed["cells"])
        if marker_cell is None:
            continue
        placed["marker_color"] = marker_color
        placed["marker_cell"] = marker_cell
        decoys.append(placed)
        occupied.update(placed["cells"])
        blue_occupied.update(placed["cells"])
        occupied.add(marker_cell)
        if len(decoys) == decoy_target:
            break

    if len(decoys) != decoy_target:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for seg in segments:
        for arrow, marker_cell in zip(seg["arrows"], seg["marker_cells"]):
            for r, c in arrow["cells"]:
                inp[r, c] = _LINE_COLOR
            inp[marker_cell] = seg["marker_color"]
        for tr, tc in seg["trigger_positions"]:
            inp[tr, tc] = _TRIGGER_COLOR

    for decoy in decoys:
        for r, c in decoy["cells"]:
            inp[r, c] = _LINE_COLOR
        inp[decoy["marker_cell"]] = decoy["marker_color"]

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            cell = (r, c)
            if cell in occupied or cell in drawn:
                continue
            if any(cell in seg["path"] for seg in segments):
                continue
            if any(abs(r - arrow["tip"][0]) + abs(c - arrow["tip"][1]) == 1 for seg in segments for arrow in seg["arrows"]):
                continue
            if any(abs(r - decoy["tip"][0]) + abs(c - decoy["tip"][1]) == 1 for decoy in decoys):
                continue
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if not np.array_equal(result, expected):
        return None

    if np.array_equal(_legacy_output(inp), expected):
        return None

    if np.array_equal(_markerless_output(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
