import random
from collections import Counter

import numpy as np


BG = 0
MARKER_COLOR = 7
BAR_COLORS = (1, 2, 3, 4, 8)
DIRS = {
    "left": (0, -1),
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
}
MISSING_TO_DIR = {
    (0, 0): "left",
    (0, 1): "up",
    (1, 1): "right",
    (1, 0): "down",
}
DIR_TO_MISSING = {v: k for k, v in MISSING_TO_DIR.items()}


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_cells(grid, start, seen):
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    stack = [(sr, sc)]
    seen.add((sr, sc))
    cells = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    cells.sort()
    return color, cells


def _extract_components(grid, colors):
    h, w = grid.shape
    color_set = set(colors)
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if int(grid[r, c]) not in color_set:
                continue
            color, cells = _component_cells(grid, (r, c), seen)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "area": len(cells),
                }
            )
    return comps


def _is_bar(comp):
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return (h == 1 or w == 1) and comp["area"] == max(h, w) and comp["area"] >= 3


def _is_marker(comp):
    if comp["color"] != MARKER_COLOR or comp["area"] != 3:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return r1 - r0 == 1 and c1 - c0 == 1


def _marker_missing_corner(marker):
    r0, c0, _r1, _c1 = marker["bbox"]
    present = {(r - r0, c - c0) for r, c in marker["cells"]}
    for rr in (0, 1):
        for cc in (0, 1):
            if (rr, cc) not in present:
                return rr, cc
    raise AssertionError("invalid L marker")


def _min_manhattan(cells_a, cells_b):
    best = None
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            dist = abs(ra - rb) + abs(ca - cb)
            if best is None or dist < best:
                best = dist
    return best


def _parse_scene(grid):
    bars = [comp for comp in _extract_components(grid, BAR_COLORS) if _is_bar(comp)]
    markers = [comp for comp in _extract_components(grid, [MARKER_COLOR]) if _is_marker(comp)]
    assert bars and markers and len(bars) == len(markers)

    used_markers = set()
    for bar in bars:
        dists = []
        for idx, marker in enumerate(markers):
            dists.append((_min_manhattan(bar["cells"], marker["cells"]), idx))
        dists.sort()
        assert len(dists) >= 2 or dists[0][1] not in used_markers
        assert len(dists) == 1 or dists[0][0] < dists[1][0], "nearest marker must be unique"
        marker_idx = dists[0][1]
        assert marker_idx not in used_markers, "markers must pair bijectively"
        used_markers.add(marker_idx)
        marker = markers[marker_idx]
        bar["marker_idx"] = marker_idx
        bar["direction"] = MISSING_TO_DIR[_marker_missing_corner(marker)]
        bar["length"] = bar["area"]

    assert len(used_markers) == len(markers)
    return bars, markers


def _simulate_bars(shape, bars, order=None):
    h, w = shape
    if order is None:
        order = sorted(range(len(bars)), key=lambda i: (bars[i]["length"], bars[i]["bbox"], bars[i]["color"]))

    out = np.zeros(shape, dtype=int)
    occupied = set()
    placements = {}
    reasons = {}

    for idx in order:
        bar = bars[idx]
        dr, dc = DIRS[bar["direction"]]
        cells = list(bar["cells"])
        cur = cells
        reason = None
        while True:
            nxt = [(r + dr, c + dc) for r, c in cur]
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in nxt):
                reason = "wall"
                break
            if any((r, c) in occupied for r, c in nxt):
                reason = "collision"
                break
            cur = nxt
        placements[idx] = cur
        reasons[idx] = reason
        for r, c in cur:
            occupied.add((r, c))
            out[r, c] = bar["color"]

    return out, placements, reasons, order


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each colored bar in the direction encoded by its nearby orange L marker, with shorter bars settling first and blocking longer ones.

    Input structure:
    - A black background contains 4-5 colored 1-pixel-thick straight bars, each a separate 4-connected component.
    - Each bar has a unique nearby orange 2x2 L-triomino marker whose missing corner encodes one of four translation directions.
    - Bars do not touch each other, markers do not touch each other, and every bar has exactly one nearest marker.

    Rule:
    - Pair every bar with its unique nearest orange L marker.
    - Decode the marker's missing corner as a translation direction: missing top-left -> left, top-right -> up, bottom-right -> right, bottom-left -> down.
    - Erase all orange marker pixels.
    - Process bars from shortest to longest; each bar slides one cell at a time in its decoded direction until the next step would leave the grid or overlap a bar that has already settled.
    - Paint only the settled bars on the black background, preserving each bar's shape and color.

    Key insight:
    - The orange marker does not encode color but motion, and the final arrangement depends on both decoding that motion correctly and applying the bars in length order so earlier settled bars become blockers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bars, _markers = _parse_scene(g)
    out, _placements, _reasons, _order = _simulate_bars(g.shape, bars)
    return out


def _bar_cells(orientation, length, anchor):
    r, c = anchor
    if orientation == "h":
        return [(r, c + k) for k in range(length)]
    return [(r + k, c) for k in range(length)]


def _touches_any(cells_a, cells_b):
    set_b = set(cells_b)
    for r, c in cells_a:
        if (r, c) in set_b:
            return True
    return False


def _bars_touch(cells_a, cells_b):
    set_b = set(cells_b)
    for r, c in cells_a:
        if (r, c) in set_b:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in set_b:
                return True
    return False


def _place_collision_pair(rng, shape, lengths):
    h, w = shape
    length0, length1 = lengths[:2]
    modes = [
        ("v", "left"),
        ("v", "right"),
        ("h", "up"),
        ("h", "down"),
    ]
    rng.shuffle(modes)
    for orientation, direction in modes:
        for _ in range(80):
            if orientation == "v" and direction == "left":
                overlap = rng.randint(1, min(length0, length1))
                start0 = rng.randint(1, h - length0 - 1)
                delta = rng.randint(-(length1 - overlap), length0 - overlap)
                start1 = start0 + delta
                if not (1 <= start1 <= h - length1 - 1):
                    continue
                bar0 = {"cells": _bar_cells("v", length0, (start0, 0)), "orientation": "v", "direction": "left", "length": length0}
                bar1 = {"cells": _bar_cells("v", length1, (start1, 1)), "orientation": "v", "direction": "left", "length": length1}
            elif orientation == "v" and direction == "right":
                overlap = rng.randint(1, min(length0, length1))
                start0 = rng.randint(1, h - length0 - 1)
                delta = rng.randint(-(length1 - overlap), length0 - overlap)
                start1 = start0 + delta
                if not (1 <= start1 <= h - length1 - 1):
                    continue
                bar0 = {"cells": _bar_cells("v", length0, (start0, w - 1)), "orientation": "v", "direction": "right", "length": length0}
                bar1 = {"cells": _bar_cells("v", length1, (start1, w - 2)), "orientation": "v", "direction": "right", "length": length1}
            elif orientation == "h" and direction == "up":
                overlap = rng.randint(1, min(length0, length1))
                start0 = rng.randint(1, w - length0 - 1)
                delta = rng.randint(-(length1 - overlap), length0 - overlap)
                start1 = start0 + delta
                if not (1 <= start1 <= w - length1 - 1):
                    continue
                bar0 = {"cells": _bar_cells("h", length0, (0, start0)), "orientation": "h", "direction": "up", "length": length0}
                bar1 = {"cells": _bar_cells("h", length1, (1, start1)), "orientation": "h", "direction": "up", "length": length1}
            else:
                overlap = rng.randint(1, min(length0, length1))
                start0 = rng.randint(1, w - length0 - 1)
                delta = rng.randint(-(length1 - overlap), length0 - overlap)
                start1 = start0 + delta
                if not (1 <= start1 <= w - length1 - 1):
                    continue
                bar0 = {"cells": _bar_cells("h", length0, (h - 1, start0)), "orientation": "h", "direction": "down", "length": length0}
                bar1 = {"cells": _bar_cells("h", length1, (h - 2, start1)), "orientation": "h", "direction": "down", "length": length1}

            if not _touches_any(bar0["cells"], bar1["cells"]):
                return [bar0, bar1]
    return None


def _can_use_final_bar(bar, settled):
    for other in settled:
        if _touches_any(bar["cells"], other["cells"]):
            return False
    return True


def _collision_ready(bar, settled):
    dr, dc = DIRS[bar["direction"]]
    nxt = [(r + dr, c + dc) for r, c in bar["cells"]]
    if any(r < 0 or c < 0 for r, c in nxt):
        return False
    h = max(r for r, _ in nxt) + 1
    w = max(c for _, c in nxt) + 1
    if any(r >= h or c >= w for r, c in nxt):
        pass
    nxt_set = set(nxt)
    return any(cell in nxt_set for other in settled for cell in other["cells"])


def _slack_for_input(bar, shape):
    h, w = shape
    dr, dc = DIRS[bar["direction"]]
    max_shift = None
    for r, c in bar["cells"]:
        if dr == -1:
            cur = h - 1 - r
        elif dr == 1:
            cur = r
        elif dc == -1:
            cur = w - 1 - c
        else:
            cur = c
        max_shift = cur if max_shift is None else min(max_shift, cur)
    return max_shift


def _random_final_bar(rng, shape, length, direction, settled, require_collision):
    h, w = shape
    orientations = ["h", "v"]
    rng.shuffle(orientations)
    for orientation in orientations:
        for _ in range(120):
            if orientation == "h":
                if w < length + 2:
                    continue
                c = rng.randint(1, w - length - 1)
                if direction == "left":
                    cells = _bar_cells("h", length, (rng.randint(1, h - 2), 1 if require_collision else 0))
                elif direction == "right":
                    cells = _bar_cells("h", length, (rng.randint(1, h - 2), (w - length - 1) if require_collision else (w - length)))
                elif direction == "up":
                    cells = _bar_cells("h", length, (1 if require_collision else 0, c))
                else:
                    cells = _bar_cells("h", length, ((h - 2) if require_collision else (h - 1), c))
            else:
                if h < length + 2:
                    continue
                r = rng.randint(1, h - length - 1)
                if direction == "left":
                    cells = _bar_cells("v", length, (r, 1 if require_collision else 0))
                elif direction == "right":
                    cells = _bar_cells("v", length, (r, (w - 2) if require_collision else (w - 1)))
                elif direction == "up":
                    cells = _bar_cells("v", length, (1 if require_collision else 0, rng.randint(1, w - 2)))
                else:
                    cells = _bar_cells("v", length, ((h - length - 1) if require_collision else (h - length), rng.randint(1, w - 2)))
            bar = {"cells": cells, "orientation": orientation, "direction": direction, "length": length}
            if _slack_for_input(bar, shape) < 2:
                continue
            if not _can_use_final_bar(bar, settled):
                continue
            dr, dc = DIRS[direction]
            nxt = [(rr + dr, cc + dc) for rr, cc in cells]
            if require_collision:
                if any(rr < 0 or rr >= h or cc < 0 or cc >= w for rr, cc in nxt):
                    continue
                if not any(cell in set(nxt) for other in settled for cell in other["cells"]):
                    continue
            else:
                if any(rr < 0 or rr >= h or cc < 0 or cc >= w for rr, cc in nxt):
                    pass
                else:
                    continue
            return bar
    return None


def _assign_input_positions(rng, shape, settled_bars):
    candidates = []
    for bar in settled_bars:
        max_shift = _slack_for_input(bar, shape)
        options = list(range(1, max_shift + 1))
        rng.shuffle(options)
        candidates.append(options)

    order = sorted(range(len(settled_bars)), key=lambda i: len(candidates[i]))
    placed = [None] * len(settled_bars)
    used = []

    def backtrack(pos):
        if pos == len(order):
            return True
        idx = order[pos]
        bar = settled_bars[idx]
        dr, dc = DIRS[bar["direction"]]
        for shift in candidates[idx]:
            cells = [(r - dr * shift, c - dc * shift) for r, c in bar["cells"]]
            trial = {"cells": cells}
            if any(_bars_touch(cells, other["cells"]) for other in used):
                continue
            placed[idx] = {
                "cells": cells,
                "orientation": bar["orientation"],
                "direction": bar["direction"],
                "length": bar["length"],
            }
            used.append(trial)
            if backtrack(pos + 1):
                return True
            used.pop()
            placed[idx] = None
        return False

    if not backtrack(0):
        return None
    return placed


def _marker_cells(top_left, missing):
    r0, c0 = top_left
    cells = []
    for dr in (0, 1):
        for dc in (0, 1):
            if (dr, dc) != missing:
                cells.append((r0 + dr, c0 + dc))
    return cells


def _assign_markers(rng, shape, bars):
    h, w = shape
    occupied = set()
    for bar in bars:
        occupied.update(bar["cells"])
    markers = [None] * len(bars)

    bar_order = sorted(range(len(bars)), key=lambda i: bars[i]["length"], reverse=True)

    def valid_marker(idx, cells):
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
            return False
        if any((r, c) in occupied for r, c in cells):
            return False
        for other in markers:
            if other is None:
                continue
            if _bars_touch(cells, other["cells"]):
                return False

        own_dist = _min_manhattan(bars[idx]["cells"], cells)
        if own_dist not in (1, 2):
            return False
        for j, other_bar in enumerate(bars):
            if j == idx:
                continue
            if _min_manhattan(other_bar["cells"], cells) <= own_dist:
                return False
        return True

    def backtrack(pos):
        if pos == len(bar_order):
            return True
        idx = bar_order[pos]
        bar = bars[idx]
        r0, c0, r1, c1 = _bbox(bar["cells"])
        missing = DIR_TO_MISSING[bar["direction"]]
        candidates = []
        for rr in range(r0 - 3, r1 + 3):
            for cc in range(c0 - 3, c1 + 3):
                cells = _marker_cells((rr, cc), missing)
                if valid_marker(idx, cells):
                    candidates.append(cells)
        rng.shuffle(candidates)
        for cells in candidates:
            markers[idx] = {"cells": cells, "missing": missing}
            occupied.update(cells)
            if backtrack(pos + 1):
                return True
            for cell in cells:
                occupied.remove(cell)
            markers[idx] = None
        return False

    if not backtrack(0):
        return None
    return markers


def _build_grid(shape, bars, markers, colors):
    grid = np.zeros(shape, dtype=int)
    for bar, color in zip(bars, colors):
        for r, c in bar["cells"]:
            grid[r, c] = color
    for marker in markers:
        for r, c in marker["cells"]:
            grid[r, c] = MARKER_COLOR
    return grid


def _bars_from_input(input_bars, colors):
    bars = []
    for bar, color in zip(input_bars, colors):
        cells = sorted(bar["cells"])
        bars.append(
            {
                "cells": cells,
                "color": color,
                "bbox": _bbox(cells),
                "area": len(cells),
                "length": len(cells),
                "direction": bar["direction"],
            }
        )
    return bars


def _validate_example(grid):
    try:
        bars, markers = _parse_scene(grid)
    except AssertionError:
        return False
    lengths = sorted(bar["length"] for bar in bars)
    if len(lengths) != len(set(lengths)):
        return False
    try:
        out, _placements, reasons, order = _simulate_bars(grid.shape, bars)
    except AssertionError:
        return False
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if "wall" not in reasons.values() or "collision" not in reasons.values():
        return False
    reversed_out, _p2, _r2, _o2 = _simulate_bars(grid.shape, bars, list(reversed(order)))
    if np.array_equal(out, reversed_out):
        return False
    marker_indices = [bar["marker_idx"] for bar in bars]
    if len(marker_indices) != len(set(marker_indices)) or len(markers) != len(bars):
        return False
    return True


def _gen_attempt(rng):
    h = rng.randint(16, 22)
    w = rng.randint(16, 22)
    count = rng.randint(4, 5)
    lengths = sorted(rng.sample(range(3, 8), count))
    colors = rng.sample(list(BAR_COLORS), count)

    pair = _place_collision_pair(rng, (h, w), lengths)
    if pair is None:
        return None
    settled = pair

    base_dirs = ["left", "up", "right", "down"]
    rng.shuffle(base_dirs)
    directions = base_dirs[:count]
    if count == 5:
        directions.append(rng.choice(base_dirs))
    directions[0] = pair[0]["direction"]
    directions[1] = pair[1]["direction"]

    remaining_dirs = directions[2:]
    for length, direction in zip(lengths[2:], remaining_dirs):
        require_collision = rng.random() < 0.45
        bar = _random_final_bar(rng, (h, w), length, direction, settled, require_collision)
        if bar is None and require_collision:
            bar = _random_final_bar(rng, (h, w), length, direction, settled, False)
        if bar is None:
            return None
        settled.append(bar)

    input_bars = _assign_input_positions(rng, (h, w), settled)
    if input_bars is None:
        return None

    markers = _assign_markers(rng, (h, w), input_bars)
    if markers is None:
        return None

    grid = _build_grid((h, w), input_bars, markers, colors)
    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
