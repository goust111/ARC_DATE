import random
from collections import Counter, deque

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
SHAPE_FAMILIES = ("rectangle", "l_shape", "t_shape", "notch", "profile")


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    h, w = grid.shape
    colors = Counter(int(grid[r, c]) for r, c in comp if int(grid[r, c]) != BG)
    assert len(colors) == 2, "each object must have a body color and one marker color"
    marker_candidates = [color for color, count in colors.items() if count == 1]
    assert len(marker_candidates) == 1, "each object must have exactly one marker pixel"
    marker_color = int(marker_candidates[0])
    body_color = int(next(color for color in colors if color != marker_color))
    marker_pos = next((int(r), int(c)) for r, c in comp if int(grid[r, c]) == marker_color)

    comp_set = {(int(r), int(c)) for r, c in comp}
    boundary: set[tuple[int, int]] = set()
    perimeter = 0
    for r, c in comp_set:
        exposed = False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG:
                perimeter += 1
                exposed = True
        if exposed:
            boundary.add((r, c))

    return {
        "cells": comp_set,
        "marker_pos": marker_pos,
        "marker_color": marker_color,
        "body_color": body_color,
        "boundary": boundary,
        "perimeter": perimeter,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink odd-perimeter objects and expand even-perimeter objects.

    The perimeter counted here is the number of object sides adjacent to black
    cells inside the grid. Out-of-bounds sides are not counted.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    additions: dict[tuple[int, int], int] = {}

    for comp in _components(g != BG):
        info = _analyze_component(g, comp)
        if info["perimeter"] % 2 == 1:
            for r, c in info["boundary"]:
                if (r, c) != info["marker_pos"]:
                    out[r, c] = BG
        else:
            for r, c in info["cells"]:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == BG:
                        additions.setdefault((nr, nc), int(info["marker_color"]))

    for (r, c), color in additions.items():
        out[r, c] = color
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, _ = _shape_dims(cells)
    return _normalize({(c, h - 1 - r) for r, c in cells})


def _flip(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    _, w = _shape_dims(cells)
    return _normalize({(r, w - 1 - c) for r, c in cells})


def _random_transform(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    shape = set(cells)
    if random.random() < 0.5:
        shape = _flip(shape)
    for _ in range(random.randrange(4)):
        shape = _rotate(shape)
    return _normalize(shape)


def _rectangle_shape() -> set[tuple[int, int]]:
    candidates = [(h, w) for h in range(2, 6) for w in range(2, 6) if 6 <= h * w <= 20]
    h, w = random.choice(candidates)
    return {(r, c) for r in range(h) for c in range(w)}


def _l_shape() -> set[tuple[int, int]]:
    candidates = []
    for h in range(3, 7):
        for w in range(3, 7):
            for t in range(1, min(2, h - 1, w - 1) + 1):
                area = h * t + w * t - t * t
                if 6 <= area <= 20:
                    candidates.append((h, w, t))
    h, w, t = random.choice(candidates)
    return {(r, c) for r in range(h) for c in range(w) if r < t or c < t}


def _t_shape() -> set[tuple[int, int]]:
    candidates = []
    for bar_h in (1, 2):
        for bar_w in range(3, 7):
            for stem_h in range(2, 6):
                for stem_w in range(1, min(2, bar_w) + 1):
                    area = bar_h * bar_w + stem_h * stem_w
                    if 6 <= area <= 20:
                        candidates.append((bar_h, bar_w, stem_h, stem_w))
    bar_h, bar_w, stem_h, stem_w = random.choice(candidates)
    start = (bar_w - stem_w) // 2
    cells = {(r, c) for r in range(bar_h) for c in range(bar_w)}
    for r in range(bar_h, bar_h + stem_h):
        for c in range(start, start + stem_w):
            cells.add((r, c))
    return cells


def _notch_shape() -> set[tuple[int, int]]:
    candidates = []
    for h in range(3, 7):
        for w in range(3, 7):
            for notch_h in range(1, h - 1):
                for notch_w in range(1, w - 1):
                    area = h * w - notch_h * notch_w
                    if 6 <= area <= 20:
                        candidates.append((h, w, notch_h, notch_w))
    h, w, notch_h, notch_w = random.choice(candidates)
    corner = random.choice(("tl", "tr", "bl", "br"))
    cells = {(r, c) for r in range(h) for c in range(w)}
    for r in range(notch_h):
        for c in range(notch_w):
            rr = r if "t" in corner else h - 1 - r
            cc = c if "l" in corner else w - 1 - c
            cells.discard((rr, cc))
    return cells


def _profile_shape() -> set[tuple[int, int]]:
    for _ in range(60):
        rows = random.randint(2, 5)
        widths = [random.randint(2, 4) for _ in range(rows)]
        if not (6 <= sum(widths) <= 20):
            continue
        offsets = [0]
        ok = True
        for idx in range(1, rows):
            prev_off = offsets[-1]
            prev_w = widths[idx - 1]
            cur_w = widths[idx]
            low = prev_off - cur_w + 1
            high = prev_off + prev_w - 1
            if low > high:
                ok = False
                break
            offsets.append(random.randint(low, high))
        if not ok:
            continue
        cells = {(r, offsets[r] + c) for r in range(rows) for c in range(widths[r])}
        shape = _normalize(cells)
        h, w = _shape_dims(shape)
        if h <= 6 and w <= 6:
            return shape
    return _rectangle_shape()


def _make_shape() -> set[tuple[int, int]]:
    family = random.choice(SHAPE_FAMILIES)
    if family == "rectangle":
        shape = _rectangle_shape()
    elif family == "l_shape":
        shape = _l_shape()
    elif family == "t_shape":
        shape = _t_shape()
    elif family == "notch":
        shape = _notch_shape()
    else:
        shape = _profile_shape()
    return _random_transform(shape)


def _buffered_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 2) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _placement_info(
    shape: set[tuple[int, int]],
    top: int,
    left: int,
    h: int,
    w: int,
) -> dict:
    placed = {(top + r, left + c) for r, c in shape}
    perimeter = 0
    boundary = set()
    touch_sides = set()
    for r, c in placed:
        if r == 0:
            touch_sides.add("top")
        if r == h - 1:
            touch_sides.add("bottom")
        if c == 0:
            touch_sides.add("left")
        if c == w - 1:
            touch_sides.add("right")
        exposed = False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in placed:
                perimeter += 1
                exposed = True
        if exposed:
            boundary.add((r, c))
    return {
        "cells": placed,
        "boundary": boundary,
        "perimeter": perimeter,
        "touch_count": len(touch_sides),
        "touches_border": bool(touch_sides),
    }


def _choose_placement(
    shape: set[tuple[int, int]],
    h: int,
    w: int,
    reserved: np.ndarray,
    target_parity: int,
) -> dict | None:
    sh, sw = _shape_dims(shape)
    candidates = []
    border_candidates = []
    interior_candidates = []
    side_touch_candidates = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            info = _placement_info(shape, top, left, h, w)
            if info["perimeter"] % 2 != target_parity:
                continue
            buffered = _buffered_cells(info["cells"], h, w, pad=2)
            if any(reserved[r, c] for r, c in buffered):
                continue
            info["buffered"] = buffered
            candidates.append(info)
            if info["touches_border"]:
                border_candidates.append(info)
            else:
                interior_candidates.append(info)
            if info["touch_count"] == 1:
                side_touch_candidates.append(info)
    if not candidates:
        return None

    if target_parity == 1:
        pool = side_touch_candidates or border_candidates or candidates
    else:
        if interior_candidates and random.random() < 0.7:
            pool = interior_candidates
        else:
            pool = candidates
    return random.choice(pool)


def _choose_marker(
    placed: set[tuple[int, int]],
    boundary: set[tuple[int, int]],
    prefer_boundary: bool,
) -> tuple[int, int]:
    boundary_cells = list(boundary)
    inner_cells = [cell for cell in placed if cell not in boundary]
    if prefer_boundary and boundary_cells:
        return random.choice(boundary_cells)
    if inner_cells:
        return random.choice(inner_cells)
    return random.choice(boundary_cells or list(placed))


def _parity_targets(n_objects: int) -> list[int]:
    if n_objects == 2:
        targets = [0, 1]
    elif n_objects == 3:
        targets = [0, 1, random.randint(0, 1)]
    else:
        targets = [0, 0, 1, 1]
    random.shuffle(targets)
    return targets


def _validate_generated(grid: np.ndarray) -> bool:
    comps = _components(grid != BG)
    if len(comps) < 2:
        return False

    parities = []
    expansion_targets: set[tuple[int, int]] = set()
    for comp in comps:
        try:
            info = _analyze_component(grid, comp)
        except AssertionError:
            return False
        parities.append(info["perimeter"] % 2)
        if info["perimeter"] % 2 == 0:
            for r, c in info["cells"]:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == BG:
                        if (nr, nc) in expansion_targets:
                            return False
                        expansion_targets.add((nr, nc))
    if 0 not in parities or 1 not in parities:
        return False
    out = main(grid)
    return out.shape == grid.shape and np.any(out != grid)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice([2, 3, 4])
    min_side = 14 + 3 * max(0, n_objects - 2)
    h = random.randint(min_side, 26)
    w = random.randint(min_side, 26)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    colors = random.sample(range(1, 10), 2 * n_objects)
    body_colors = colors[:n_objects]
    marker_colors = colors[n_objects:]
    targets = _parity_targets(n_objects)

    for idx in range(n_objects):
        placed_info = None
        for _ in range(100):
            shape = _make_shape()
            placed_info = _choose_placement(shape, h, w, reserved, targets[idx])
            if placed_info is not None:
                break
        if placed_info is None:
            return None

        boundary = placed_info["boundary"]
        prefer_boundary = targets[idx] == 1 and random.random() < 0.5
        marker_pos = _choose_marker(placed_info["cells"], boundary, prefer_boundary)
        for r, c in placed_info["cells"]:
            grid[r, c] = body_colors[idx]
        grid[marker_pos] = marker_colors[idx]
        for r, c in placed_info["buffered"]:
            reserved[r, c] = True

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
