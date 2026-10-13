from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_direction(
    marker_r: int, marker_c: int, r0: int, c0: int, r1: int, c1: int
) -> str:
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = marker_r - center_r
    dc = marker_c - center_c

    if marker_r < center_r and abs(dc) <= 0.5:
        return "up"
    if marker_r > center_r and abs(dc) <= 0.5:
        return "down"
    if marker_c < center_c and abs(dr) <= 0.5:
        return "left"
    if marker_c > center_c and abs(dr) <= 0.5:
        return "right"
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each foreground object onto the grid edge indicated by its internal marker.

    Each object is a 4-connected component on a grey background. Inside each object,
    exactly one differently colored singleton pixel acts as a direction marker:
    above the bbox center -> up, below -> down, left -> left, right -> right.
    The original object is erased and replaced by a 1-pixel-thick boundary segment
    whose extent matches the object's bbox span perpendicular to the projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)

    out = np.full((h, w), bg, dtype=int)
    for comp in _components(g != bg):
        colors = Counter(int(g[r, c]) for r, c in comp)
        marker_colors = [color for color, count in colors.items() if count == 1]
        assert len(marker_colors) == 1, "expected one singleton marker color per object"
        marker_color = marker_colors[0]
        body_colors = [color for color in colors if color != marker_color]
        assert len(body_colors) == 1, "expected one body color per object"
        body_color = body_colors[0]

        marker_r, marker_c = next((r, c) for r, c in comp if int(g[r, c]) == marker_color)
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        direction = _infer_direction(marker_r, marker_c, r0, c0, r1, c1)

        if direction == "up":
            out[0, c0 : c1 + 1] = body_color
        elif direction == "down":
            out[h - 1, c0 : c1 + 1] = body_color
        elif direction == "left":
            out[r0 : r1 + 1, 0] = body_color
        else:
            out[r0 : r1 + 1, w - 1] = body_color

    return out


BG = 5
BODY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ["up", "down", "left", "right"]
GRID_MIN = 12
GRID_MAX = 20
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 4
SIZE_CHOICES = [3, 5, 7]
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _shape_cells(height: int, width: int, rng: random.Random) -> set[tuple[int, int]]:
    cr, cc = height // 2, width // 2
    cells = {(r, cc) for r in range(height)}
    cells |= {(cr, c) for c in range(width)}
    target = rng.randint(len(cells), min(height * width - 1, len(cells) + (height * width) // 3))
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    return cells


def _marker_pos(
    cells: set[tuple[int, int]], height: int, width: int, direction: str, rng: random.Random
) -> tuple[int, int]:
    cr, cc = height // 2, width // 2
    if direction == "up":
        options = [(r, cc) for r in range(0, cr) if (r, cc) in cells]
    elif direction == "down":
        options = [(r, cc) for r in range(cr + 1, height) if (r, cc) in cells]
    elif direction == "left":
        options = [(cr, c) for c in range(0, cc) if (cr, c) in cells]
    else:
        options = [(cr, c) for c in range(cc + 1, width) if (cr, c) in cells]
    assert options
    return rng.choice(options)


def _with_margin(r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0 - 1, r0 + height + 1):
        for c in range(c0 - 1, c0 + width + 1):
            cells.add((r, c))
    return cells


def _segment_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _projection_conflict(
    specs: list[tuple[str, int, int]], direction: str, start: int, stop: int
) -> bool:
    for side, s0, s1 in specs:
        if side == direction and _segment_overlap(s0, s1, start, stop):
            return True
    return False


def _place_object(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    edge_specs: list[tuple[str, int, int]],
    body_color: int,
    marker_color: int,
    direction: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    dims = [(hh, ww) for hh in SIZE_CHOICES for ww in SIZE_CHOICES]
    rng.shuffle(dims)
    for height, width in dims:
        if height > h - 2 or width > w - 2:
            continue
        cells = _shape_cells(height, width, rng)
        mr, mc = _marker_pos(cells, height, width, direction, rng)
        starts = [(r0, c0) for r0 in range(1, h - height) for c0 in range(1, w - width)]
        rng.shuffle(starts)
        span0, span1 = (0, 0)
        if direction in {"up", "down"}:
            span0, span1 = 0, width - 1
        else:
            span0, span1 = 0, height - 1
        for r0, c0 in starts[:PLACE_ATTEMPTS]:
            abs_cells = {(r0 + r, c0 + c) for r, c in cells}
            if abs_cells & occupied:
                continue
            if direction in {"up", "down"}:
                start, stop = c0 + span0, c0 + span1
            else:
                start, stop = r0 + span0, r0 + span1
            if _projection_conflict(edge_specs, direction, start, stop):
                continue

            for r, c in abs_cells:
                grid[r, c] = body_color
            grid[r0 + mr, c0 + mc] = marker_color
            occupied |= _with_margin(r0, c0, height, width)
            edge_specs.append((direction, start, stop))
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        height = rng.randint(GRID_MIN, GRID_MAX)
        width = rng.randint(GRID_MIN, GRID_MAX)
        grid = np.full((height, width), BG, dtype=int)

        max_objects = min(OBJECT_COUNT_MAX, len(BODY_COLORS) // 2, max(1, min(height, width) // 4))
        n_objects = rng.randint(OBJECT_COUNT_MIN, max_objects)
        palette = BODY_COLORS[:]
        rng.shuffle(palette)
        dirs = [rng.choice(DIRECTIONS) for _ in range(n_objects)]

        occupied: set[tuple[int, int]] = set()
        edge_specs: list[tuple[str, int, int]] = []
        ok = True
        for idx in range(n_objects):
            body_color = palette[2 * idx]
            marker_color = palette[2 * idx + 1]
            if not _place_object(
                grid, occupied, edge_specs, body_color, marker_color, dirs[idx], rng
            ):
                ok = False
                break

        if ok:
            main(grid)
            return grid
    raise RuntimeError("failed to generate a valid input")
