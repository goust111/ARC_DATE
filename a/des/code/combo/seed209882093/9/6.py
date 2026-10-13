from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME_COLOR = 6
ALL_COLORS = tuple(range(10))
DIR_PRIORITY = ("top", "right", "bottom", "left")
DIR_FROM_WALL = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}

GENERATION_ATTEMPTS = 400
OBJECT_PLACEMENT_ATTEMPTS = 200

BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
)

SINGLE_PATTERNS = (
    ("top",),
    ("right",),
    ("bottom",),
    ("left",),
)
PAIR_PATTERNS = (
    ("top", "right"),
    ("right", "bottom"),
    ("bottom", "left"),
    ("left", "top"),
)
PATTERN_POOL = ((),) + SINGLE_PATTERNS + PAIR_PATTERNS


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_walls(coords: np.ndarray, h: int, w: int) -> dict[str, bool]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return {
        "top": bool(np.any(rows == 1)),
        "right": bool(np.any(cols == w - 2)),
        "bottom": bool(np.any(rows == h - 2)),
        "left": bool(np.any(cols == 1)),
    }


def _copy_direction(touches: dict[str, bool] | set[str] | tuple[str, ...]) -> tuple[int, int] | None:
    if isinstance(touches, dict):
        present = {wall for wall, active in touches.items() if active}
    else:
        present = set(touches)
    for wall in DIR_PRIORITY:
        if wall in present:
            return DIR_FROM_WALL[wall]
    return None


def _plan_copies(coords: np.ndarray, h: int, w: int) -> tuple[int, list[np.ndarray]]:
    touches = _touches_walls(coords, h, w)
    k = int(sum(touches.values()))
    if k == 0:
        return 0, []
    direction = _copy_direction(touches)
    if direction is None:
        return k, []
    r0, c0, r1, c1 = _bbox(coords)
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    step = box_h + 1 if direction[0] else box_w + 1
    dr, dc = direction
    copies: list[np.ndarray] = []
    for i in range(1, k + 1):
        shifted = coords.copy()
        shifted[:, 0] += dr * step * i
        shifted[:, 1] += dc * step * i
        copies.append(shifted)
    return k, copies


def _corner_segment(h: int, w: int, corner: str, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == "h":
        row = 1 if corner in ("tl", "tr") else h - 2
        if corner in ("tl", "bl"):
            cols = range(1, 1 + length)
        else:
            cols = range(w - 1 - length, w - 1)
        return [(row, c) for c in cols]
    col = 1 if corner in ("tl", "bl") else w - 2
    if corner in ("tl", "tr"):
        rows = range(1, 1 + length)
    else:
        rows = range(h - 1 - length, h - 1)
    return [(r, col) for r in rows]


def _is_legend_candidate(grid: np.ndarray, bg: int, frame: int, cells: list[tuple[int, int]]) -> tuple[bool, list[int]]:
    h, w = grid.shape
    cell_set = set(cells)
    colors = [int(grid[r, c]) for r, c in cells]
    if any(color in (bg, frame) for color in colors):
        return False, []
    if len(set(colors)) != len(colors):
        return False, []
    for r, c in cells:
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cell_set:
                val = int(grid[rr, cc])
                if val != bg and val != frame:
                    return False, []
    return True, colors


def _find_legend(grid: np.ndarray, bg: int, frame: int) -> tuple[list[int], np.ndarray]:
    candidates: list[tuple[int, list[int], list[tuple[int, int]]]] = []
    for length in (4, 3):
        for corner in ("tl", "tr", "bl", "br"):
            for orient in ("h", "v"):
                cells = _corner_segment(grid.shape[0], grid.shape[1], corner, orient, length)
                ok, colors = _is_legend_candidate(grid, bg, frame, cells)
                if ok:
                    candidates.append((length, colors, cells))
    assert candidates, "legend not found"
    candidates.sort(key=lambda item: (-item[0], tuple(item[2])))
    _, colors, cells = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return colors, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each wall-touching object away from the first touched wall
    (priority: top > right > bottom > left). The number of touched walls sets
    both the copy count and the lookup index in the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    frame = int(g[0, 0])
    assert np.all(g[0, :] == frame)
    assert np.all(g[-1, :] == frame)
    assert np.all(g[:, 0] == frame)
    assert np.all(g[:, -1] == frame)

    interior_vals = [int(v) for v in g[1:-1, 1:-1].ravel() if int(v) != frame]
    bg = int(Counter(interior_vals).most_common(1)[0][0])
    legend_colors, legend_mask = _find_legend(g, bg, frame)

    out = g.copy()
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, frame)):
        mask = (g == int(color)) & (~legend_mask)
        if not mask.any():
            continue
        components = sorted(_components(mask), key=_bbox)
        for coords in components:
            k, copies = _plan_copies(coords, h, w)
            if k == 0 or k > len(legend_colors):
                continue
            paint = int(legend_colors[k - 1])
            for copy in copies:
                for rr, cc in copy:
                    rr = int(rr)
                    cc = int(cc)
                    if 1 <= rr < h - 1 and 1 <= cc < w - 1:
                        out[rr, cc] = paint
    return out


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(norm)


def _random_shape() -> tuple[tuple[int, int], ...]:
    cells = list(random.choice(BASE_SHAPES))
    for _ in range(random.randrange(4)):
        cells = [(c, -r) for r, c in cells]
    if random.random() < 0.5:
        cells = [(r, -c) for r, c in cells]
    if random.random() < 0.5:
        cells = [(-r, c) for r, c in cells]
    return _normalize(cells)


def _dilate4(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _sample_patterns(count: int) -> list[tuple[str, ...]]:
    patterns: list[tuple[str, ...]] = [(), random.choice(SINGLE_PATTERNS)]
    if count >= 3 and random.random() < 0.75:
        patterns.append(random.choice(PAIR_PATTERNS))
    while len(patterns) < count:
        patterns.append(random.choice(PATTERN_POOL))
    random.shuffle(patterns)
    return patterns[:count]


def _place_shape(h: int, w: int, shape: tuple[tuple[int, int], ...], pattern: tuple[str, ...]) -> np.ndarray | None:
    local = np.array(shape, dtype=int)
    _, _, r1, c1 = _bbox(local)
    box_h = r1 + 1
    box_w = c1 + 1
    sides = set(pattern)

    row_min = 1 if "top" in sides else 2
    row_max = (h - 1 - box_h) if "bottom" in sides else (h - 2 - box_h)
    col_min = 1 if "left" in sides else 2
    col_max = (w - 1 - box_w) if "right" in sides else (w - 2 - box_w)

    if "top" in sides and "bottom" in sides:
        return None
    if "left" in sides and "right" in sides:
        return None

    k = len(pattern)
    direction = _copy_direction(sides)
    if direction is not None:
        dr, dc = direction
        if dr > 0:
            row_max = min(row_max, h - box_h - 1 - k * (box_h + 1))
        elif dr < 0:
            row_min = max(row_min, 1 + k * (box_h + 1))
        elif dc > 0:
            col_max = min(col_max, w - box_w - 1 - k * (box_w + 1))
        elif dc < 0:
            col_min = max(col_min, 1 + k * (box_w + 1))

    if row_min > row_max or col_min > col_max:
        return None

    top = random.randint(row_min, row_max)
    left = random.randint(col_min, col_max)
    placed = local.copy()
    placed[:, 0] += top
    placed[:, 1] += left
    return placed


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.choice((3, 4))
    object_count = 3 if legend_len == 4 else random.randint(3, 4)
    size_lo, size_hi = (14, 20) if object_count == 4 else (12, 18)
    h = random.randint(size_lo, size_hi)
    w = random.randint(size_lo, size_hi)

    bg = random.choice([c for c in ALL_COLORS if c != FRAME_COLOR])
    remaining = [c for c in ALL_COLORS if c not in (FRAME_COLOR, bg)]
    legend_colors = random.sample(remaining, legend_len)
    object_colors = random.sample([c for c in remaining if c not in legend_colors], object_count)

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :] = FRAME_COLOR
    grid[-1, :] = FRAME_COLOR
    grid[:, 0] = FRAME_COLOR
    grid[:, -1] = FRAME_COLOR

    corner = random.choice(("tl", "tr", "bl", "br"))
    orient = random.choice(("h", "v"))
    legend_cells = _corner_segment(h, w, corner, orient, legend_len)
    for color, (r, c) in zip(legend_colors, legend_cells):
        grid[r, c] = int(color)

    patterns = _sample_patterns(object_count)
    input_occupied = set(legend_cells)
    final_occupied = set(legend_cells)
    input_blocked = _dilate4(input_occupied, h, w)
    final_blocked = _dilate4(final_occupied, h, w)
    pair_count = 0

    for color, pattern in zip(object_colors, patterns):
        placed = None
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            shape = _random_shape()
            coords = _place_shape(h, w, shape, pattern)
            if coords is None:
                continue
            touches = _touches_walls(coords, h, w)
            k, copies = _plan_copies(coords, h, w)
            if k != len(pattern):
                continue
            if set(wall for wall, active in touches.items() if active) != set(pattern):
                continue

            original_cells = _coords_to_set(coords)
            if original_cells & input_blocked:
                continue

            full_cells = set(original_cells)
            valid = True
            for copy in copies:
                copy_cells = _coords_to_set(copy)
                if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in copy_cells):
                    valid = False
                    break
                if full_cells & copy_cells:
                    valid = False
                    break
                full_cells |= copy_cells
            if not valid or full_cells & final_blocked:
                continue

            placed = coords
            input_occupied |= original_cells
            final_occupied |= full_cells
            input_blocked = _dilate4(input_occupied, h, w)
            final_blocked = _dilate4(final_occupied, h, w)
            grid[coords[:, 0], coords[:, 1]] = int(color)
            if len(pattern) == 2:
                pair_count += 1
            break
        if placed is None:
            return None

    if pair_count == 0 and random.random() < 0.6:
        return None

    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
