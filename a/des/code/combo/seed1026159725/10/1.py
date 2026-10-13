from __future__ import annotations

from collections import deque
import random

import numpy as np


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome object using the legend row associated with its bbox aspect:
    top row -> horizontal, middle row -> square, bottom row -> vertical.
    Then erase the 3-row legend area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4

    bg = _mode(g[3:, :].ravel())
    legend_colors: list[int] = []
    for r in range(3):
        row_colors = [int(v) for v in np.unique(g[r, :]) if int(v) != bg]
        assert len(row_colors) == 1, "each legend row must have exactly one non-background color"
        legend_colors.append(row_colors[0])

    body_colors = [int(v) for v in np.unique(g[3:, :]) if int(v) != bg]
    assert len(body_colors) == 1, "expected exactly one foreground object color below the legend"
    object_color = body_colors[0]

    object_mask = np.zeros_like(g, dtype=bool)
    object_mask[3:, :] = g[3:, :] == object_color

    out = g.copy()
    out[:3, :] = bg
    for comp in _components(object_mask):
        r0, c0, r1, c1 = _bbox(comp)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if width > height:
            new_color = legend_colors[0]
        elif width == height:
            new_color = legend_colors[1]
        else:
            new_color = legend_colors[2]
        for r, c in comp:
            out[r, c] = new_color
    return out


_COLORS = tuple(range(10))
_GRID_H_RANGE = (14, 22)
_GRID_W_RANGE = (12, 22)
_EXTRA_OBJECTS = (0, 1, 2, 3)
_MAX_ATTEMPTS = 400


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _connect_path(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    row_first: bool,
) -> None:
    r, c = start
    er, ec = end
    if row_first:
        while r != er:
            r += _sign(er - r)
            cells.add((r, c))
        while c != ec:
            c += _sign(ec - c)
            cells.add((r, c))
    else:
        while c != ec:
            c += _sign(ec - c)
            cells.add((r, c))
        while r != er:
            r += _sign(er - r)
            cells.add((r, c))


def _grow_connected_shape(height: int, width: int) -> np.ndarray:
    if height == 1 and width == 1:
        return np.ones((1, 1), dtype=bool)

    style = random.choice(("solid", "grown", "grown", "grown"))
    if style == "solid":
        return np.ones((height, width), dtype=bool)

    anchors = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]
    cells = {anchors[0]}
    cur = anchors[0]
    for nxt in anchors[1:]:
        _connect_path(cells, cur, nxt, row_first=bool(random.getrandbits(1)))
        cur = nxt

    total = height * width
    min_area = min(total, max(len(cells), int(np.ceil(total * 0.35))))
    max_area = min(total, max(min_area, int(np.ceil(total * 0.8))))
    target_area = random.randint(min_area, max_area)

    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_bbox(kind: str, max_height: int, max_width: int) -> tuple[int, int] | None:
    if kind == "horizontal":
        max_h = min(4, max_height)
        if max_h < 1 or max_width < 2:
            return None
        height = random.randint(1, max_h)
        min_w = height + 1
        if max_width < min_w:
            return None
        width = random.randint(min_w, min(max_width, min_w + 5))
        return height, width
    if kind == "vertical":
        max_w = min(4, max_width)
        if max_w < 1 or max_height < 2:
            return None
        width = random.randint(1, max_w)
        min_h = width + 1
        if max_height < min_h:
            return None
        height = random.randint(min_h, min(max_height, min_h + 5))
        return height, width

    side_max = min(6, max_height, max_width)
    if side_max < 1:
        return None
    side = random.randint(1, side_max)
    return side, side


def _can_place(grid: np.ndarray, bg: int, top: int, left: int, mask: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = mask.shape
    if top < 4 or left < 1 or top + ph > h - 1 or left + pw > w - 1:
        return False
    return bool(np.all(grid[top - 1 : top + ph + 1, left - 1 : left + pw + 1] == bg))


def _place_mask(grid: np.ndarray, color: int, top: int, left: int, mask: np.ndarray) -> None:
    ph, pw = mask.shape
    region = grid[top : top + ph, left : left + pw]
    region[mask] = color


def _sample_object_specs() -> list[str]:
    specs = ["horizontal", "square", "vertical"]
    for _ in range(random.choice(_EXTRA_OBJECTS)):
        specs.append(random.choice(("horizontal", "square", "vertical")))
    random.shuffle(specs)
    return specs


def _build_case() -> np.ndarray | None:
    height = random.randint(*_GRID_H_RANGE)
    width = random.randint(*_GRID_W_RANGE)
    bg, object_color, horiz_color, square_color, vert_color = random.sample(_COLORS, 5)

    grid = np.full((height, width), bg, dtype=int)
    grid[0, :] = horiz_color
    grid[1, :] = square_color
    grid[2, :] = vert_color

    specs = _sample_object_specs()
    body_bg_budget = (height - 3) * width
    object_area = 0
    for kind in specs:
        placed = False
        for _ in range(80):
            dims = _sample_bbox(kind, max_height=height - 5, max_width=width - 2)
            if dims is None:
                return None
            box_h, box_w = dims
            mask = _grow_connected_shape(box_h, box_w)
            max_top = height - box_h - 1
            max_left = width - box_w - 1
            if max_top < 4 or max_left < 1:
                continue
            top = random.randint(4, max_top)
            left = random.randint(1, max_left)
            if not _can_place(grid, bg, top, left, mask):
                continue
            _place_mask(grid, object_color, top, left, mask)
            object_area += int(mask.sum())
            placed = True
            break
        if not placed:
            return None

    if object_area == 0 or body_bg_budget - object_area <= object_area:
        return None
    if _mode(grid.ravel()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    colors_in_output = {int(v) for v in np.unique(out) if int(v) != bg}
    if not {horiz_color, square_color, vert_color}.issubset(colors_in_output):
        return None
    if np.any(out[:3, :] != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
