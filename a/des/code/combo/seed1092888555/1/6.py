from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4

GEN_ATTEMPTS = 400
PANEL_SIZES = (6, 8)
GRID_LAYOUTS = (
    (1, 4),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
)
INSTRUCTIONS = ("tl", "tr", "bl", "br")


def _separator_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(indices)
    runs: list[tuple[int, int]] = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            runs.append((start, end))
            start = end = idx
    runs.append((start, end))
    return runs


def _intervals_from_separators(length: int, sep_indices: list[int]) -> list[tuple[int, int]]:
    runs = _separator_runs(sep_indices)
    if not runs:
        return [(0, length)]
    out: list[tuple[int, int]] = []
    start = 0
    for run_start, run_end in runs:
        out.append((start, run_start))
        start = run_end + 1
    out.append((start, length))
    assert all(b > a for a, b in out), "separators must leave non-empty panels"
    return out


def _connected4(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _tight_patch(panel: np.ndarray, color: int) -> np.ndarray:
    pts = np.argwhere(panel == color)
    assert pts.size > 0, "expected a non-empty object"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return panel[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _transform_patch(patch: np.ndarray, instruction: str) -> np.ndarray:
    if instruction == "tl":
        return patch.copy()
    if instruction == "tr":
        return np.fliplr(patch)
    if instruction == "bl":
        return np.flipud(patch)
    if instruction == "br":
        return np.rot90(patch, 2)
    raise AssertionError("unknown instruction")


def _marker_instruction(panel: np.ndarray) -> str:
    red_pts = np.argwhere(panel == RED)
    assert red_pts.shape == (1, 2), "each panel must contain exactly one red marker"
    r, c = map(int, red_pts[0])
    h, w = panel.shape
    assert h == w and h > 0
    top = r < h // 2
    left = c < w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _center_top_left(outer: int, inner: int) -> int:
    assert 0 < inner <= outer
    return (outer - inner) // 2


def _render_centered_object(panel_size: int, patch: np.ndarray) -> np.ndarray:
    out = np.full((panel_size, panel_size), BG, dtype=int)
    mask = patch == BLUE
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    tight = patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    th, tw = tight.shape
    rr = _center_top_left(panel_size, th)
    cc = _center_top_left(panel_size, tw)
    out[rr : rr + th, cc : cc + tw][tight == BLUE] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-panel grid, the red marker pixel's quadrant encodes a dihedral
    transform to apply to the blue object inside the same panel.

    Quadrant mapping:
    - top-left: unchanged
    - top-right: horizontal flip
    - bottom-left: vertical flip
    - bottom-right: rotate 180 degrees

    The red marker is erased, and the transformed blue object is redrawn centered
    inside its panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == YELLOW)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == YELLOW)]
    row_intervals = _intervals_from_separators(h, sep_rows)
    col_intervals = _intervals_from_separators(w, sep_cols)
    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    panel_size = row_sizes.pop()
    assert col_sizes == {panel_size}, "panels must be equal-sized squares"

    out = np.full_like(g, BG)
    if sep_rows:
        out[sep_rows, :] = YELLOW
    if sep_cols:
        out[:, sep_cols] = YELLOW

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            blue_mask = panel == BLUE
            assert bool(blue_mask.any()), "each panel must contain a blue object"
            assert _connected4(blue_mask), "blue object must be 4-connected"

            instruction = _marker_instruction(panel)
            patch = _tight_patch(panel, BLUE)
            transformed = _transform_patch(patch, instruction)
            centered = _render_centered_object(panel_size, transformed)
            out[r0:r1, c0:c1] = centered

    return out


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    h = random.randint(2, max_h)
    w = random.randint(2, max_w)
    area = random.randint(max(3, (h * w) // 3), min(h * w - 1, h * w - 1))
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)
    if len(cells) != area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not _connected4(mask):
        return None
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return None
    if (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    ):
        return None
    return mask


def _quadrant_cells(panel_size: int, instruction: str) -> list[tuple[int, int]]:
    half = panel_size // 2
    if instruction == "tl":
        rows = range(0, half)
        cols = range(0, half)
    elif instruction == "tr":
        rows = range(0, half)
        cols = range(half, panel_size)
    elif instruction == "bl":
        rows = range(half, panel_size)
        cols = range(0, half)
    else:
        rows = range(half, panel_size)
        cols = range(half, panel_size)
    coords = [(r, c) for r in rows for c in cols]
    random.shuffle(coords)
    return coords


def _place_panel(panel_size: int, instruction: str) -> np.ndarray | None:
    panel = np.full((panel_size, panel_size), BG, dtype=int)
    for _ in range(80):
        shape = _random_shape(max_h=panel_size - 2, max_w=panel_size - 2)
        if shape is None:
            continue
        sh, sw = shape.shape
        top = random.randint(0, panel_size - sh)
        left = random.randint(0, panel_size - sw)
        object_cells = {(top + r, left + c) for r, c in np.argwhere(shape)}

        blocked = set(object_cells)
        for r, c in tuple(object_cells):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < panel_size and 0 <= nc < panel_size:
                        blocked.add((nr, nc))

        candidates = [pos for pos in _quadrant_cells(panel_size, instruction) if pos not in blocked]
        if not candidates:
            continue

        marker = random.choice(candidates)
        for r, c in object_cells:
            panel[r, c] = BLUE
        panel[marker] = RED
        return panel
    return None


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    rows, cols = random.choice([layout for layout in GRID_LAYOUTS if layout[0] * layout[1] == len(panels)])
    panel_size = panels[0].shape[0]
    grid_h = rows * panel_size + (rows - 1)
    grid_w = cols * panel_size + (cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for r in range(1, rows):
        grid[r * panel_size + (r - 1), :] = YELLOW
    for c in range(1, cols):
        grid[:, c * panel_size + (c - 1)] = YELLOW

    idx = 0
    for pr in range(rows):
        for pc in range(cols):
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panels[idx]
            idx += 1
    return grid


def _random_layout() -> tuple[int, int]:
    return random.choice(GRID_LAYOUTS)


def _generate_candidate() -> np.ndarray | None:
    rows, cols = _random_layout()
    count = rows * cols
    panel_size = random.choice(PANEL_SIZES)

    instructions = list(INSTRUCTIONS)
    extra = count - len(instructions)
    if extra > 0:
        instructions.extend(random.choice(INSTRUCTIONS) for _ in range(extra))
    random.shuffle(instructions)

    panels: list[np.ndarray] = []
    for instruction in instructions:
        panel = _place_panel(panel_size, instruction)
        if panel is None:
            return None
        panels.append(panel)

    grid_h = rows * panel_size + (rows - 1)
    grid_w = cols * panel_size + (cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for r in range(1, rows):
        grid[r * panel_size + (r - 1), :] = YELLOW
    for c in range(1, cols):
        grid[:, c * panel_size + (c - 1)] = YELLOW

    idx = 0
    for pr in range(rows):
        for pc in range(cols):
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panels[idx]
            idx += 1

    out = main(grid)
    if np.count_nonzero(grid == RED) != count:
        return None
    if np.any(out == RED):
        return None
    if np.count_nonzero(out == BLUE) != np.count_nonzero(grid == BLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-transform ARC instance")
