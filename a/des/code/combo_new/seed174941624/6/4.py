import random
from collections import deque

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
FRAME_COUNT_RANGE = (2, 5)
CORE_SIZE_RANGE = (1, 6)
INNER_FRAME_SIZE_RANGE = (3, 7)
GRID_MARGIN_RANGE = (0, 4)
GENERATION_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_color(grid: np.ndarray, coords: list[tuple[int, int]]) -> int:
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1, "each layer must be monochrome"
    return next(iter(colors))


def _is_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    expected = 2 * (height + width) - 4
    if len(coords) != expected:
        return False
    cells = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in cells):
                return False
    return True


def _is_solid_rect(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _ordered_layers(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    layers: list[tuple[tuple[int, int, int, int], int]] = []
    for coords in _components(grid != BLACK):
        color = _component_color(grid, coords)
        assert _is_frame(coords) or _is_solid_rect(coords), "expected only frames and an optional solid core"
        layers.append((_bbox(coords), color))

    assert layers, "expected at least one non-black layer"
    layers.sort(
        key=lambda item: (
            (item[0][2] - item[0][0] + 1) * (item[0][3] - item[0][1] + 1),
            -(item[0][0] + item[0][1]),
        ),
        reverse=True,
    )

    for i in range(len(layers) - 1):
        outer = layers[i][0]
        inner = layers[i + 1][0]
        assert inner[0] == outer[0] + 2
        assert inner[1] == outer[1] + 2
        assert inner[2] == outer[2] - 2
        assert inner[3] == outer[3] - 2

    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse a nested stack of colored rectangular layers into horizontal color bands.

    Input structure:
    - The background is black.
    - Non-black objects are one-cell-thick axis-aligned rectangular frames nested inside each other.
    - Each deeper layer is inset by exactly 2 cells, leaving a 1-cell black gap between consecutive layers.
    - The center may contain a solid monochrome rectangular core, also inset by 2 cells, or may remain black.

    Rule:
    - Identify every nested layer from the largest outer frame to the smallest inner layer.
    - Treat a solid core, when present, as the final innermost layer after the frames.
    - Let the ordered layer colors be C1, C2, ..., Cn from outside to inside; create an n x n output grid.
    - Fill the bottom row with C1, the next row up with C2, and continue until the top row is Cn.

    Key insight:
    - The colors are encoded by containment depth rather than adjacency: black gaps separate the layers, so the solver must recover the nesting order from the concentric rectangles and then compress that order into a stacked stripe summary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layers = _ordered_layers(g)
    colors = [color for _bbox_, color in layers]
    n = len(colors)
    out = np.empty((n, n), dtype=int)
    for idx, color in enumerate(colors):
        out[n - 1 - idx, :] = color
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _split_margin(rng: random.Random, slack: int) -> tuple[int, int]:
    total = rng.randint(0, min(2 * GRID_MARGIN_RANGE[1], slack))
    first = rng.randint(0, total)
    return first, total - first


def _gen_attempt(rng: random.Random) -> np.ndarray:
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    has_core = rng.choice([False, True])

    if has_core:
        core_h = rng.randint(*CORE_SIZE_RANGE)
        core_w = rng.randint(*CORE_SIZE_RANGE)
        outer_h = core_h + 4 * frame_count
        outer_w = core_w + 4 * frame_count
        layer_count = frame_count + 1
    else:
        inner_h = rng.randint(*INNER_FRAME_SIZE_RANGE)
        inner_w = rng.randint(*INNER_FRAME_SIZE_RANGE)
        outer_h = inner_h + 4 * (frame_count - 1)
        outer_w = inner_w + 4 * (frame_count - 1)
        layer_count = frame_count

    top_margin, bottom_margin = _split_margin(rng, 30 - outer_h)
    left_margin, right_margin = _split_margin(rng, 30 - outer_w)
    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin

    assert grid_h <= 30 and grid_w <= 30, "generator parameters must stay within ARC limits"

    top = top_margin
    left = left_margin
    grid = np.zeros((grid_h, grid_w), dtype=int)
    colors = rng.sample(COLOR_POOL, layer_count)

    for i in range(frame_count):
        bbox = (top + 2 * i, left + 2 * i, top + outer_h - 1 - 2 * i, left + outer_w - 1 - 2 * i)
        _draw_frame(grid, bbox, colors[i])

    if has_core:
        r0 = top + 2 * frame_count
        c0 = left + 2 * frame_count
        r1 = top + outer_h - 1 - 2 * frame_count
        c1 = left + outer_w - 1 - 2 * frame_count
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[-1]

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
