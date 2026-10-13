from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


GEN_RETRIES = 400
GRID_HEIGHT_BOUNDS = (14, 30)
GRID_WIDTH_BOUNDS = (12, 30)
PERIOD_BOUNDS = (3, 7)
SPRITE_COUNT_BOUNDS = (2, 4)
MAX_SPRITE_WIDTH = 4
MIN_COLUMN_GAP = 1
LEFT_MARGIN = 2
RIGHT_MARGIN = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _next_floor_row(row: int, phase: int, period: int) -> int:
    return int(phase + ((row - phase) // period + 1) * period)


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each colored sprite onto the hidden periodic floor encoded by two gray dots.

    Input structure:
    - The background is black, and exactly two gray pixels appear in the leftmost
      column; their row distance defines a period `p`.
    - Every row whose index is congruent to the gray-dot rows modulo `p` is an
      implicit floor row, even though those rows are not drawn.
    - All other non-black foreground pixels form 4-connected colored sprites, and
      different sprites do not share any columns.

    Rule:
    - Keep the gray dots fixed and clear every sprite from its original position.
    - If a sprite's height is smaller than `p`, drop it vertically until its
      bottom row is one cell above the first implicit floor row below it; if no
      such floor lies inside the grid, let it fall to the bottom edge.
    - If a sprite's height is at least `p`, it cannot fit between neighboring
      floor rows, so it ignores all floors and falls straight to the bottom.

    Key insight:
    - The gray dots encode an invisible periodic support pattern, and whether a
      sprite stops on that pattern or passes through it depends only on the
      sprite height relative to the hidden floor period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_coords = np.argwhere(g == GRAY)
    assert marker_coords.shape == (2, 2)
    assert np.all(marker_coords[:, 1] == 0)
    marker_rows = sorted(int(r) for r in marker_coords[:, 0])
    period = marker_rows[1] - marker_rows[0]
    assert period >= 1
    phase = marker_rows[0] % period if period > 0 else 0

    sprite_mask = (g != BLACK) & (g != GRAY)
    components = _find_components(sprite_mask)
    assert components

    used_columns: set[int] = set()
    for coords in components:
        cols = {int(c) for c in coords[:, 1]}
        assert used_columns.isdisjoint(cols)
        used_columns.update(cols)

    out = np.full((h, w), BLACK, dtype=int)
    out[marker_rows[0], 0] = GRAY
    out[marker_rows[1], 0] = GRAY

    occupied = np.zeros((h, w), dtype=bool)
    for coords in sorted(components, key=lambda pts: _bbox(pts)[1]):
        r0, _c0, r1, _c1 = _bbox(coords)
        height = r1 - r0 + 1

        if height >= period:
            shift = h - 1 - r1
        else:
            next_floor = _next_floor_row(r1, phase, period)
            target_bottom = h - 1 if next_floor >= h else next_floor - 1
            shift = target_bottom - r1

        assert shift >= 0
        moved = coords.copy()
        moved[:, 0] += shift
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < w))
        assert not np.any(occupied[moved[:, 0], moved[:, 1]])

        occupied[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = g[coords[:, 0], coords[:, 1]]

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _random_connected_mask(height: int, max_width: int) -> np.ndarray:
    width = random.randint(1, max_width)
    col = random.randrange(width)
    cells = {(0, col)}

    for r in range(1, height):
        col = max(0, min(width - 1, col + random.choice((-1, 0, 1))))
        cells.add((r, col))
        if width > 1 and random.random() < 0.35:
            side = col + random.choice((-1, 1))
            if 0 <= side < width:
                cells.add((r, side))

    max_cells = height * width
    extra_cap = min(max_cells - len(cells), height + width)
    target_cells = len(cells) + random.randint(0, max(0, extra_cap))

    while len(cells) < target_cells:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _short_sprite_top(height: int, floors: list[int]) -> int | None:
    gap_idx = random.randint(1, len(floors) - 1)
    upper_floor = floors[gap_idx - 1]
    lower_floor = floors[gap_idx]
    lo = upper_floor + 1
    hi = lower_floor - height
    if hi < lo:
        return None
    moving_hi = lower_floor - height - 1
    return random.randint(lo, moving_hi) if moving_hi >= lo else hi


def _column_starts(widths: list[int], grid_width: int) -> list[int]:
    base = LEFT_MARGIN + RIGHT_MARGIN + sum(widths) + MIN_COLUMN_GAP * (len(widths) - 1)
    extra = grid_width - base
    assert extra >= 0
    slack = _random_composition(extra, len(widths) + 1)

    starts: list[int] = []
    col = LEFT_MARGIN + slack[0]
    for idx, width in enumerate(widths):
        starts.append(col)
        col += width
        if idx < len(widths) - 1:
            col += MIN_COLUMN_GAP + slack[idx + 1]
    return starts


def _build_sprite_specs(height: int, period: int, phase: int) -> list[dict[str, object]] | None:
    floors = list(range(phase, height, period))
    if len(floors) < 2:
        return None

    count = random.randint(*SPRITE_COUNT_BOUNDS)
    branches = ["short", "tall"] + [random.choice(("short", "tall")) for _ in range(count - 2)]
    random.shuffle(branches)
    specs: list[dict[str, object]] = []

    for branch in branches:
        if branch == "short":
            short_choices = list(range(1, period))
            if period > 3 and random.random() < 0.75:
                short_choices = list(range(1, period - 1)) or [1]
            sprite_h = random.choice(short_choices)
            top = _short_sprite_top(sprite_h, floors)
            if top is None:
                return None
        else:
            max_tall_h = min(height - 2, period + 6)
            if max_tall_h < period:
                return None
            sprite_h = random.randint(period, max_tall_h)
            max_top = height - sprite_h - 2
            if max_top < 0:
                return None
            top = random.randint(0, max_top)

        mask = _random_connected_mask(sprite_h, MAX_SPRITE_WIDTH)
        specs.append({"branch": branch, "top": top, "mask": mask})

    return specs


def _render_grid(
    grid_height: int,
    grid_width: int,
    phase: int,
    period: int,
    colors: list[int],
    specs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((grid_height, grid_width), BLACK, dtype=int)
    grid[phase, 0] = GRAY
    grid[phase + period, 0] = GRAY

    starts = _column_starts([int(spec["mask"].shape[1]) for spec in specs], grid_width)
    for color, start, spec in zip(colors, starts, specs):
        mask = spec["mask"]
        top = int(spec["top"])
        rows, cols = np.argwhere(mask).T
        grid[rows + top, cols + start] = color

    return grid


def _valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    return not np.array_equal(grid, out)


def _generate_attempt() -> np.ndarray | None:
    period = random.randint(*PERIOD_BOUNDS)
    phase = random.randint(0, period - 1)
    min_height = max(GRID_HEIGHT_BOUNDS[0], phase + 2 * period + 2)
    if min_height > GRID_HEIGHT_BOUNDS[1]:
        return None
    grid_height = random.randint(min_height, GRID_HEIGHT_BOUNDS[1])

    specs = _build_sprite_specs(grid_height, period, phase)
    if specs is None:
        return None

    widths = [int(spec["mask"].shape[1]) for spec in specs]
    min_width = max(
        GRID_WIDTH_BOUNDS[0],
        LEFT_MARGIN + RIGHT_MARGIN + sum(widths) + MIN_COLUMN_GAP * (len(widths) - 1),
    )
    if min_width > GRID_WIDTH_BOUNDS[1]:
        return None
    grid_width = random.randint(min_width, GRID_WIDTH_BOUNDS[1])

    colors = random.sample(list(SPRITE_COLORS), len(specs))
    grid = _render_grid(grid_height, grid_width, phase, period, colors, specs)
    return grid if _valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
