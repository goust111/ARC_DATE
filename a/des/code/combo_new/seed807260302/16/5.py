import random
from collections import Counter

import numpy as np


BG = 0
CROSS = 2
FRAME = 5
SEED_PALETTE = (1, 3, 4, 6, 7, 8, 9)
SIZE_CHOICES = tuple(range(13, 30, 2))
MAX_RETRIES = 300


def _mode(values) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _border_positions(height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(width):
        cells.append((0, c))
        cells.append((height - 1, c))
    for r in range(1, height - 1):
        cells.append((r, 0))
        cells.append((r, width - 1))
    return cells


def _detect_structure(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    height, width = grid.shape
    assert height % 2 == 1 and width % 2 == 1, "expected odd-sized grid"

    mid_r = height // 2
    mid_c = width // 2
    cross_color = int(grid[mid_r, mid_c])

    assert np.all(grid[mid_r, :] == cross_color), "expected a full horizontal center cross"
    assert np.all(grid[:, mid_c] == cross_color), "expected a full vertical center cross"

    ignore = {
        (0, mid_c),
        (height - 1, mid_c),
        (mid_r, 0),
        (mid_r, width - 1),
    }
    frame_samples = [int(grid[r, c]) for r, c in _border_positions(height, width) if (r, c) not in ignore]
    frame_color = _mode(frame_samples)

    for r, c in _border_positions(height, width):
        val = int(grid[r, c])
        if r == mid_r or c == mid_c:
            assert val in {frame_color, cross_color}, "cross may only intersect the border at the midpoints"
        else:
            assert val == frame_color, "expected a uniform frame away from the cross"

    interior = grid[1:-1, 1:-1].copy()
    interior[mid_r - 1, :] = cross_color
    interior[:, mid_c - 1] = cross_color
    bg_samples = [
        int(interior[r, c])
        for r in range(interior.shape[0])
        for c in range(interior.shape[1])
        if r != mid_r - 1 and c != mid_c - 1
    ]
    bg_color = _mode(bg_samples)

    assert len({bg_color, cross_color, frame_color}) == 3, "background, cross, and frame must be distinct"
    return mid_r, mid_c, frame_color, cross_color, bg_color


def _grow_branch(out: np.ndarray, r: int, c: int, color: int, mid_r: int, mid_c: int) -> None:
    height, width = out.shape
    end_c = 1 if c < mid_c else width - 2
    end_r = 1 if r < mid_r else height - 2

    c0, c1 = sorted((c, end_c))
    r0, r1 = sorted((r, end_r))
    out[r, c0 : c1 + 1] = color
    out[r0 : r1 + 1, end_c] = color


def _apply_rule(grid: np.ndarray, seeds: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    mid_r = out.shape[0] // 2
    mid_c = out.shape[1] // 2
    for r, c, color in seeds:
        _grow_branch(out, int(r), int(c), int(color), mid_r, mid_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each quadrant seed into an outward L-shaped branch anchored to that quadrant's frame side.

    Input structure:
    - The background is black.
    - A one-pixel-wide red cross spans the central row and central column, splitting the grid into four quadrants.
    - A grey one-pixel frame surrounds the grid.
    - Each quadrant contains isolated colored seed pixels; seeds in one quadrant share a color, while different quadrants may use different colors.

    Rule:
    - Keep the red cross and grey frame unchanged.
    - For each seed, determine its quadrant from its position relative to the center cross.
    - Draw a horizontal segment from the seed toward that quadrant's outer frame side, stopping one cell before the grey border.
    - From the horizontal segment's outer endpoint, draw a vertical segment toward the same quadrant's outer frame side, also stopping one cell before the grey border.
    - Paint both segments with the seed color and keep the seed pixel itself.

    Key insight:
    - The center cross encodes four independent growth directions: each seed grows away from the cross toward its quadrant corner, and the two leg lengths come directly from the seed's distances to the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mid_r, mid_c, frame_color, cross_color, bg_color = _detect_structure(g)
    seed_mask = np.ones(g.shape, dtype=bool)
    seed_mask[0, :] = False
    seed_mask[-1, :] = False
    seed_mask[:, 0] = False
    seed_mask[:, -1] = False
    seed_mask[mid_r, :] = False
    seed_mask[:, mid_c] = False
    seed_mask &= g != bg_color
    seed_mask &= g != frame_color
    seed_mask &= g != cross_color

    out = g.copy()
    for r, c in np.argwhere(seed_mask):
        _grow_branch(out, int(r), int(c), int(g[r, c]), mid_r, mid_c)
    return out


def _candidate_values(start: int, stop: int) -> list[int]:
    return list(range(start, stop))


def _search_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    shuffled = cells[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    def backtrack(index: int) -> bool:
        if len(chosen) == count:
            return True
        remaining_need = count - len(chosen)
        if len(shuffled) - index < remaining_need:
            return False
        for i in range(index, len(shuffled)):
            r, c = shuffled[i]
            if r in used_rows or c in used_cols:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            if backtrack(i + 1):
                return True
            chosen.pop()
            used_rows.remove(r)
            used_cols.remove(c)
        return False

    return chosen[:] if backtrack(0) else None


def _quadrant_ranges(height: int, width: int, quadrant: str) -> tuple[list[int], list[int]]:
    mid_r = height // 2
    mid_c = width // 2
    top_rows = _candidate_values(2, mid_r - 1)
    bottom_rows = _candidate_values(mid_r + 2, height - 2)
    left_cols = _candidate_values(2, mid_c - 1)
    right_cols = _candidate_values(mid_c + 2, width - 2)

    if quadrant == "ul":
        return top_rows, left_cols
    if quadrant == "ur":
        return top_rows, right_cols
    if quadrant == "ll":
        return bottom_rows, left_cols
    return bottom_rows, right_cols


def _choose_quadrant_positions(height: int, width: int, quadrant: str, max_count: int) -> list[tuple[int, int]] | None:
    rows, cols = _quadrant_ranges(height, width, quadrant)
    cells = [(r, c) for r in rows for c in cols]
    if not cells:
        return None

    options = []
    for count in range(1, max_count + 1):
        placement = _search_positions(cells, count)
        if placement is not None:
            options.append(placement)
    if not options:
        return None
    return random.choice(options)


def _render_input(height: int, width: int, seeds: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    mid_r = height // 2
    mid_c = width // 2
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME
    grid[mid_r, :] = CROSS
    grid[:, mid_c] = CROSS
    for r, c, color in seeds:
        grid[r, c] = color
    return grid


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.choice(SIZE_CHOICES)
    width = random.choice(SIZE_CHOICES)
    max_count = 2 if min(height, width) < 17 else 3
    quadrant_colors = random.sample(SEED_PALETTE, 4)
    quadrant_order = ("ul", "ur", "ll", "lr")

    seeds: list[tuple[int, int, int]] = []
    for quadrant, color in zip(quadrant_order, quadrant_colors):
        positions = _choose_quadrant_positions(height, width, quadrant, max_count)
        if positions is None:
            return None
        seeds.extend((r, c, color) for r, c in positions)

    inp = _render_input(height, width, seeds)
    expected = _apply_rule(inp, seeds)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
