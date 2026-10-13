from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
LEFT_COLOR = 1
SEPARATOR_COLOR = 2
MIDDLE_COLOR = 3
SOURCE_COLOR_POOL = (4, 6, 7, 8, 9)
PANEL_SIZE = 5
MAX_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else {int(v) for v in exclude}
    return sorted(int(v) for v in np.unique(grid) if int(v) not in excluded)


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = _mask_coords(mask)
    if not coords:
        return False
    shape = mask.shape
    seen = {coords[0]}
    queue = deque([coords[0]])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, shape):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return len(seen) == len(coords)


def _grow_connected_mask(
    target_size: int,
    *,
    shape: tuple[int, int],
    seed_mask: np.ndarray | None = None,
    forbidden: np.ndarray | None = None,
) -> np.ndarray | None:
    mask = np.zeros(shape, dtype=bool) if seed_mask is None else seed_mask.copy()
    blocked = np.zeros(shape, dtype=bool) if forbidden is None else forbidden
    if np.any(mask & blocked):
        return None

    if not mask.any():
        start_choices = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if not blocked[r, c]]
        if not start_choices:
            return None
        mask[random.choice(start_choices)] = True

    while int(mask.sum()) < int(target_size):
        frontier = set()
        for r, c in _mask_coords(mask):
            for nr, nc in _neighbors4(r, c, shape):
                if mask[nr, nc] or blocked[nr, nc]:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return None
        mask[random.choice(sorted(frontier))] = True

    return mask


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n = int(g.shape[0])
    h, w = map(int, g.shape)
    assert n >= 3 and w == 3 * n + 1, "expected n x (3n+1) layout: left | sep | middle | source"
    assert n * n <= 30, "output must stay within ARC size limits"

    bg = _get_bg(g)
    sep_col = n
    sep_color = int(g[0, sep_col])
    assert sep_color != bg, "separator must be non-background"
    assert np.all(g[:, sep_col] == sep_color), "separator must be a solid column"

    left = g[:, :n]
    middle = g[:, n + 1 : 2 * n + 1]
    source = g[:, 2 * n + 1 :]
    assert left.shape == middle.shape == source.shape == (n, n)

    left_colors = _colors_present(left, exclude={bg})
    middle_colors = _colors_present(middle, exclude={bg})
    assert len(left_colors) == 1 and len(middle_colors) == 1, "each logic panel must have one foreground color"
    assert left_colors[0] != middle_colors[0], "logic panel colors must differ"
    assert sep_color not in left_colors and sep_color not in middle_colors, "separator color must be unique"

    return g, left, middle, source, bg, sep_color, n


def main(grid: np.ndarray) -> np.ndarray:
    """
    Derive a stamp template from the two logic panels' shared empty cells, then stamp it into source-indexed output blocks.

    Input structure:
    - The grid is laid out as three equal square regions: a left logic panel, one solid separator column, a middle logic panel, and a right source panel.
    - The two logic panels share the same background and each contains exactly one connected foreground shape in its own color.
    - The source panel is a same-size square on the right containing isolated non-background pixels on a black background.

    Rule:
    - Compare the left and middle logic panels cell by cell and mark every coordinate where both are background; this shared-empty mask is the stamp template.
    - Create an output canvas of size (N^2) x (N^2), where N is the panel size, initialized to background.
    - For each non-background source pixel at source coordinate (r, c), recolor the template mask to that pixel's color and stamp it into output block (r, c).
    - Leave every output cell untouched if no stamped template covers it.

    Key insight:
    - The template is not shown directly: it must first be inferred as the intersection of the two panels' empty space, and only then expanded blockwise according to the source pixels.
    """
    g, left, middle, source, bg, _sep_color, n = _parse_input(grid)
    del g

    template_mask = (left == bg) & (middle == bg)
    assert np.any(template_mask), "derived template must be non-empty"

    out = np.full((n * n, n * n), bg, dtype=int)
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            if color == bg:
                continue
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            block[template_mask] = color
    return out


def _generate_panel_pair() -> tuple[np.ndarray, np.ndarray] | None:
    shape = (PANEL_SIZE, PANEL_SIZE)
    left_size = random.randint(4, 8)
    middle_size = random.randint(4, 8)
    shared_size = random.randint(1, min(3, left_size - 1, middle_size - 1))

    shared = _grow_connected_mask(shared_size, shape=shape)
    if shared is None:
        return None

    left_mask = _grow_connected_mask(left_size, shape=shape, seed_mask=shared)
    if left_mask is None:
        return None

    left_only = left_mask & ~shared
    middle_mask = _grow_connected_mask(
        middle_size,
        shape=shape,
        seed_mask=shared,
        forbidden=left_only,
    )
    if middle_mask is None:
        return None

    overlap = left_mask & middle_mask
    template_mask = ~(left_mask | middle_mask)
    union_area = int((left_mask | middle_mask).sum())
    template_area = int(template_mask.sum())

    if int(overlap.sum()) < 1:
        return None
    if int((left_mask & ~middle_mask).sum()) < 1 or int((middle_mask & ~left_mask).sum()) < 1:
        return None
    if not _is_connected(left_mask) or not _is_connected(middle_mask):
        return None
    if template_area < 3 or not _is_connected(template_mask):
        return None
    if not (8 <= union_area <= 15 and 10 <= template_area <= 17):
        return None

    left = np.full(shape, BG, dtype=int)
    middle = np.full(shape, BG, dtype=int)
    left[left_mask] = LEFT_COLOR
    middle[middle_mask] = MIDDLE_COLOR
    return left, middle


def _generate_source() -> np.ndarray | None:
    source = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    target_count = random.randint(2, 4)
    positions: list[tuple[int, int]] = []
    cells = [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE)]
    random.shuffle(cells)
    for r, c in cells:
        if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in positions):
            positions.append((r, c))
        if len(positions) == target_count:
            break
    if len(positions) != target_count:
        return None

    for color, (r, c) in zip(random.sample(SOURCE_COLOR_POOL, target_count), positions):
        source[r, c] = color
    return source


def _assemble_input(left: np.ndarray, middle: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [left, np.full((PANEL_SIZE, 1), SEPARATOR_COLOR, dtype=int), middle, source],
        axis=1,
    )


def _validate_input(grid: np.ndarray) -> bool:
    try:
        _g, left, middle, source, bg, sep_color, n = _parse_input(grid)
    except AssertionError:
        return False

    left_mask = left != bg
    middle_mask = middle != bg
    template_mask = (~left_mask) & (~middle_mask)
    if n != PANEL_SIZE:
        return False
    if not (4 <= int(left_mask.sum()) <= 8 and 4 <= int(middle_mask.sum()) <= 8):
        return False
    if not _is_connected(left_mask) or not _is_connected(middle_mask):
        return False
    if int((left_mask & middle_mask).sum()) < 1:
        return False
    if int((left_mask & ~middle_mask).sum()) < 1 or int((middle_mask & ~left_mask).sum()) < 1:
        return False
    if int(template_mask.sum()) < 3 or not _is_connected(template_mask):
        return False

    source_positions = _mask_coords(source != bg)
    if not (2 <= len(source_positions) <= 4):
        return False
    source_colors = [int(source[r, c]) for r, c in source_positions]
    if len(set(source_colors)) != len(source_colors):
        return False
    if set(source_colors) & {bg, LEFT_COLOR, MIDDLE_COLOR, sep_color}:
        return False
    if any(abs(r1 - r2) + abs(c1 - c2) == 1 for (r1, c1), (r2, c2) in combinations(source_positions, 2)):
        return False

    expected = np.full((n * n, n * n), bg, dtype=int)
    for r, c in source_positions:
        block = expected[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[template_mask] = int(source[r, c])
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        panels = _generate_panel_pair()
        source = _generate_source()
        if panels is None or source is None:
            continue
        left, middle = panels
        grid = _assemble_input(left, middle, source)
        if _validate_input(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
