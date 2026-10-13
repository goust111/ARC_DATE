from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, size: int) -> Iterable[tuple[int, int]]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            yield rr, cc


def _has_adjacent_pair(mask: np.ndarray) -> bool:
    size = int(mask.shape[0])
    for r, c in np.argwhere(mask):
        for rr, cc in _neighbors4(int(r), int(c), size):
            if bool(mask[rr, cc]):
                return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    size = int(mask.shape[0])
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, size):
            if bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _flip_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [
        mask.copy(),
        np.fliplr(mask).copy(),
        np.flipud(mask).copy(),
        np.flipud(np.fliplr(mask)).copy(),
    ]


def _is_template_panel(panel: np.ndarray, bg: int) -> bool:
    fg = panel != bg
    if int(fg.sum()) < 2:
        return False
    colors = np.unique(panel[fg])
    if len(colors) != 1:
        return False
    if not _has_adjacent_pair(fg):
        return False
    variants = {variant.tobytes() for variant in _flip_variants(fg)}
    return len(variants) == 4


def _split_panels(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    if w == 2 * h:
        return h, grid[:, :h].copy(), grid[:, h:].copy()
    if h == 2 * w:
        return w, grid[:w, :].copy(), grid[w:, :].copy()
    raise AssertionError("expected two equal N×N panels laid out side by side or stacked")


def _locate_panels(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    size, panel_a, panel_b = _split_panels(grid)
    candidates: list[tuple[tuple[int, int, int, int], int, np.ndarray, np.ndarray]] = []
    for bg in map(int, np.unique(grid)):
        for template_idx, (template, source) in enumerate(((panel_a, panel_b), (panel_b, panel_a))):
            if not _is_template_panel(template, bg):
                continue
            source_mask = source != bg
            if not bool(source_mask.any()):
                continue
            source_colors = np.unique(source[source_mask])
            score = (
                -int(len(source_colors) >= 2),
                -int(bg in set(map(int, np.unique(source))) and bg in set(map(int, np.unique(template)))),
                int((template != bg).sum()),
                int(source_mask.sum()),
                -int(len(source_colors)),
                template_idx,
            )
            candidates.append((score, bg, source, template))
    assert candidates, "failed to identify source and template panels"
    candidates.sort()
    _score, bg, source, template = candidates[0]
    return size, bg, source, template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp center-facing mirrored copies of the template into the output block for each source pixel.

    Input structure
    - The input contains two equal N×N square panels arranged side by side or stacked.
    - One panel is a sparse source grid of colored foreground pixels; the other is a single-color asymmetric template mask on the same background.
    - N is even, so the source panel has four unambiguous quadrants around its center.

    Rule
    - Build an N²×N² background output divided into N×N blocks of size N×N.
    - For each non-background source pixel at (r, c), recolor the full template mask to that pixel's color.
    - If c is in the right half of the source panel, flip the template left-right; if r is in the lower half, flip it top-bottom; then place it into output block (r, c).

    Key insight
    - The output is not made from one rigid stamp orientation: each source pixel chooses one of four mirrored template variants according to its quadrant, so all copies face toward the source panel's center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    size, bg, source, template_panel = _locate_panels(g)
    assert size % 2 == 0, "expected an even panel size"
    assert size * size <= 30, "output would exceed ARC size limits"

    template_mask = template_panel != bg
    out = np.full((size * size, size * size), int(bg), dtype=int)
    half = size // 2

    for r in range(size):
        for c in range(size):
            color = int(source[r, c])
            if color == bg:
                continue
            patch = np.full((size, size), int(bg), dtype=int)
            patch[template_mask] = color
            if c >= half:
                patch = np.fliplr(patch)
            if r >= half:
                patch = np.flipud(patch)
            out[r * size : (r + 1) * size, c * size : (c + 1) * size] = patch

    return out


PANEL_SIZES = (2, 4)
PANEL_SIZE_WEIGHTS = (1, 4)
BACKGROUND = 0
INPUT_COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _valid_template_mask(mask: np.ndarray) -> bool:
    filled = int(mask.sum())
    size = int(mask.shape[0])
    if not (2 <= filled < size * size):
        return False
    if not _is_connected(mask):
        return False
    if not _has_adjacent_pair(mask):
        return False
    variants = {variant.tobytes() for variant in _flip_variants(mask)}
    return len(variants) == 4


def _generate_template_mask(size: int) -> np.ndarray:
    if size == 2:
        mask = np.ones((2, 2), dtype=bool)
        missing = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
        mask[missing] = False
        return mask

    for _ in range(200):
        target_area = random.randint(4, 7)
        mask = np.zeros((size, size), dtype=bool)
        start = (random.randrange(size), random.randrange(size))
        mask[start] = True
        while int(mask.sum()) < target_area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for rr, cc in _neighbors4(int(r), int(c), size):
                    if not bool(mask[rr, cc]):
                        frontier.add((rr, cc))
            if not frontier:
                break
            rr, cc = random.choice(tuple(frontier))
            mask[rr, cc] = True
        if _valid_template_mask(mask):
            return mask
    raise RuntimeError("failed to generate a valid asymmetric template mask")


def _quadrant_cells(size: int) -> list[list[tuple[int, int]]]:
    half = size // 2
    return [
        [(r, c) for r in range(0, half) for c in range(0, half)],
        [(r, c) for r in range(0, half) for c in range(half, size)],
        [(r, c) for r in range(half, size) for c in range(0, half)],
        [(r, c) for r in range(half, size) for c in range(half, size)],
    ]


def _generate_source_positions(size: int) -> list[tuple[int, int]]:
    quadrants = _quadrant_cells(size)
    positions = [random.choice(cells) for cells in quadrants]
    if size == 2:
        return positions

    all_cells = [(r, c) for r in range(size) for c in range(size)]
    remaining = [cell for cell in all_cells if cell not in positions]
    random.shuffle(remaining)
    extra_count = random.randint(0, 3)
    positions.extend(remaining[:extra_count])
    random.shuffle(positions)
    return positions


def _assemble_input(source: np.ndarray, template: np.ndarray, horizontal: bool, source_first: bool) -> np.ndarray:
    first, second = (source, template) if source_first else (template, source)
    if horizontal:
        return np.hstack((first, second))
    return np.vstack((first, second))


def _generate_source_colors(count: int, template_color: int) -> list[int]:
    palette = [color for color in INPUT_COLORS if color != template_color]
    return random.sample(palette, count)


def _gen_attempt() -> np.ndarray | None:
    size = random.choices(PANEL_SIZES, weights=PANEL_SIZE_WEIGHTS, k=1)[0]
    template_mask = _generate_template_mask(size)
    template_color = random.choice(INPUT_COLORS)

    source_positions = _generate_source_positions(size)
    source_colors = _generate_source_colors(len(source_positions), template_color)
    source_panel = np.full((size, size), BACKGROUND, dtype=int)
    for (r, c), color in zip(source_positions, source_colors):
        source_panel[r, c] = color

    template_panel = np.full((size, size), BACKGROUND, dtype=int)
    template_panel[template_mask] = template_color

    horizontal = random.choice((False, True))
    source_first = random.choice((False, True))
    grid = _assemble_input(source_panel, template_panel, horizontal, source_first)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if grid.shape not in ((size, 2 * size), (2 * size, size)):
        return None
    if out.shape != (size * size, size * size):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    recovered_size, bg, recovered_source, recovered_template = _locate_panels(grid)
    if recovered_size != size or bg != BACKGROUND:
        return None
    if not np.array_equal(recovered_source, source_panel):
        return None
    if not np.array_equal(recovered_template, template_panel):
        return None

    if size == 4 and int((source_panel != BACKGROUND).sum()) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
