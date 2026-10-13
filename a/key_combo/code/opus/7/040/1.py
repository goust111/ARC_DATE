from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

TEMPLATE_COLORS = (BLUE, RED)
CONTROL_COLORS = (GREEN, YELLOW, ORANGE, TEAL, MAROON)
COMPONENT_COUNT_OPTIONS = (2, 3, 4, 5)
GENERATION_ATTEMPTS = 500

BASE_COMPONENT_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def is_hole_free(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    padded = np.pad(bg, 1, constant_values=True)
    h, w = padded.shape
    seen = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return not np.any(padded & ~seen)


def normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _ in shape)
    return normalize_shape(tuple((c, max_r - r) for r, c in shape))


def flip_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in shape)
    return normalize_shape(tuple((r, max_c - c) for r, c in shape))


def shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    cur = normalize_shape(shape)
    for _ in range(4):
        variants.add(cur)
        variants.add(flip_shape(cur))
        cur = rotate_shape(cur)
    return list(variants)


COMPONENT_SHAPES = tuple(
    variant
    for base in BASE_COMPONENT_SHAPES
    for variant in shape_variants(base)
)


def _paint_stamp(out: np.ndarray, top: int, left: int, template_mask: np.ndarray, color: int) -> None:
    h, w = template_mask.shape
    view = out[top : top + h, left : left + w]
    view[template_mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the cropped upper template into output row groups whose horizontal spacing is set by control-component counts.

    Input structure:
    - The background is black.
    - Exactly one full-width grey separator row splits the grid into an upper template region and a lower control region.
    - The upper region contains one compact blue/red template shape on black background; the shape is 4-connected and hole-free.
    - The lower region contains 2-4 non-background control colors, each split into a distinct number of 4-connected monochrome components.

    Rule:
    - Find the grey separator row, crop the upper template to its tight bounding box, and treat every non-background template cell as stamp foreground.
    - In the lower region, count the 4-connected components of each non-background color; call that count K for the color.
    - Build an output whose height is one template-height row group per control color and whose width is template_width times the largest K.
    - Sort control colors by increasing K, then in each color's row group stamp recolored copies of the template at coarse positions 0, K, 2K, ... until the output width is exhausted.

    Key insight:
    - The lower colored fragments are not objects to preserve; each color's fragmentation count encodes the repeat period for recoloring and placing the upper template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(sep_rows) == 1, "expected exactly one full-width grey separator row"
    sep = int(sep_rows[0])
    assert 0 < sep < g.shape[0] - 1, "separator must split non-empty upper and lower regions"

    bg = get_bg(g)
    top = g[:sep]
    control = g[sep + 1 :]

    template_mask = top != bg
    assert template_mask.any(), "upper region must contain a template"
    template = crop_bbox(top, bbox_from_mask(template_mask))
    template_mask = template != bg
    assert template.shape[0] in (3, 4) and template.shape[1] in (3, 4), "template must crop to 3x3..4x4"
    assert set(map(int, np.unique(template[template_mask]))) <= set(TEMPLATE_COLORS), "template colors must be blue/red"
    assert len(components_mask(template_mask)) == 1, "template foreground must be 4-connected"
    assert is_hole_free(template_mask), "template must be hole-free"

    items: list[tuple[int, int]] = []
    for color in sorted(set(map(int, np.unique(control))) - {bg}):
        comps = components_mask(control == color)
        assert 2 <= len(comps) <= 5, "each control color must have 2..5 components"
        assert all(1 <= comp.shape[0] <= 4 for comp in comps), "each control component must have 1..4 cells"
        items.append((color, len(comps)))
    assert 2 <= len(items) <= 4, "expected 2..4 control colors"

    counts = [count for _, count in items]
    assert len(set(counts)) == len(counts), "control colors must have distinct component counts"
    items.sort(key=lambda item: item[1])

    template_h, template_w = template.shape
    max_count = max(counts)
    out = np.full((len(items) * template_h, max_count * template_w), bg, dtype=int)
    for group_idx, (color, period) in enumerate(items):
        row0 = group_idx * template_h
        for coarse_col in range(0, max_count, period):
            _paint_stamp(out, row0, coarse_col * template_w, template_mask, color)
    return out


def _random_polyomino(height: int, width: int, area: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    while len(cells) < area:
        frontier.clear()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _generate_template() -> np.ndarray:
    for _ in range(200):
        height = random.randint(3, 4)
        width = random.randint(3, 4)
        max_area = height * width - 1
        min_area = max(5, (height * width + 1) // 2)
        mask = _random_polyomino(height, width, random.randint(min_area, max_area))
        if bbox_from_mask(mask) != (0, 0, height - 1, width - 1):
            continue
        if len(components_mask(mask)) != 1 or not is_hole_free(mask):
            continue
        template = np.zeros((height, width), dtype=int)
        coords = [tuple(coord) for coord in np.argwhere(mask)]
        random.shuffle(coords)
        split = random.randint(1, len(coords) - 1)
        blue_cells = set(coords[:split])
        for r, c in coords:
            template[r, c] = BLUE if (r, c) in blue_cells else RED
        if BLUE in template and RED in template:
            return template
    raise RuntimeError("failed to generate a valid template")


def _can_place_component(occupied: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    h, w = occupied.shape
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _place_component(grid: np.ndarray, occupied: np.ndarray, shape: tuple[tuple[int, int], ...], color: int, top: int, left: int) -> None:
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        grid[r, c] = int(color)
        occupied[r, c] = True


def _generate_control_region(height: int, width: int, color_counts: list[tuple[int, int]]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    jobs: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for color, count in color_counts:
        for _ in range(count):
            jobs.append((color, random.choice(COMPONENT_SHAPES)))
    jobs.sort(key=lambda item: len(item[1]), reverse=True)
    random.shuffle(jobs)
    jobs.sort(key=lambda item: len(item[1]), reverse=True)

    for color, shape in jobs:
        variants = shape_variants(shape)
        random.shuffle(variants)
        placed = False
        for variant in variants:
            max_r = max(r for r, _ in variant)
            max_c = max(c for _, c in variant)
            placements = [
                (top, left)
                for top in range(height - max_r)
                for left in range(width - max_c)
            ]
            random.shuffle(placements)
            for top, left in placements:
                if _can_place_component(occupied, variant, top, left):
                    _place_component(grid, occupied, variant, color, top, left)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    sep_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    if len(sep_rows) != 1:
        return False
    if np.any(grid[np.logical_and(grid == GREY, ~np.repeat(np.all(grid == GREY, axis=1)[:, None], grid.shape[1], axis=1))]):
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    control = grid[int(sep_rows[0]) + 1 :]
    bg = get_bg(grid)
    counts = [len(components_mask(control == color)) for color in sorted(set(map(int, np.unique(control))) - {bg})]
    return len(set(counts)) == len(counts)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        template = _generate_template()
        template_h, template_w = template.shape

        color_total = random.randint(2, 4)
        control_palette = random.sample(CONTROL_COLORS, color_total)
        count_values = random.sample(COMPONENT_COUNT_OPTIONS, color_total)
        color_counts = list(zip(control_palette, count_values))
        random.shuffle(color_counts)

        width = random.randint(max(template_w + 4, 12), 18)
        top_height = template_h + random.randint(2, 5)
        control_height = random.randint(8, 13)
        if top_height + 1 + control_height > 30:
            continue

        control = _generate_control_region(control_height, width, color_counts)
        if control is None:
            continue

        grid = np.zeros((top_height + 1 + control_height, width), dtype=int)
        sep = top_height
        grid[sep, :] = GREY

        top_margin = random.randint(1, top_height - template_h)
        left_margin = random.randint(1, width - template_w - 1)
        grid[top_margin : top_margin + template_h, left_margin : left_margin + template_w] = template
        grid[sep + 1 :, :] = control

        if _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 1")
