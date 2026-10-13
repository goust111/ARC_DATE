from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _unique_non_bg_color(panel: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(panel) if int(color) != bg]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    return colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the pixelwise overlap of two equal panels with the other panel's foreground color.

    Input structure
    - The dominant color is the shared background.
    - The grid is split by one full-height vertical separator column into two equal panels.
    - The left panel contains exactly one non-background foreground color.
    - The right panel contains exactly one non-background foreground color.

    Rule
    - Compare the two panels position by position.
    - Wherever both panels have a foreground pixel, recolor the left pixel with the
      right panel's foreground color and the right pixel with the left panel's
      foreground color.
    - Leave every non-overlapping foreground pixel, every background pixel, and the
      separator unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    height, width = g.shape
    assert width >= 3 and width % 2 == 1, "grid must contain two equal-width panels and one separator column"
    mid = width // 2

    separator = g[:, mid]
    assert np.all(separator == separator[0]), "separator must be a uniform full-height column"

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape and left.size > 0, "panels must be equal and non-empty"

    bg = _dominant_color(g)
    left_color = _unique_non_bg_color(left, bg)
    right_color = _unique_non_bg_color(right, bg)
    assert left_color != right_color, "panel foreground colors must differ"
    assert int(separator[0]) not in {left_color, right_color}, "separator cannot reuse a foreground color"

    left_mask = left == left_color
    right_mask = right == right_color
    assert bool(left_mask.any()) and bool(right_mask.any()), "both panels need visible foreground"

    overlap = left_mask & right_mask

    out = g.copy()
    left_out = out[:, :mid]
    right_out = out[:, mid + 1 :]
    left_out[overlap] = right_color
    right_out[overlap] = left_color
    return out


# --------------- generator ---------------

LEFT_COLOR = 1
RIGHT_COLOR = 2
AUX_COLORS = [0, 3, 4, 5, 6, 7, 8, 9]

HEIGHT_RANGE = (6, 14)
PANEL_WIDTH_RANGE = (5, 12)
MAX_COMPONENTS = 3
OVERLAP_MODES = ("low", "medium", "high")
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue

        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)

    return comps


def _partition_total(total: int, parts: int) -> list[int]:
    remaining = total
    sizes: list[int] = []
    for idx in range(parts):
        left = parts - idx - 1
        size = random.randint(1, remaining - left)
        sizes.append(size)
        remaining -= size
    sizes.sort(reverse=True)
    return sizes


def _grow_connected_subset(region: list[tuple[int, int]], size: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    region_set = set(region)
    start = random.choice(region)
    chosen = {start}
    frontier = {
        (start[0] + dr, start[1] + dc)
        for dr, dc in DIRS4
        if (start[0] + dr, start[1] + dc) in region_set
    }

    while len(chosen) < size:
        assert frontier, "connected region unexpectedly exhausted"
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in chosen:
            continue
        chosen.add(cell)
        r, c = cell
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < shape[0] and 0 <= nc < shape[1] and (nr, nc) in region_set and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    return list(chosen)


def _sample_mask(shape: tuple[int, int], available: np.ndarray, total: int, n_components: int) -> np.ndarray | None:
    free = available.copy()
    mask = np.zeros(shape, dtype=bool)

    for size in _partition_total(total, n_components):
        options = [comp for comp in _components(free) if len(comp) >= size]
        if not options:
            return None
        region = random.choice(options)
        part = _grow_connected_subset(region, size, shape)
        for r, c in part:
            mask[r, c] = True
            free[r, c] = False

    return mask


def _pick_overlap(left_total: int, right_total: int) -> int:
    max_overlap = min(left_total, right_total) - 1
    assert max_overlap >= 1

    ratio_ranges = {
        "low": (0.15, 0.35),
        "medium": (0.36, 0.65),
        "high": (0.66, 1.00),
    }
    lo, hi = ratio_ranges[random.choice(OVERLAP_MODES)]
    overlap = int(round(random.uniform(lo, hi) * max_overlap))
    return max(1, min(max_overlap, overlap))


def _foreground_totals(area: int) -> tuple[int, int]:
    panel_min = max(3, area // 8)
    panel_max = max(panel_min + 1, area // 3)
    return random.randint(panel_min, panel_max), random.randint(panel_min, panel_max)


def _build_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    area = height * width
    left_total, right_total = _foreground_totals(area)
    overlap_count = _pick_overlap(left_total, right_total)
    left_only_count = left_total - overlap_count
    right_only_count = right_total - overlap_count

    assert left_only_count > 0 and right_only_count > 0

    counts = {
        "overlap": overlap_count,
        "left_only": left_only_count,
        "right_only": right_only_count,
    }

    available = np.ones((height, width), dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for name, total in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        n_components = random.randint(1, min(MAX_COMPONENTS, total))
        mask = _sample_mask((height, width), available, total, n_components)
        if mask is None:
            return None
        masks[name] = mask
        available[mask] = False

    return masks["overlap"], masks["left_only"], masks["right_only"]


def _expected_output(inp: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    out = inp.copy()
    mid = inp.shape[1] // 2
    left_out = out[:, :mid]
    right_out = out[:, mid + 1 :]
    left_out[overlap] = RIGHT_COLOR
    right_out[overlap] = LEFT_COLOR
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)

    colors = random.sample(AUX_COLORS, 2)
    bg, separator = colors[0], colors[1]

    built = _build_masks(height, panel_width)
    if built is None:
        return None
    overlap, left_only, right_only = built

    left_mask = overlap | left_only
    right_mask = overlap | right_only
    assert bool(overlap.any()) and bool(left_only.any()) and bool(right_only.any())

    grid = np.full((height, 2 * panel_width + 1), bg, dtype=int)
    mid = panel_width
    grid[:, mid] = separator
    left_panel = grid[:, :mid]
    right_panel = grid[:, mid + 1 :]
    left_panel[left_mask] = LEFT_COLOR
    right_panel[right_mask] = RIGHT_COLOR

    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None

    expected = _expected_output(grid, overlap)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
