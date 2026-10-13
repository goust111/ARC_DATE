from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
COLOR_POOL = tuple(color for color in range(10) if color != RED)

PANEL_LAYOUT_CHOICES = ((3, 3), (3, 5), (5, 3))
PANEL_HEIGHT_RANGE = (4, 6)
PANEL_WIDTH_RANGE = (4, 6)
EXTRA_COLOR_RANGE = (1, 3)
SITE_COUNT_CAP = 8
SEPARATOR_WIDTH = 1

MOTIF_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one color"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "majority vote must be unique"
    return int(ranked[0][0])


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one color"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background color must be uniquely dominant"
    return int(ranked[0][0])


def _content_bands(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(separator_mask.tolist()):
        if is_separator:
            if start is not None:
                bands.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        bands.append((start, len(separator_mask)))
    return bands


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(g):
        color = int(color)
        row_mask = np.all(g == color, axis=1)
        col_mask = np.all(g == color, axis=0)
        if not bool(row_mask.any()) or not bool(col_mask.any()):
            continue

        row_bands = _content_bands(row_mask)
        col_bands = _content_bands(col_mask)
        if len(row_bands) < 2 or len(col_bands) < 2:
            continue

        row_sizes = {int(r1 - r0) for r0, r1 in row_bands}
        col_sizes = {int(c1 - c0) for c0, c1 in col_bands}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        candidates.append((color, row_bands, col_bands))

    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _extract_panels(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> np.ndarray:
    panel_h = int(row_bands[0][1] - row_bands[0][0])
    panel_w = int(col_bands[0][1] - col_bands[0][0])
    panels = np.empty((len(row_bands) * len(col_bands), panel_h, panel_w), dtype=int)

    idx = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            panel = grid[r0:r1, c0:c1]
            assert panel.shape == (panel_h, panel_w), "panels must share one size"
            panels[idx] = panel
            idx += 1
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify defect pixels by local majority vote across separator-delimited panels,
    then keep only positions where the number of defective panels is odd.

    The output has the same size as a single panel. It is filled with the panel
    background color, and every odd-parity defect position is recolored red.
    """
    g = np.asarray(grid, dtype=int)
    _separator, row_bands, col_bands = _separator_structure(g)
    panels = _extract_panels(g, row_bands, col_bands)
    assert panels.shape[0] >= 3, "need multiple panels for voting"

    bg = _dominant_color(panels.reshape(-1))
    panel_h, panel_w = map(int, panels.shape[1:])

    defect_counts = np.zeros((panel_h, panel_w), dtype=int)
    for r in range(panel_h):
        for c in range(panel_w):
            consensus = _unique_mode(panels[:, r, c])
            defect_counts[r, c] = int(np.count_nonzero(panels[:, r, c] != consensus))

    out = np.full((panel_h, panel_w), bg, dtype=int)
    out[defect_counts % 2 == 1] = RED
    return out


def _ensure_row_col_coverage(mask: np.ndarray) -> np.ndarray:
    height, width = map(int, mask.shape)
    for r in range(height):
        if not bool(mask[r].any()):
            mask[r, random.randrange(width)] = True
    for c in range(width):
        if not bool(mask[:, c].any()):
            mask[random.randrange(height), c] = True
    return mask


def _random_mask_scatter(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        mask[r, random.randrange(width)] = True
    for c in range(width):
        mask[random.randrange(height), c] = True
    while int(mask.sum()) < target:
        mask[random.randrange(height), random.randrange(width)] = True
    return _ensure_row_col_coverage(mask)


def _random_mask_strokes(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(random.randint(2, 4)):
        shape = random.choice(("hseg", "vseg", "rect"))
        if shape == "hseg":
            r = random.randrange(height)
            c0 = random.randrange(width)
            c1 = random.randrange(c0, width)
            mask[r, c0 : c1 + 1] = True
        elif shape == "vseg":
            c = random.randrange(width)
            r0 = random.randrange(height)
            r1 = random.randrange(r0, height)
            mask[r0 : r1 + 1, c] = True
        else:
            r0 = random.randrange(height)
            r1 = random.randrange(r0, height)
            c0 = random.randrange(width)
            c1 = random.randrange(c0, width)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True

    mask = _ensure_row_col_coverage(mask)
    while int(mask.sum()) < target:
        r = random.randrange(height)
        c = random.randrange(width)
        mask[r, c] = True
    return mask


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _random_mask_growth(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    frontier: list[tuple[int, int]] = []
    for _ in range(random.randint(1, 3)):
        seed = (random.randrange(height), random.randrange(width))
        if not bool(mask[seed]):
            mask[seed] = True
            frontier.append(seed)

    while int(mask.sum()) < target:
        r, c = random.choice(frontier)
        options = [(nr, nc) for nr, nc in _neighbors4(r, c, height, width) if not bool(mask[nr, nc])]
        if options:
            nr, nc = random.choice(options)
            mask[nr, nc] = True
            frontier.append((nr, nc))
        else:
            frontier = [(rr, cc) for rr, cc in frontier if any(not bool(mask[nr, nc]) for nr, nc in _neighbors4(rr, cc, height, width))]
            if not frontier:
                frontier = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
                if len(frontier) == height * width:
                    break
                while True:
                    cell = (random.randrange(height), random.randrange(width))
                    if not bool(mask[cell]):
                        mask[cell] = True
                        frontier.append(cell)
                        break

    return _ensure_row_col_coverage(mask)


def _make_consensus_panel(height: int, width: int, bg: int, fg: int) -> np.ndarray:
    area = int(height * width)
    min_fg = max(height, width, max(4, area // 5))
    max_fg = min(area - 3, max(min_fg, int(np.floor(0.42 * area))))

    for _ in range(MOTIF_ATTEMPTS):
        target = random.randint(min_fg, max_fg)
        style = random.choice(("scatter", "strokes", "growth"))
        if style == "scatter":
            mask = _random_mask_scatter(height, width, target)
        elif style == "strokes":
            mask = _random_mask_strokes(height, width, target)
        else:
            mask = _random_mask_growth(height, width, target)

        fg_count = int(mask.sum())
        if fg_count < min_fg or fg_count > max_fg:
            continue
        if not bool(np.all(mask.any(axis=1))) or not bool(np.all(mask.any(axis=0))):
            continue

        panel = np.full((height, width), bg, dtype=int)
        panel[mask] = fg
        return panel

    raise RuntimeError("failed to create a valid consensus panel")


def _different_color(base: int, bg: int, fg: int, extras: list[int]) -> int:
    choices = [bg, fg, *extras]
    choices = [color for color in choices if color != int(base)]
    return int(random.choice(choices))


def _site_counts(panel_count: int, site_count: int, max_site_count: int) -> list[int]:
    odd_choices = [count for count in range(1, max_site_count + 1) if count % 2 == 1]
    even_choices = [count for count in range(2, max_site_count + 1) if count % 2 == 0]
    assert odd_choices and even_choices, "need both odd and even defect-count branches"

    extra_cap = min(4, max_site_count)
    counts = [
        random.choice(odd_choices),
        random.choice(even_choices),
        *[random.randint(1, extra_cap) for _ in range(site_count - 2)],
    ]

    while sum(counts) < panel_count:
        growable = [idx for idx in range(2, site_count) if counts[idx] < max_site_count]
        assert growable, "site capacities must cover all panels"
        counts[random.choice(growable)] += 1

    return counts


def _assign_sites(panel_count: int, counts: list[int]) -> tuple[list[set[int]], list[int]]:
    assignments = [set() for _ in counts]
    remaining = counts[:]
    panel_loads = [0] * panel_count

    panel_order = list(range(panel_count))
    random.shuffle(panel_order)
    for panel_idx in panel_order:
        available = [site_idx for site_idx, slots in enumerate(remaining) if slots > 0]
        assert available, "assignment capacity exhausted too early"
        best_remaining = max(remaining[site_idx] for site_idx in available)
        choices = [site_idx for site_idx in available if remaining[site_idx] == best_remaining]
        site_idx = random.choice(choices)
        assignments[site_idx].add(panel_idx)
        remaining[site_idx] -= 1
        panel_loads[panel_idx] += 1

    site_order = list(range(len(counts)))
    random.shuffle(site_order)
    for site_idx in site_order:
        while remaining[site_idx] > 0:
            candidates = [panel_idx for panel_idx in range(panel_count) if panel_idx not in assignments[site_idx]]
            assert candidates, "a site cannot mark more panels than exist"
            min_load = min(panel_loads[panel_idx] for panel_idx in candidates)
            balanced = [panel_idx for panel_idx in candidates if panel_loads[panel_idx] <= min_load + 1]
            panel_idx = random.choice(balanced)
            assignments[site_idx].add(panel_idx)
            remaining[site_idx] -= 1
            panel_loads[panel_idx] += 1

    return assignments, panel_loads


def _panel_fg_is_unique(panel: np.ndarray, bg: int, fg: int) -> bool:
    non_bg = panel[panel != bg]
    if non_bg.size == 0:
        return False
    counts = Counter(map(int, non_bg.reshape(-1)))
    ranked = counts.most_common()
    return ranked[0][0] == fg and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])


def _render_grid(
    panels: list[np.ndarray], panel_rows: int, panel_cols: int, panel_h: int, panel_w: int, separator: int
) -> np.ndarray:
    height = panel_rows * panel_h + (panel_rows - 1) * SEPARATOR_WIDTH
    width = panel_cols * panel_w + (panel_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), separator, dtype=int)

    idx = 0
    for br in range(panel_rows):
        for bc in range(panel_cols):
            r0 = br * (panel_h + SEPARATOR_WIDTH)
            c0 = bc * (panel_w + SEPARATOR_WIDTH)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panels[idx]
            idx += 1
    return grid


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    panel_rows, panel_cols = random.choice(PANEL_LAYOUT_CHOICES)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    panel_count = panel_rows * panel_cols
    area = panel_h * panel_w

    separator, bg, fg = random.sample(COLOR_POOL, 3)
    remaining_colors = [color for color in COLOR_POOL if color not in {separator, bg, fg}]
    extra_colors = random.sample(remaining_colors, random.randint(*EXTRA_COLOR_RANGE))

    consensus = _make_consensus_panel(panel_h, panel_w, bg, fg)
    max_site_count = (panel_count - 1) // 2

    min_sites = max(4, (panel_count + max_site_count - 1) // max_site_count)
    max_sites = min(area, max(min_sites, SITE_COUNT_CAP))
    site_count = random.randint(min_sites, max_sites)

    coords = [(r, c) for r in range(panel_h) for c in range(panel_w)]
    random.shuffle(coords)
    sites = coords[:site_count]

    counts = _site_counts(panel_count, site_count, max_site_count)
    assignments, panel_loads = _assign_sites(panel_count, counts)
    if max(panel_loads) > max(4, area // 4):
        return None

    panels = [consensus.copy() for _ in range(panel_count)]
    for (r, c), site_panels in zip(sites, assignments):
        for panel_idx in site_panels:
            panels[panel_idx][r, c] = _different_color(int(consensus[r, c]), bg, fg, extra_colors)

    if any(not _panel_fg_is_unique(panel, bg, fg) for panel in panels):
        return None

    try:
        if _dominant_color(np.stack(panels, axis=0)) != bg:
            return None
    except AssertionError:
        return None

    expected = np.full((panel_h, panel_w), bg, dtype=int)
    odd_seen = False
    even_seen = False
    for (r, c), site_panels in zip(sites, assignments):
        if len(site_panels) % 2 == 1:
            expected[r, c] = RED
            odd_seen = True
        else:
            even_seen = True
    if not odd_seen or not even_seen:
        return None

    inp = _render_grid(panels, panel_rows, panel_cols, panel_h, panel_w, separator)
    if np.any(inp == RED):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError("failed to generate a valid panel-defect input")
