from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
CONTENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ARC_SIDE = 30
GENERATION_ATTEMPTS = 400


def _separator_indices(grid: np.ndarray) -> tuple[int, int]:
    row_candidates = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and np.all(grid[r] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != BG and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(row_candidates) == 1 and len(col_candidates) == 1
    sep_r = int(row_candidates[0])
    sep_c = int(col_candidates[0])
    sep_color = int(grid[sep_r, sep_c])
    assert np.all(grid[sep_r] == sep_color)
    assert np.all(grid[:, sep_c] == sep_color)
    return sep_r, sep_c


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    sep_r, sep_c = _separator_indices(grid)
    panels = [
        grid[:sep_r, :sep_c],
        grid[:sep_r, sep_c + 1 :],
        grid[sep_r + 1 :, :sep_c],
        grid[sep_r + 1 :, sep_c + 1 :],
    ]
    panel_shape = panels[0].shape
    assert panel_shape[0] > 1 and panel_shape[1] > 1
    assert all(panel.shape == panel_shape for panel in panels)
    return panels


def _panel_info(panel: np.ndarray) -> tuple[np.ndarray, int, int]:
    hint_color = int(panel[-1, -1])
    assert hint_color != BG

    visible_mask = panel != BG
    visible_mask[-1, -1] = False
    assert bool(visible_mask.any())

    colors = np.unique(panel[visible_mask])
    assert len(colors) == 1
    visible_color = int(colors[0])
    assert visible_color != hint_color
    return visible_mask, visible_color, hint_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four complementary panel fragments into one pattern and recolor each fragment via the corner hints.

    Input structure:
    - The background is black.
    - A single grey horizontal line and a single grey vertical line split the grid into four equal panels.
    - Each panel contains some visible pixels from one shared underlying pattern; across the four panels these visible subsets are complementary.
    - All visible pattern pixels inside one panel share one color, and the panel's bottom-right cell is a one-pixel hint block whose color matches another panel's visible color.

    Rule:
    - Ignore the grey separators and treat the bottom-right cell of each panel as a hint, not as part of the pattern.
    - Overlay the four panel fragments to recover the full pattern, noting which panel contributed each visible pixel.
    - If panel A's hint color equals panel B's visible color, recolor every pixel contributed by panel B with panel A's visible color.
    - Output the recovered full pattern at single-panel size on a black background.

    Key insight:
    - The four panels do not compete spatially; they partition one hidden pattern, while the corner hints define an indirect cross-panel recoloring by saying whose pixels should inherit the current panel's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels = _split_panels(g)
    panel_h, panel_w = panels[0].shape

    masks: list[np.ndarray] = []
    visible_colors: list[int] = []
    hint_colors: list[int] = []
    for panel in panels:
        mask, visible_color, hint_color = _panel_info(panel)
        masks.append(mask)
        visible_colors.append(visible_color)
        hint_colors.append(hint_color)

    assert len(set(visible_colors)) == 4
    assert len(set(hint_colors)) == 4
    assert set(hint_colors) == set(visible_colors)

    coverage = np.zeros((panel_h, panel_w), dtype=int)
    for mask in masks:
        coverage += mask.astype(int)
    assert np.all(coverage <= 1)

    color_to_panel = {color: idx for idx, color in enumerate(visible_colors)}
    source_to_output_color: list[int | None] = [None] * 4
    for panel_idx, hint_color in enumerate(hint_colors):
        source_idx = color_to_panel[hint_color]
        assert source_idx != panel_idx
        assert source_to_output_color[source_idx] is None
        source_to_output_color[source_idx] = visible_colors[panel_idx]
    assert all(color is not None for color in source_to_output_color)

    out = np.full((panel_h, panel_w), BG, dtype=int)
    for source_idx, output_color in enumerate(source_to_output_color):
        out[masks[source_idx]] = int(output_color)
    return out


def _neighbors4(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) != (h - 1, w - 1):
            out.append((nr, nc))
    return out


def _make_pattern(panel_h: int, panel_w: int) -> np.ndarray | None:
    usable = panel_h * panel_w - 1
    min_cells = max(8, usable // 4)
    max_cells = max(min_cells, (usable * 3) // 5)
    if max_cells < 4:
        return None

    target = random.randint(min_cells, max_cells)
    allowed = [(r, c) for r in range(panel_h) for c in range(panel_w) if (r, c) != (panel_h - 1, panel_w - 1)]
    start = random.choice(allowed)

    mask = np.zeros((panel_h, panel_w), dtype=bool)
    mask[start] = True
    frontier = set(_neighbors4(panel_h, panel_w, *start))

    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nbr in _neighbors4(panel_h, panel_w, *cell):
            if not mask[nbr]:
                frontier.add(nbr)

    if int(mask.sum()) != target:
        return None

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) < 2 or len(cols) < 2:
        return None
    return mask


def _positive_split(total: int, parts: int, min_each: int) -> list[int] | None:
    if total < parts * min_each:
        return None
    counts = [min_each] * parts
    remaining = total - parts * min_each
    while remaining:
        idx = random.randrange(parts)
        counts[idx] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _partition_random(mask: np.ndarray) -> list[np.ndarray] | None:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    min_each = 2 if len(coords) >= 12 else 1
    counts = _positive_split(len(coords), 4, min_each)
    if counts is None:
        return None

    random.shuffle(coords)
    labels = []
    for idx, count in enumerate(counts):
        labels.extend([idx] * count)
    random.shuffle(labels)

    parts = [np.zeros_like(mask, dtype=bool) for _ in range(4)]
    for (r, c), label in zip(coords, labels):
        parts[label][r, c] = True
    return parts


def _partition_formula(mask: np.ndarray) -> list[np.ndarray] | None:
    offset_r = random.randrange(4)
    offset_c = random.randrange(4)
    formula = random.choice(
        (
            lambda r, c: (r + offset_r) % 4,
            lambda r, c: (c + offset_c) % 4,
            lambda r, c: (r + c + offset_r) % 4,
            lambda r, c: ((r + offset_r) % 2) * 2 + ((c + offset_c) % 2),
        )
    )

    parts = [np.zeros_like(mask, dtype=bool) for _ in range(4)]
    for r, c in np.argwhere(mask):
        label = int(formula(int(r), int(c)))
        parts[label][int(r), int(c)] = True
    return parts


def _partition_voronoi(mask: np.ndarray) -> list[np.ndarray] | None:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    if len(coords) < 4:
        return None
    seeds = random.sample(coords, 4)
    parts = [np.zeros_like(mask, dtype=bool) for _ in range(4)]
    for r, c in coords:
        best_idx = min(
            range(4),
            key=lambda idx: (abs(r - seeds[idx][0]) + abs(c - seeds[idx][1]), idx),
        )
        parts[best_idx][r, c] = True
    return parts


def _partition_pattern(mask: np.ndarray) -> list[np.ndarray] | None:
    builders = (_partition_random, _partition_formula, _partition_voronoi)
    parts = random.choice(builders)(mask)
    if parts is None:
        return None

    counts = [int(part.sum()) for part in parts]
    min_each = 2 if int(mask.sum()) >= 12 else 1
    if min(counts) < min_each:
        return None
    return parts


def _random_derangement(size: int) -> list[int]:
    perm = list(range(size))
    while True:
        random.shuffle(perm)
        if all(idx != perm[idx] for idx in range(size)):
            return perm[:]


def _expected_output(parts: list[np.ndarray], colors: list[int], hint_targets: list[int]) -> np.ndarray:
    panel_h, panel_w = parts[0].shape
    out = np.full((panel_h, panel_w), BG, dtype=int)
    for panel_idx, source_idx in enumerate(hint_targets):
        out[parts[source_idx]] = colors[panel_idx]
    return out


def _assemble_input(parts: list[np.ndarray], colors: list[int], hint_targets: list[int]) -> np.ndarray:
    panel_h, panel_w = parts[0].shape
    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), BG, dtype=int)
    grid[panel_h, :] = SEPARATOR
    grid[:, panel_w] = SEPARATOR

    starts = (
        (0, 0),
        (0, panel_w + 1),
        (panel_h + 1, 0),
        (panel_h + 1, panel_w + 1),
    )
    for idx, (r0, c0) in enumerate(starts):
        panel = grid[r0 : r0 + panel_h, c0 : c0 + panel_w]
        panel[parts[idx]] = colors[idx]
        panel[-1, -1] = colors[hint_targets[idx]]
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(5, 14)
    panel_w = random.randint(5, 14)
    if 2 * panel_h + 1 > MAX_ARC_SIDE or 2 * panel_w + 1 > MAX_ARC_SIDE:
        return None

    pattern = _make_pattern(panel_h, panel_w)
    if pattern is None:
        return None

    parts = _partition_pattern(pattern)
    if parts is None:
        return None

    colors = random.sample(list(CONTENT_COLORS), 4)
    hint_targets = _random_derangement(4)
    grid = _assemble_input(parts, colors, hint_targets)
    expected = _expected_output(parts, colors, hint_targets)

    try:
        actual = main(grid)
    except Exception:
        return None

    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
