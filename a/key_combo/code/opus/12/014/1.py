from __future__ import annotations

import numpy as np


BACKGROUND = 0
FOREGROUND = 2
SEPARATOR = 5
MAX_DEPTH = 4
_RNG = np.random.default_rng()


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BACKGROUND, FOREGROUND, SEPARATOR}, "input uses only black, red, and grey"

    _, width = g.shape
    separators = [c for c in range(width) if np.all(g[:, c] == SEPARATOR)]
    assert separators, "expected one or more full-height grey separator columns"

    non_separator_grey = (g == SEPARATOR)
    non_separator_grey[:, separators] = False
    assert not np.any(non_separator_grey), "grey should appear only in separator columns"

    panel_count = len(separators) + 1
    assert 2 <= panel_count <= MAX_DEPTH, "depth colors only support up to four panels"

    total_panel_width = width - len(separators)
    panel_width, remainder = divmod(total_panel_width, panel_count)
    assert remainder == 0 and panel_width > 0, "separator columns must split equal-width panels"

    expected_separators = [panel_width + i * (panel_width + 1) for i in range(panel_count - 1)]
    assert separators == expected_separators, "separator columns must alternate with equal-width panels"

    panels: list[np.ndarray] = []
    start = 0
    for idx in range(panel_count):
        stop = start + panel_width
        panel = g[:, start:stop]
        assert panel.shape[1] == panel_width
        assert not np.any(panel == SEPARATOR), "panel regions cannot contain separator color"
        panels.append(panel)
        start = stop
        if idx < panel_count - 1:
            assert np.all(g[:, start] == SEPARATOR)
            start += 1

    assert start == width
    assert all(np.any(panel == FOREGROUND) for panel in panels), "every panel should contain foreground"
    return panels


def _aligned_foreground_masks(grid: np.ndarray, *, mirror_even: bool) -> list[np.ndarray]:
    panels = _split_panels(grid)
    masks: list[np.ndarray] = []
    for idx, panel in enumerate(panels, start=1):
        fg = panel == FOREGROUND
        if mirror_even and idx % 2 == 0:
            fg = np.fliplr(fg)
        masks.append(fg)
    return masks


def _depth_map_from_masks(masks: list[np.ndarray]) -> np.ndarray:
    assert masks
    counts = np.zeros(masks[0].shape, dtype=int)
    for mask in masks:
        counts += mask.astype(int)
    assert counts.max(initial=0) <= MAX_DEPTH
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold all separator-split panels onto the first panel and color each cell by its overlap depth.

    Input structure:
    - The grid has black background, red foreground pixels, and full-height grey separator columns.
    - The separator columns split the grid into equal-width panels of the same height.
    - Foreground appears only inside panels; even-numbered panels are displayed before folding.

    Rule:
    - Remove the grey separator columns and treat the leftmost panel as the output coordinate system.
    - Keep odd-numbered panels as-is, but mirror every even-numbered panel horizontally before aligning it.
    - Count how many aligned panels contribute a foreground pixel at each cell, keeping black for depth 0 and mapping depths 1..4 to colors 1..4.

    Key insight:
    - The output is a folded overlap-depth image, so the solver must both recover the alternating mirror rule and count stacked foreground contributions instead of choosing a single overlaid color.
    """
    depth_map = _depth_map_from_masks(_aligned_foreground_masks(grid, mirror_even=True))
    return depth_map.astype(int)


def _random_partition(total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [total]
    cuts = sorted(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _grow_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    free = {(r, c) for r in range(height) for c in range(width)}
    component_count = 1 if count <= 4 else int(_RNG.integers(1, min(3, count) + 1))
    sizes = _random_partition(count, component_count)
    cells: list[tuple[int, int]] = []

    for size in sizes:
        ordered_free = sorted(free)
        seed = ordered_free[int(_RNG.integers(len(ordered_free)))]
        free.remove(seed)
        component = [seed]
        frontier = {nb for nb in _neighbors4(seed[0], seed[1], height, width) if nb in free}

        while len(component) < size:
            if frontier:
                ordered = sorted(frontier)
                nxt = ordered[int(_RNG.integers(len(ordered)))]
            else:
                ordered = sorted(free)
                nxt = ordered[int(_RNG.integers(len(ordered)))]
            free.remove(nxt)
            frontier.discard(nxt)
            component.append(nxt)
            for nb in _neighbors4(nxt[0], nxt[1], height, width):
                if nb in free:
                    frontier.add(nb)

        cells.extend(component)

    return cells


def _build_input_from_aligned_masks(masks: list[np.ndarray]) -> np.ndarray:
    panel_count = len(masks)
    height, panel_width = masks[0].shape
    grid = np.full((height, panel_count * panel_width + panel_count - 1), BACKGROUND, dtype=int)

    for idx, mask in enumerate(masks, start=1):
        patch = mask if idx % 2 == 1 else np.fliplr(mask)
        start = (idx - 1) * (panel_width + 1)
        grid[:, start:start + panel_width] = np.where(patch, FOREGROUND, BACKGROUND)
        if idx < panel_count:
            grid[:, start + panel_width] = SEPARATOR

    return grid


def _panel_signatures(masks: list[np.ndarray]) -> set[bytes]:
    return {np.packbits(mask.astype(np.uint8), axis=None).tobytes() for mask in masks}


def _is_valid_aligned_layout(masks: list[np.ndarray]) -> bool:
    panel_count = len(masks)
    counts = _depth_map_from_masks(masks)

    if any(int(mask.sum()) < 2 for mask in masks):
        return False
    if len(_panel_signatures(masks)) != panel_count:
        return False
    if not all(depth in counts for depth in range(1, panel_count + 1)):
        return False
    if counts.max(initial=0) != panel_count:
        return False

    even_masks = [mask for idx, mask in enumerate(masks, start=1) if idx % 2 == 0]
    if not even_masks or any(np.array_equal(mask, np.fliplr(mask)) for mask in even_masks):
        return False

    mirrored_wrong = np.zeros_like(counts)
    for idx, mask in enumerate(masks, start=1):
        mirrored_wrong += (mask if idx % 2 == 1 else np.fliplr(mask)).astype(int)
    if np.array_equal(mirrored_wrong, counts):
        return False

    return True


def _sample_aligned_masks() -> list[np.ndarray] | None:
    panel_count = int(_RNG.choice([3, 4]))
    height = int(_RNG.integers(4, 8))
    max_panel_width = min(6, (30 - (panel_count - 1)) // panel_count)
    panel_width = int(_RNG.integers(4, max_panel_width + 1))

    min_cells = panel_count + 2
    max_cells = min(height * panel_width - 1, max(min_cells + 1, int(np.ceil(height * panel_width * 0.5))))
    if min_cells > max_cells:
        return None
    cell_count = int(_RNG.integers(min_cells, max_cells + 1))

    coords = _grow_cells(height, panel_width, cell_count)
    if len(set(coords)) != cell_count:
        return None

    depths = list(range(1, panel_count + 1))
    depths.extend(int(x) for x in _RNG.integers(1, panel_count + 1, size=cell_count - panel_count))
    _RNG.shuffle(depths)

    masks = [np.zeros((height, panel_width), dtype=bool) for _ in range(panel_count)]
    for (r, c), depth in zip(coords, depths):
        chosen = _RNG.choice(panel_count, size=depth, replace=False)
        for panel_idx in chosen:
            masks[int(panel_idx)][r, c] = True

    if not _is_valid_aligned_layout(masks):
        return None
    return masks


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if actual.ndim != 2 or max(actual.shape) > 30:
        return False

    panel_count = len(_split_panels(grid))
    if not all(depth in actual for depth in range(1, panel_count + 1)):
        return False

    wrong = _depth_map_from_masks(_aligned_foreground_masks(grid, mirror_even=False))
    if np.array_equal(wrong, actual):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        masks = _sample_aligned_masks()
        if masks is None:
            continue

        grid = _build_input_from_aligned_masks(masks)
        expected = _depth_map_from_masks(masks)
        if _semantic_checks(grid, expected):
            return grid

    raise RuntimeError("failed to generate a valid folded-depth input")
