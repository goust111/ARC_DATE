from __future__ import annotations

from collections import Counter
import random

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_ATTEMPTS = 500


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following arc_utils.get_bg semantics."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _split_input(
    grid: np.ndarray,
) -> tuple[int, list[np.ndarray], list[int]]:
    """Extract the shared background, four row-major panels, and top-down legend."""
    height, width = grid.shape
    assert height >= 7 and height % 2 == 1, "expected two equal panel rows and one divider row"
    panel_h = (height - 1) // 2
    divider_r = panel_h

    # In a valid layout the vertical arm of the cross spans the complete input
    # height.  Its column also determines the width of both panel columns.
    candidates: list[tuple[int, int]] = []
    for divider_c in range(1, width):
        block_w = 2 * divider_c + 1
        if block_w >= width:
            continue
        divider_color = int(grid[divider_r, divider_c])
        if not bool(np.all(grid[:, divider_c] == divider_color)):
            continue
        if not bool(np.all(grid[divider_r, :block_w] == divider_color)):
            continue
        expected = np.zeros((height, block_w), dtype=bool)
        expected[divider_r, :] = True
        expected[:, divider_c] = True
        if np.array_equal(grid[:, :block_w] == divider_color, expected):
            candidates.append((divider_c, divider_color))

    assert len(candidates) == 1, "expected one unambiguous cross divider"
    panel_w, divider_color = candidates[0]
    block_w = 2 * panel_w + 1

    bg = _background(grid)
    assert bg != divider_color, "background and divider must be visibly distinct"
    assert not bool(np.any(grid[:, block_w:] == divider_color)), "divider color must be confined to the cross"

    panels = [
        grid[:panel_h, :panel_w].copy(),
        grid[:panel_h, panel_w + 1 : block_w].copy(),
        grid[panel_h + 1 :, :panel_w].copy(),
        grid[panel_h + 1 :, panel_w + 1 : block_w].copy(),
    ]
    assert all(panel.shape == (panel_h, panel_w) for panel in panels)

    panel_colors: list[int] = []
    for panel in panels:
        colors = set(map(int, np.unique(panel))) - {bg}
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        panel_colors.append(colors.pop())
    assert len(set(panel_colors)) == 4, "the four panel colors must be distinct"

    # Everything to the right of the divided block is background except one
    # contiguous four-cell vertical legend.  The blank gutter separates the
    # legend structurally from the panel table.
    legend_area = grid[:, block_w:]
    coords = np.argwhere(legend_area != bg)
    assert coords.shape[0] == 4, "legend must contain exactly four non-background cells"
    legend_cols = set(map(int, coords[:, 1]))
    assert len(legend_cols) == 1, "legend must be vertical"
    legend_c = legend_cols.pop()
    assert legend_c >= 1, "a background gutter must separate panels and legend"
    legend_rows = sorted(map(int, coords[:, 0]))
    assert legend_rows == list(range(legend_rows[0], legend_rows[0] + 4)), "legend cells must be contiguous"
    legend_top_down = [int(legend_area[r, legend_c]) for r in legend_rows]
    assert len(set(legend_top_down)) == 4, "legend colors must be distinct"
    assert set(legend_top_down).isdisjoint(panel_colors), "legend colors must not encode panel identity"
    assert divider_color not in legend_top_down
    return bg, panels, legend_top_down


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite four count-ranked panel masks and recolor their layers from a vertical legend.

    Input structure:
    - A colored cross divides the left part of the grid into four equal panels on a shared background.
    - Every panel contains a mask in one distinct foreground color, and the four foreground counts are distinct.
    - To the right, after a background gutter, a vertical four-cell strip has four distinct colors unrelated to the panel colors.

    Rule:
    - Rank panel masks by foreground count from largest to smallest; this is the layer order from bottom to top.
    - Transparently overlay the masks in that order, so each later, smaller mask overwrites lower layers at shared coordinates.
    - Recolor the bottom layer with the strip's bottom color and proceed upward through the strip, ending with its top color.

    Key insight:
    - Relative mask count determines depth independently of panel position, while the legend's bottom-to-top order determines new layer colors independently of original colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, panels, legend_top_down = _split_input(g)
    masks = [panel != bg for panel in panels]
    counts = [int(mask.sum()) for mask in masks]
    assert all(count > 0 for count in counts), "panel masks must be non-empty"
    assert len(set(counts)) == 4, "panel foreground counts must be distinct"

    bottom_to_top = sorted(range(4), key=lambda i: counts[i], reverse=True)
    layer_colors = list(reversed(legend_top_down))
    out = np.full(panels[0].shape, bg, dtype=int)
    for panel_i, color in zip(bottom_to_top, layer_colors):
        out[masks[panel_i]] = color
    return out


def _grow_mask(
    shape: tuple[int, int],
    count: int,
    seed: tuple[int, int],
    forbidden: np.ndarray,
) -> np.ndarray | None:
    """Grow a random 4-connected mask without entering forbidden cells."""
    height, width = shape
    if forbidden[seed]:
        return None
    mask = np.zeros(shape, dtype=bool)
    mask[seed] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and not forbidden[rr, cc] and not mask[rr, cc]:
                frontier.add((rr, cc))

    add_frontier(*seed)
    while int(mask.sum()) < count:
        if not frontier:
            return None
        candidates = list(frontier)
        # Alternating between tips and broad fronts produces both winding and
        # compact masks while keeping every generated mask connected.
        if random.random() < 0.65:
            neighbor_counts = [
                sum(
                    0 <= r + dr < height
                    and 0 <= c + dc < width
                    and mask[r + dr, c + dc]
                    for dr, dc in _NEIGHBORS4
                )
                for r, c in candidates
            ]
            target_degree = random.choice((min(neighbor_counts), max(neighbor_counts)))
            candidates = [cell for cell, degree in zip(candidates, neighbor_counts) if degree == target_degree]
        cell = random.choice(candidates)
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)
    return mask


def _connected(mask: np.ndarray) -> bool:
    """Small self-contained 4-connectivity check using (row, column) coordinates."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            nxt = (rr, cc)
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _make_depth_masks(shape: tuple[int, int], counts: list[int]) -> list[np.ndarray] | None:
    """Build bottom-to-top blobs with one clear witness overlap per adjacent pair."""
    height, width = shape
    occupied = np.zeros(shape, dtype=bool)
    first_seed = (random.randrange(height), random.randrange(width))
    first = _grow_mask(shape, counts[0], first_seed, occupied)
    if first is None:
        return None
    layers = [first]
    occupied |= first

    for count in counts[1:]:
        previous_unique = layers[-1] & (np.sum(np.asarray(layers), axis=0) == 1)
        seed_options: list[tuple[int, int]] = []
        for r, c in np.argwhere(previous_unique):
            rr, cc = int(r), int(c)
            if any(
                0 <= rr + dr < height
                and 0 <= cc + dc < width
                and not occupied[rr + dr, cc + dc]
                for dr, dc in _NEIGHBORS4
            ):
                seed_options.append((rr, cc))
        if not seed_options:
            return None
        seed = random.choice(seed_options)
        forbidden = occupied.copy()
        forbidden[seed] = False
        layer = _grow_mask(shape, count, seed, forbidden)
        if layer is None:
            return None
        layers.append(layer)
        occupied |= layer
    return layers


def _render_semantics(
    masks: list[np.ndarray],
    counts: list[int],
    legend_top_down: list[int],
    bg: int,
    *,
    overwrite: bool = True,
    panel_order: list[int] | None = None,
) -> np.ndarray:
    """Independent generator oracle used to reject semantic shortcuts."""
    order = panel_order if panel_order is not None else sorted(range(4), key=lambda i: counts[i], reverse=True)
    colors = list(reversed(legend_top_down))
    out = np.full(masks[0].shape, bg, dtype=int)
    for panel_i, color in zip(order, colors):
        if overwrite:
            out[masks[panel_i]] = color
        else:
            paint = masks[panel_i] & (out == bg)
            out[paint] = color
    return out


def _semantic_checks(
    masks: list[np.ndarray],
    counts: list[int],
    legend_top_down: list[int],
    panel_colors: list[int],
    bg: int,
) -> bool:
    """Validate uniqueness, necessary dependencies, clarity, and anti-shortcut witnesses."""
    stack = np.asarray(masks, dtype=bool)
    coverage = stack.sum(axis=0)
    order = sorted(range(4), key=lambda i: counts[i], reverse=True)

    if [int(mask.sum()) for mask in masks] != counts or len(set(counts)) != 4:
        return False
    if not all(_connected(mask) for mask in masks):
        return False
    if int(coverage.max()) != 2 or int(np.sum(coverage == 2)) != 3:
        return False
    if not all(bool(np.any(masks[i] & (coverage == 1))) for i in range(4)):
        return False
    # Each adjacent depth pair has its own unambiguous overwrite witness; no
    # non-adjacent pair overlaps because total double-covered cells is three.
    if not all(int(np.sum(masks[order[k - 1]] & masks[order[k]])) == 1 for k in range(1, 4)):
        return False

    expected = _render_semantics(masks, counts, legend_top_down, bg)
    if set(map(int, np.unique(expected))) - {bg} != set(legend_top_down):
        return False
    if not (set(panel_colors).isdisjoint(legend_top_down) and set(panel_colors).isdisjoint({bg})):
        return False
    # Fixed panel order, reversed legend direction, and non-overwriting paint
    # must all give observably different answers on every generated example.
    if np.array_equal(expected, _render_semantics(masks, counts, legend_top_down, bg, panel_order=[0, 1, 2, 3])):
        return False
    if np.array_equal(expected, _render_semantics(masks, counts, list(reversed(legend_top_down)), bg)):
        return False
    if np.array_equal(expected, _render_semantics(masks, counts, legend_top_down, bg, overwrite=False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(7, 10)
    panel_w = random.randint(7, 10)
    area = panel_h * panel_w

    count_hi = min(14, area // 5)
    count_values = random.sample(range(6, count_hi + 1), 4)
    depth_counts = sorted(count_values, reverse=True)
    depth_panels = list(range(4))
    random.shuffle(depth_panels)
    if depth_panels in ([0, 1, 2, 3], [3, 2, 1, 0]):
        return None

    depth_masks = _make_depth_masks((panel_h, panel_w), depth_counts)
    if depth_masks is None:
        return None
    masks = [np.zeros((panel_h, panel_w), dtype=bool) for _ in range(4)]
    counts = [0] * 4
    for depth, panel_i in enumerate(depth_panels):
        masks[panel_i] = depth_masks[depth]
        counts[panel_i] = depth_counts[depth]

    # Ten independent visible roles exactly fit the ARC palette: background,
    # divider, four panel identities, and four legend colors.
    roles = random.sample(range(10), 10)
    bg, divider = roles[:2]
    panel_colors = roles[2:6]
    legend_top_down = roles[6:10]
    if not _semantic_checks(masks, counts, legend_top_down, panel_colors, bg):
        return None

    block_h = 2 * panel_h + 1
    block_w = 2 * panel_w + 1
    gutter = random.randint(1, 3)
    right_margin = random.randint(0, 2)
    width = block_w + gutter + 1 + right_margin
    grid = np.full((block_h, width), bg, dtype=int)

    origins = (
        (0, 0),
        (0, panel_w + 1),
        (panel_h + 1, 0),
        (panel_h + 1, panel_w + 1),
    )
    for panel_i, (r0, c0) in enumerate(origins):
        panel = grid[r0 : r0 + panel_h, c0 : c0 + panel_w]
        panel[masks[panel_i]] = panel_colors[panel_i]
    grid[panel_h, :block_w] = divider
    grid[:, panel_w] = divider

    legend_r0 = random.randint(0, block_h - 4)
    legend_c = block_w + gutter
    grid[legend_r0 : legend_r0 + 4, legend_c] = legend_top_down

    expected = _render_semantics(masks, counts, legend_top_down, bg)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying every solver assumption."""
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid count-ranked overlay after {_GEN_ATTEMPTS} attempts")
