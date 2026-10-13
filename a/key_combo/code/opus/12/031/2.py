from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_opus_12_031_2",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    Color = _ARC_UTILS.Color


GEN_RETRIES = 500
INPUT_SEPARATOR_WIDTH = 1
PANEL_COUNT = 4
PANEL_HEIGHT_RANGE = (5, 8)
PANEL_WIDTH_RANGE = (4, 6)
BASE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
FG_COLOR_BY_BASE = {
    Color.BLUE: Color.PINK,
    Color.RED: Color.TEAL,
    Color.GREEN: Color.MAROON,
    Color.YELLOW: Color.RED,
}


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    """Convert sorted indices into inclusive contiguous bands."""
    indices = np.asarray(indices, dtype=int)
    assert indices.size > 0, "expected at least one index"
    bands: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in map(int, indices[1:]):
        if value != prev + 1:
            bands.append((start, prev))
            start = value
        prev = value
    bands.append((start, prev))
    return bands


def _panel_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    grey_cols = np.where(np.all(grid == Color.GREY, axis=0))[0]
    assert grey_cols.size > 0, "expected grey separator columns"
    sep_bands = _bands(grey_cols)
    panel_cols = np.where(~np.all(grid == Color.GREY, axis=0))[0]
    panels = _bands(panel_cols)
    assert len(panels) == PANEL_COUNT, f"expected {PANEL_COUNT} panels, got {len(panels)}"
    widths = {c1 - c0 + 1 for c0, c1 in panels}
    assert len(widths) == 1, "all panels must have equal width"
    sep_widths = {c1 - c0 + 1 for c0, c1 in sep_bands}
    assert len(sep_widths) == 1, "all separators must share one width"
    return panels, sep_widths.pop()


def _base_color(panel: np.ndarray) -> int:
    counts = Counter(map(int, panel[panel != Color.BLACK].ravel()))
    counts.pop(Color.GREY, None)
    assert counts, "panel must contain a non-black base color"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "panel base must be uniquely dominant"
    return int(ranked[0][0])


def _foreground_mask(panel: np.ndarray) -> np.ndarray:
    base = _base_color(panel)
    return (panel != base) & (panel != Color.BLACK)


def _overlay_pair(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    out = np.full(left_mask.shape, Color.BLACK, dtype=int)
    out[left_mask | right_mask] = Color.ORANGE
    return out


def _solve_from_panels(panels: list[np.ndarray], sep_width: int) -> np.ndarray:
    ranked = sorted(
        ((int(np.count_nonzero(_foreground_mask(panel))), idx, panel) for idx, panel in enumerate(panels)),
        key=lambda item: (item[0], item[1]),
    )
    sorted_panels = [panel for _count, _idx, panel in ranked]
    panel_h, panel_w = sorted_panels[0].shape
    out = np.full((panel_h, 2 * panel_w + sep_width), Color.BLACK, dtype=int)
    out[:, panel_w : panel_w + sep_width] = Color.GREY
    pair0 = _overlay_pair(_foreground_mask(sorted_panels[0]), _foreground_mask(sorted_panels[1]))
    pair1 = _overlay_pair(_foreground_mask(sorted_panels[2]), _foreground_mask(sorted_panels[3]))
    out[:, :panel_w] = pair0
    out[:, panel_w + sep_width :] = pair1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the four horizontal panels by foreground area, then overlay adjacent ranked pairs as orange unions.

    Input structure:
    - The grid contains four equal-sized rectangular panels laid out horizontally on a black background.
    - Consecutive full grey columns separate neighboring panels.
    - Each panel has one dominant non-black base color and a smaller set of non-base foreground pixels.

    Rule:
    - Split the grid into its four panels using the full grey separator columns.
    - Count foreground cells in each panel, where foreground means any non-black cell that is not that panel's dominant base color.
    - Reorder the panels by ascending foreground count, breaking ties by original left-to-right position.
    - Overlay panels 1+2 and 3+4 in the ranked order; output orange where either panel in a pair has foreground, black elsewhere, and place one grey separator band between the two output panels.

    Key insight:
    - Base colors and foreground colors are distractors: only each panel's foreground mask matters, and sorting those masks globally changes which panels are paired before the boolean union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "grid must be a non-empty 2D array"
    panels, sep_width = _panel_bands(g)
    crops = [g[:, c0 : c1 + 1] for c0, c1 in panels]
    heights = {panel.shape[0] for panel in crops}
    assert len(heights) == 1, "all panels must have equal height"
    return _solve_from_panels(crops, sep_width)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _partition(total: int, parts: int) -> list[int]:
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _random_mask(h: int, w: int, count: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(80):
        n_comp = 1 if count < 4 or random.random() < 0.75 else 2
        n_comp = min(n_comp, count)
        sizes = _partition(count, n_comp)
        random.shuffle(cells)
        seeds: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 2 for rr, cc in seeds):
                seeds.append(cell)
                if len(seeds) == n_comp:
                    break
        if len(seeds) != n_comp:
            continue

        mask = np.zeros((h, w), dtype=bool)
        occupied: set[tuple[int, int]] = set()
        ok = True
        for seed, size in zip(seeds, sizes):
            component = {seed}
            occupied.add(seed)
            mask[seed] = True
            while len(component) < size:
                frontier = {
                    nbr
                    for rr, cc in component
                    for nbr in _neighbors4(rr, cc, h, w)
                    if nbr not in occupied
                }
                if not frontier:
                    ok = False
                    break
                chosen = random.choice(tuple(frontier))
                component.add(chosen)
                occupied.add(chosen)
                mask[chosen] = True
            if not ok:
                break
        if ok and int(mask.sum()) == count:
            return mask
    return None


def _assemble_input(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.zeros((panel_h, PANEL_COUNT * panel_w + (PANEL_COUNT - 1) * INPUT_SEPARATOR_WIDTH), dtype=int)
    col = 0
    for idx, panel in enumerate(panels):
        grid[:, col : col + panel_w] = panel
        col += panel_w
        if idx != PANEL_COUNT - 1:
            grid[:, col : col + INPUT_SEPARATOR_WIDTH] = Color.GREY
            col += INPUT_SEPARATOR_WIDTH
    return grid


def _render_expected(sorted_masks: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = sorted_masks[0].shape
    out = np.full((panel_h, 2 * panel_w + INPUT_SEPARATOR_WIDTH), Color.BLACK, dtype=int)
    out[:, panel_w : panel_w + INPUT_SEPARATOR_WIDTH] = Color.GREY
    out[:, :panel_w] = _overlay_pair(sorted_masks[0], sorted_masks[1])
    out[:, panel_w + INPUT_SEPARATOR_WIDTH :] = _overlay_pair(sorted_masks[2], sorted_masks[3])
    return out


def _alt_output_from_order(masks: list[np.ndarray], order: list[int]) -> np.ndarray:
    return _render_expected([masks[idx] for idx in order])


def _sorted_pair_contributions(sorted_masks: list[np.ndarray]) -> bool:
    for left, right in ((sorted_masks[0], sorted_masks[1]), (sorted_masks[2], sorted_masks[3])):
        if not np.any(left & ~right):
            return False
        if not np.any(right & ~left):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    area = panel_h * panel_w
    max_fg = (area - 1) // 2
    if max_fg < 5:
        return None

    counts = sorted(random.sample(range(2, max_fg + 1), PANEL_COUNT))
    masks_by_rank: list[np.ndarray] = []
    for count in counts:
        mask = _random_mask(panel_h, panel_w, count)
        if mask is None:
            return None
        masks_by_rank.append(mask)

    if not _sorted_pair_contributions(masks_by_rank):
        return None

    order = list(range(PANEL_COUNT))
    random.shuffle(order)

    panel_bases = list(BASE_COLORS)
    random.shuffle(panel_bases)
    input_panels: list[np.ndarray] = []
    for input_idx, rank_idx in enumerate(order):
        base = panel_bases[input_idx]
        fg = FG_COLOR_BY_BASE[base]
        panel = np.full((panel_h, panel_w), base, dtype=int)
        panel[masks_by_rank[rank_idx]] = fg
        input_panels.append(panel)

    inp = _assemble_input(input_panels)
    expected = _render_expected(masks_by_rank)

    input_masks = [masks_by_rank[rank_idx] for rank_idx in order]
    original_output = _alt_output_from_order(input_masks, [0, 1, 2, 3])
    descending_output = _alt_output_from_order(masks_by_rank, [3, 2, 1, 0])

    if np.array_equal(expected, original_output):
        return None
    if np.array_equal(expected, descending_output):
        return None
    if np.array_equal(expected[:, :panel_w], expected[:, panel_w + INPUT_SEPARATOR_WIDTH :]):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
