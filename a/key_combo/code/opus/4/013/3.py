from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np


def _load_color_class():
    try:
        from arc_utils import Color as color_cls

        return color_cls
    except Exception:
        pass

    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if utils_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("_arc_utils_013_3", utils_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.Color
        except Exception:
            pass

    class _FallbackColor:
        BLACK = 0
        BLUE = 1
        RED = 2
        YELLOW = 4
        GREY = 5

    return _FallbackColor


Color = _load_color_class()
PANEL_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.YELLOW))


def _full_gray_line_positions(grid: np.ndarray) -> tuple[int, int, int]:
    gray = int(Color.GREY)
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == gray)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == gray)]
    assert len(full_rows) == 1, "expected exactly one full-width grey divider row"
    assert len(full_cols) == 2, "expected exactly two full-height grey divider columns"
    return int(full_rows[0]), int(full_cols[0]), int(full_cols[1])


def _panel_layout(grid: np.ndarray) -> tuple[int, int]:
    divider_r, divider_c0, divider_c1 = _full_gray_line_positions(grid)
    panel_h = divider_r
    panel_w = divider_c0
    assert grid.shape[0] == 2 * panel_h + 1, "top and bottom panel bands must have equal height"
    assert divider_c1 == 2 * panel_w + 1, "grey columns must split the width into three equal panels"
    assert grid.shape[1] == 3 * panel_w + 2, "expected three equal-width panels plus two divider columns"
    return panel_h, panel_w


def _panel_column_slice(index: int, panel_w: int) -> slice:
    start = index * (panel_w + 1)
    return slice(start, start + panel_w)


def _extract_panel_masks(grid: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    panel_h, panel_w = _panel_layout(grid)
    top = grid[:panel_h, :]
    bottom = grid[panel_h + 1 :, :]

    top_masks: dict[int, np.ndarray] = {}
    bottom_masks: dict[int, np.ndarray] = {}
    for idx, color in enumerate(PANEL_COLORS):
        cols = _panel_column_slice(idx, panel_w)
        top_patch = top[:, cols]
        bottom_patch = bottom[:, cols]
        top_masks[color] = top_patch == color
        bottom_masks[color] = bottom_patch == color
    return top_masks, bottom_masks


def _overlay_masks(masks: dict[int, np.ndarray], order: tuple[int, int, int]) -> np.ndarray:
    shape = next(iter(masks.values())).shape
    out = np.full(shape, int(Color.BLACK), dtype=int)
    for color in order:
        out[masks[color]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare color growth between upper and lower panel rows to choose the overlay depth.

    Input structure:
    - The grid has a black background with one full-width grey horizontal divider and
      two full-height grey vertical dividers, forming a 2x3 panel layout.
    - The three upper panels and the three lower panels are aligned by column; from
      left to right they contain blue, red, and yellow foreground pixels on black.
    - Each colored panel is read in its own local coordinates, while the grey dividers
      are only structural guides.

    Rule:
    - Count the foreground pixels of each color in its upper panel and in the aligned
      lower panel.
    - Compute the growth difference `lower_count - upper_count` for blue, red, and yellow.
    - Sort the colors by that difference from smallest to largest, then overlay the
      three lower panels onto one shared output canvas in that back-to-front order.

    Key insight:
    - The lower panels provide the visible content, but their occlusion order is not
      spatially encoded; it is hidden in the cross-band count comparison.
    """
    g = np.asarray(grid, dtype=int)
    top_masks, bottom_masks = _extract_panel_masks(g)
    growth = {
        color: int(bottom_masks[color].sum()) - int(top_masks[color].sum())
        for color in PANEL_COLORS
    }
    order = tuple(sorted(PANEL_COLORS, key=lambda color: growth[color]))
    return _overlay_masks(bottom_masks, order)


def _neighbors4(h: int, w: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_mask(
    h: int,
    w: int,
    target: int,
    rng: np.random.Generator,
    *,
    seeds: set[tuple[int, int]] | None = None,
    blocked: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    blocked = set() if blocked is None else set(blocked)
    cells = set() if seeds is None else set(seeds)
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    for cell in cells:
        assert cell not in blocked

    if not cells:
        available = [cell for cell in all_cells if cell not in blocked]
        assert available
        cells.add(available[int(rng.integers(len(available)))])

    while len(cells) < target:
        frontier = {
            nbr
            for cell in cells
            for nbr in _neighbors4(h, w, cell)
            if nbr not in cells and nbr not in blocked
        }
        if frontier:
            candidates = sorted(frontier)
        else:
            candidates = [cell for cell in all_cells if cell not in cells and cell not in blocked]
            assert candidates, "ran out of free cells while growing a panel mask"
        choice = candidates[int(rng.integers(len(candidates)))]
        cells.add(choice)

    return cells


def _sample_counts(area: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = 4
    high = min(9, area - 1)
    diffs_pool = np.array([d for d in range(-4, 5) if d != 0], dtype=int)
    for _ in range(2000):
        top = rng.integers(low, high + 1, size=3)
        if len(set(map(int, top))) != 3:
            continue
        diffs = np.array(rng.choice(diffs_pool, size=3, replace=False), dtype=int)
        if not (np.any(diffs < 0) and np.any(diffs > 0)):
            continue
        bottom = top + diffs
        if np.any(bottom < low) or np.any(bottom > high):
            continue
        if len(set(map(int, bottom))) != 3:
            continue
        if int(bottom.sum()) - 3 > area:
            continue

        growth_rank = tuple(np.argsort(diffs))
        if growth_rank in {
            tuple(np.argsort(top)),
            tuple(np.argsort(-top)),
            tuple(np.argsort(bottom)),
            tuple(np.argsort(-bottom)),
        }:
            continue
        return top.astype(int), bottom.astype(int), diffs.astype(int)

    raise RuntimeError("failed to sample non-degenerate panel counts")


def _make_top_masks(h: int, w: int, top_counts: np.ndarray, rng: np.random.Generator) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for color, count in zip(PANEL_COLORS, top_counts):
        coords = _grow_mask(h, w, int(count), rng)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        masks[color] = mask
    return masks


def _make_bottom_masks(h: int, w: int, bottom_counts: np.ndarray, rng: np.random.Generator) -> dict[int, np.ndarray]:
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    shared_coords = rng.choice(len(all_cells), size=3, replace=False)
    pair_cells = [all_cells[int(idx)] for idx in shared_coords]

    shared_by_color = {
        PANEL_COLORS[0]: {pair_cells[0], pair_cells[1]},
        PANEL_COLORS[1]: {pair_cells[0], pair_cells[2]},
        PANEL_COLORS[2]: {pair_cells[1], pair_cells[2]},
    }
    reserved_shared = set(pair_cells)
    used_unique: set[tuple[int, int]] = set()
    masks: dict[int, np.ndarray] = {}

    for color, count in zip(PANEL_COLORS, bottom_counts):
        seeds = set(shared_by_color[color])
        blocked = used_unique | (reserved_shared - seeds)
        coords = _grow_mask(h, w, int(count), rng, seeds=seeds, blocked=blocked)
        used_unique |= coords - seeds
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        masks[color] = mask
    return masks


def _assemble_input(
    top_masks: dict[int, np.ndarray],
    bottom_masks: dict[int, np.ndarray],
) -> np.ndarray:
    h, w = next(iter(top_masks.values())).shape
    grid = np.full((2 * h + 1, 3 * w + 2), int(Color.BLACK), dtype=int)
    grid[h, :] = int(Color.GREY)
    grid[:, w] = int(Color.GREY)
    grid[:, 2 * w + 1] = int(Color.GREY)

    for idx, color in enumerate(PANEL_COLORS):
        cols = _panel_column_slice(idx, w)
        top_view = grid[:h, cols]
        bottom_view = grid[h + 1 :, cols]
        top_view[top_masks[color]] = int(color)
        bottom_view[bottom_masks[color]] = int(color)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    top_masks: dict[int, np.ndarray],
    bottom_masks: dict[int, np.ndarray],
) -> bool:
    top_counts = np.array([int(top_masks[color].sum()) for color in PANEL_COLORS], dtype=int)
    bottom_counts = np.array([int(bottom_masks[color].sum()) for color in PANEL_COLORS], dtype=int)
    diffs = bottom_counts - top_counts

    if np.any(diffs == 0) or len(set(map(int, diffs))) != 3:
        return False
    if len(set(map(int, top_counts))) != 3 or len(set(map(int, bottom_counts))) != 3:
        return False

    growth_rank = tuple(np.argsort(diffs))
    forbidden_ranks = {
        tuple(np.argsort(top_counts)),
        tuple(np.argsort(-top_counts)),
        tuple(np.argsort(bottom_counts)),
        tuple(np.argsort(-bottom_counts)),
    }
    if growth_rank in forbidden_ranks:
        return False

    order = tuple(PANEL_COLORS[idx] for idx in growth_rank)
    expected = _overlay_masks(bottom_masks, order)
    if not np.array_equal(main(grid), expected):
        return False

    alt_orders = [
        tuple(colors)
        for colors in itertools.permutations(PANEL_COLORS)
        if tuple(colors) != order
    ]
    if not any(not np.array_equal(_overlay_masks(bottom_masks, alt), expected) for alt in alt_orders):
        return False

    pairwise_overlap = (
        (bottom_masks[PANEL_COLORS[0]] & bottom_masks[PANEL_COLORS[1]]).sum()
        + (bottom_masks[PANEL_COLORS[0]] & bottom_masks[PANEL_COLORS[2]]).sum()
        + (bottom_masks[PANEL_COLORS[1]] & bottom_masks[PANEL_COLORS[2]]).sum()
    )
    if int(pairwise_overlap) < 3:
        return False

    return True


def _generate_once(rng: np.random.Generator) -> np.ndarray | None:
    panel_h = int(rng.integers(4, 7))
    panel_w = int(rng.integers(4, 7))
    area = panel_h * panel_w

    top_counts, bottom_counts, _diffs = _sample_counts(area, rng)
    top_masks = _make_top_masks(panel_h, panel_w, top_counts, rng)
    bottom_masks = _make_bottom_masks(panel_h, panel_w, bottom_counts, rng)
    grid = _assemble_input(top_masks, bottom_masks)

    return grid if _semantic_checks(grid, top_masks, bottom_masks) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
