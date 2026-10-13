from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox


FILL_COLOR = 9
BG = 0
COLOR_POOL = tuple(range(1, 9))
MAX_GENERATE_ATTEMPTS = 400
GRID_MIN = 24
GRID_MAX = 30
PANEL_COUNT_RANGE = (4, 6)
PANEL_HEIGHT_RANGE = (4, 6)
PANEL_WIDTH_RANGE = (4, 6)
SLOT_STEP = 7
SLOT_OFFSET = 1


def _edge_signature(panel: np.ndarray, side: str) -> tuple[int, ...]:
    col = 0 if side == "left" else int(panel.shape[1] - 1)
    return tuple(int(r) for r in range(int(panel.shape[0])) if int(panel[r, col]) != BG)


def _extract_panels(grid: np.ndarray) -> list[dict]:
    comps = components_mask(np.asarray(grid, dtype=int) != BG, connectivity=8)
    panels = []
    for coords in comps:
        b = bbox_coords(coords)
        panels.append({"bbox": b, "panel": crop_bbox(grid, b)})
    return panels


def _chain_order(panels: list[np.ndarray]) -> list[int]:
    n = len(panels)
    assert 4 <= n <= 6
    heights = {int(p.shape[0]) for p in panels}
    assert len(heights) == 1

    lefts = [_edge_signature(panel, "left") for panel in panels]
    rights = [_edge_signature(panel, "right") for panel in panels]
    assert all(sig for sig in lefts) and all(sig for sig in rights)
    assert len(set(lefts)) == n
    assert len(set(rights)) == n
    for i in range(n):
        assert lefts[i] != rights[i]

    left_to_idx = {sig: i for i, sig in enumerate(lefts)}
    succ: list[int | None] = [None] * n
    indeg = [0] * n
    outdeg = [0] * n
    for i, sig in enumerate(rights):
        j = left_to_idx.get(sig)
        if j is None:
            continue
        assert j != i
        succ[i] = j
        outdeg[i] = 1
        indeg[j] += 1

    starts = [i for i in range(n) if indeg[i] == 0]
    ends = [i for i in range(n) if outdeg[i] == 0]
    assert len(starts) == 1
    assert len(ends) == 1
    for i in range(n):
        if i in starts:
            assert outdeg[i] == 1
        elif i in ends:
            assert indeg[i] == 1
        else:
            assert indeg[i] == 1 and outdeg[i] == 1

    order = []
    cur = starts[0]
    seen = set()
    while cur is not None:
        assert cur not in seen
        seen.add(cur)
        order.append(cur)
        cur = succ[cur]
    assert len(order) == n
    return order


def _render_ordered_panels(ordered_panels: list[np.ndarray]) -> np.ndarray:
    total_h = int(sum(int(panel.shape[0]) for panel in ordered_panels))
    total_w = int(sum(int(panel.shape[1]) for panel in ordered_panels))
    out = np.full((total_h, total_w), FILL_COLOR, dtype=int)

    r0 = 0
    c0 = 0
    for panel in ordered_panels:
        h, w = map(int, panel.shape)
        out[r0 : r0 + h, c0 : c0 + w] = panel
        r0 += h
        c0 += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Chain scattered rectangular panels by exact edge-row matches, then diagonally concatenate them with chestnut filler.

    Input structure:
    - Black background surrounds 4-6 scattered, non-overlapping rectangular panels.
    - Each panel is a single 8-connected multi-color object with the same height but possibly different width.
    - The leftmost and rightmost columns of each panel contain sparse foreground pixels; those row offsets define the panel's edge signatures.

    Rule:
    - For every ordered pair of panels A,B, connect A -> B exactly when the foreground row offsets on A's right edge equal those on B's left edge.
    - These matches form one unique linear chain using every panel exactly once.
    - Place the chain diagonally: the first panel starts at (0,0), and each next panel starts at the previous panel's bottom-right plus (1,1), so the total canvas is the sum of panel heights by the sum of panel widths.
    - Fill every non-panel cell inside that diagonal staircase canvas with color 9, then crop to the occupied content.

    Key insight:
    - Input positions are irrelevant; only exact equality of left/right edge foreground row-offset sets determines the panel order, and that order alone fixes the diagonal staircase output.
    """
    g = np.asarray(grid, dtype=int)
    panels = [item["panel"] for item in _extract_panels(g)]
    order = _chain_order(panels)
    ordered_panels = [panels[i] for i in order]
    return _render_ordered_panels(ordered_panels)


def _all_signatures(height: int) -> list[tuple[int, ...]]:
    sigs = []
    for mask in range(1, (1 << int(height)) - 1):
        sig = tuple(int(r) for r in range(int(height)) if (mask >> int(r)) & 1)
        if 1 <= len(sig) < height:
            sigs.append(sig)
    return sigs


def _choose_signature_chain(height: int, n_panels: int) -> list[tuple[int, ...]]:
    pool = _all_signatures(height)
    for _ in range(300):
        seq = random.sample(pool, n_panels + 1)
        counts = [len(sig) for sig in seq]
        same_count_diff_shape = any(
            counts[i] == counts[j] and seq[i] != seq[j]
            for i in range(len(seq))
            for j in range(i + 1, len(seq))
        )
        if len(set(counts)) == 1:
            continue
        if not same_count_diff_shape:
            continue
        return seq
    raise RuntimeError("failed to build signature chain with counterexamples")


def _choose_widths(n_panels: int) -> list[int]:
    for _ in range(300):
        widths = []
        used = 0
        for i in range(n_panels):
            remaining = n_panels - i - 1
            low = PANEL_WIDTH_RANGE[0]
            high = min(PANEL_WIDTH_RANGE[1], 30 - used - PANEL_WIDTH_RANGE[0] * remaining)
            widths.append(random.randint(low, high))
            used += widths[-1]
        if len(set(widths)) == 1:
            continue
        if widths == sorted(widths) or widths == sorted(widths, reverse=True):
            continue
        return widths
    raise RuntimeError("failed to choose non-degenerate widths")


def _make_panel(
    height: int,
    width: int,
    left_sig: tuple[int, ...],
    right_sig: tuple[int, ...],
    palette: tuple[int, ...],
    phase: int,
) -> np.ndarray:
    panel = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(1, width - 1):
            idx = (2 * r + 3 * c + phase + (r % 2)) % len(palette)
            panel[r, c] = int(palette[idx])

    left_color = int(palette[phase % len(palette)])
    right_color = int(palette[(phase + 1) % len(palette)])
    for r in left_sig:
        panel[int(r), 0] = left_color
    for r in right_sig:
        panel[int(r), width - 1] = right_color
    return panel


def _candidate_slots(grid_h: int, grid_w: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    rows = list(range(SLOT_OFFSET, grid_h - max_h + 1, SLOT_STEP))
    cols = list(range(SLOT_OFFSET, grid_w - max_w + 1, SLOT_STEP))
    return [(int(r), int(c)) for r in rows for c in cols]


def _place_panels(panels: list[np.ndarray], grid_h: int, grid_w: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    max_h = max(int(panel.shape[0]) for panel in panels)
    max_w = max(int(panel.shape[1]) for panel in panels)
    slots = _candidate_slots(grid_h, grid_w, max_h, max_w)
    assert len(slots) >= len(panels)
    chosen_slots = random.sample(slots, len(panels))
    random.shuffle(chosen_slots)

    positions: list[tuple[int, int]] = []
    for panel, (r0, c0) in zip(panels, chosen_slots):
        h, w = map(int, panel.shape)
        assert r0 + h <= grid_h and c0 + w <= grid_w
        grid[r0 : r0 + h, c0 : c0 + w] = panel
        positions.append((int(r0), int(c0)))
    return grid, positions


def _order_pairs(panels: list[np.ndarray]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(_edge_signature(panel, "left"), _edge_signature(panel, "right")) for panel in panels]


def _is_simple_spatial_order(positions: list[tuple[int, int]]) -> bool:
    n = len(positions)
    target = list(range(n))
    reverse = target[::-1]
    orders = [
        sorted(range(n), key=lambda i: (positions[i][0], positions[i][1])),
        sorted(range(n), key=lambda i: (positions[i][1], positions[i][0])),
        sorted(range(n), key=lambda i: positions[i][0]),
        sorted(range(n), key=lambda i: positions[i][1]),
    ]
    return any(order == target or order == reverse for order in orders)


def _validate_case(grid: np.ndarray, ordered_panels: list[np.ndarray], positions: list[tuple[int, int]]) -> bool:
    if _is_simple_spatial_order(positions):
        return False

    extracted = [item["panel"] for item in _extract_panels(grid)]
    if len(extracted) != len(ordered_panels):
        return False

    try:
        recovered = [extracted[i] for i in _chain_order(extracted)]
    except AssertionError:
        return False

    if _order_pairs(recovered) != _order_pairs(ordered_panels):
        return False

    expected = _render_ordered_panels(ordered_panels)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if FILL_COLOR not in actual:
        return False
    if max(actual.shape) > 30 or max(grid.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_panels = random.randint(*PANEL_COUNT_RANGE)
    max_height = min(PANEL_HEIGHT_RANGE[1], 30 // n_panels)
    if max_height < PANEL_HEIGHT_RANGE[0]:
        return None
    height = random.randint(PANEL_HEIGHT_RANGE[0], max_height)
    widths = _choose_widths(n_panels)
    sig_chain = _choose_signature_chain(height, n_panels)

    ordered_panels = []
    for i in range(n_panels):
        color_count = random.choice((2, 3))
        palette = tuple(random.sample(COLOR_POOL, color_count))
        ordered_panels.append(
            _make_panel(height, widths[i], sig_chain[i], sig_chain[i + 1], palette, phase=i)
        )

    shuffled_panels = ordered_panels[:]
    random.shuffle(shuffled_panels)

    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    max_w = max(widths)
    if len(_candidate_slots(grid_h, grid_w, height, max_w)) < n_panels:
        return None

    grid, placed_positions = _place_panels(shuffled_panels, grid_h, grid_w)
    chain_positions = []
    for panel in ordered_panels:
        for idx, candidate in enumerate(shuffled_panels):
            if candidate is panel:
                chain_positions.append(placed_positions[idx])
                break
        else:
            return None

    if not _validate_case(grid, ordered_panels, chain_positions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
