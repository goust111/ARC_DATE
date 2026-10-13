import random
from collections import deque
from itertools import combinations, permutations

import numpy as np


PANEL_SIZE = 4
INPUT_SHAPE = (4, 19)
PANEL_STARTS = (0, 5, 10, 15)
SEPARATOR_COLS = (4, 9, 14)
PANEL_COLORS = (1, 2, 3, 4)
PANEL_COUNT = 4
ALL_CELLS = tuple((r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE))
MAX_GENERATION_ATTEMPTS = 500
WRONG_ORDERS = ((0, 1, 2, 3), (2, 3, 0, 1), (3, 2, 1, 0))

_RNG = random.Random()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _count_components4(mask: np.ndarray) -> int:
    """Count 4-connected True components in a small boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    return [grid[:, start : start + PANEL_SIZE] for start in PANEL_STARTS]


def _overlay_panels(panels: list[np.ndarray], order: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    filled = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for panel_index in order:
        panel = panels[panel_index]
        mask = (panel != 0) & (~filled)
        out[mask] = panel[mask]
        filled |= mask
    return out


def _render_input(panel_masks: list[np.ndarray], panel_colors: list[int]) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    for idx, start in enumerate(PANEL_STARTS):
        grid[:, start : start + PANEL_SIZE] = np.where(panel_masks[idx], panel_colors[idx], 0)
    return grid


def _arr_sig(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.int8)
    return arr.shape[0].to_bytes(1, "little") + arr.shape[1].to_bytes(1, "little") + arr.tobytes()


def _build_feasible_plans() -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    plans = []
    for counts in combinations(range(2, 11), 4):
        v0 = counts[0]
        for v1 in range(1, counts[1] + 1):
            h1 = counts[1] - v1
            if h1 > v0:
                continue
            for v2 in range(1, counts[2] + 1):
                h2 = counts[2] - v2
                if h2 > v0 + v1:
                    continue
                for v3 in range(1, counts[3] + 1):
                    h3 = counts[3] - v3
                    visible = (v0, v1, v2, v3)
                    hidden = (0, h1, h2, h3)
                    if h3 > v0 + v1 + v2:
                        continue
                    if not (10 <= sum(visible) <= 16):
                        continue
                    if sum(hidden) < 2:
                        continue
                    if sum(int(h > 0) for h in hidden[1:]) < 2:
                        continue
                    if v2 + v3 < 3:
                        continue
                    plans.append((counts, visible))
    return plans


FEASIBLE_PLANS = _build_feasible_plans()
VALID_LAYER_ORDERS = [
    order for order in permutations(range(PANEL_COUNT)) if order not in {(0, 1, 2, 3), (2, 3, 0, 1)}
]


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the four aligned panels after ranking them by foreground density.

    Input structure:
    - The input is a 4x19 grid split into four 4x4 panels by black separator columns at 4, 9, and 14.
    - Each panel uses black as background and contains exactly one foreground color.
    - The four foreground colors are distinct and collectively are blue, red, green, and yellow.

    Rule:
    - Count the non-black pixels in each 4x4 panel.
    - Sort the panels by that count from fewest to most; the sparsest panel is the front layer.
    - Overlay the four panels cell by cell in that front-to-back order, treating black as transparent.
    - Output the first visible non-black color at each aligned cell, or black if every panel is black there.

    Key insight:
    - Panel position does not determine visibility; the hidden layer order must be computed from each panel's own foreground count before overlaying.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, "expected a 4x19 grid of four 4x4 panels"
    assert np.all(g[:, SEPARATOR_COLS] == 0), "separator columns must be black"

    panels = _extract_panels(g)
    panel_colors = []
    panel_counts = []
    for panel in panels:
        colors = set(map(int, np.unique(panel))) - {0}
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        color = colors.pop()
        panel_colors.append(color)
        panel_counts.append(int(np.count_nonzero(panel)))
        assert color in PANEL_COLORS, "unexpected foreground color"
        assert 2 <= panel_counts[-1] <= 10, "panel density must stay within the task range"

    assert set(panel_colors) == set(PANEL_COLORS), "panel colors must be blue, red, green, yellow in some order"
    assert len(set(panel_counts)) == PANEL_COUNT, "panel foreground counts must be distinct"

    order = tuple(idx for _count, idx in sorted((panel_counts[idx], idx) for idx in range(PANEL_COUNT)))
    return _overlay_panels(panels, order)


def _sample_visible_sets(visible_counts: tuple[int, int, int, int]) -> list[set[tuple[int, int]]]:
    remaining = list(ALL_CELLS)
    _RNG.shuffle(remaining)
    visible_sets = []
    cursor = 0
    for count in visible_counts:
        chosen = set(remaining[cursor : cursor + count])
        cursor += count
        visible_sets.append(chosen)
    return visible_sets


def _build_layer_masks(
    counts_sorted: tuple[int, int, int, int],
    visible_counts: tuple[int, int, int, int],
) -> list[np.ndarray] | None:
    visible_sets = _sample_visible_sets(visible_counts)
    layer_sets = []
    overlap_positions = set()
    prefix_visible = set()

    for layer_rank, (target_area, visible_area) in enumerate(zip(counts_sorted, visible_counts)):
        visible_set = visible_sets[layer_rank]
        hidden_area = target_area - visible_area
        if hidden_area < 0:
            return None
        if hidden_area > len(prefix_visible):
            return None
        if hidden_area:
            hidden_set = set(_RNG.sample(tuple(prefix_visible), hidden_area))
            overlap_positions |= hidden_set
        else:
            hidden_set = set()
        layer_sets.append(visible_set | hidden_set)
        prefix_visible |= visible_set

    if len(overlap_positions) < 2:
        return None

    masks = []
    occupancy = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for layer_set, target_area in zip(layer_sets, counts_sorted):
        mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        for r, c in layer_set:
            mask[r, c] = True
            occupancy[r, c] += 1
        if int(mask.sum()) != target_area:
            return None
        components = _count_components4(mask)
        if not (1 <= components <= 5):
            return None
        masks.append(mask)

    if int(np.count_nonzero(occupancy >= 2)) < 2:
        return None
    return masks


def _semantic_checks(inp: np.ndarray, layer_order: tuple[int, int, int, int]) -> np.ndarray | None:
    try:
        out = main(inp)
    except Exception:
        return None

    panels = _extract_panels(inp)
    counts = [int(np.count_nonzero(panel)) for panel in panels]
    assert tuple(idx for _count, idx in sorted((counts[idx], idx) for idx in range(PANEL_COUNT))) == layer_order

    colors = set(map(int, out.ravel())) - {0}
    if len(colors) != 4:
        return None

    occupancy = sum((panel != 0).astype(int) for panel in panels)
    if int(np.count_nonzero(occupancy >= 2)) < 2:
        return None

    if layer_order in WRONG_ORDERS:
        return None

    for wrong_order in WRONG_ORDERS:
        if wrong_order == layer_order:
            continue
        wrong_out = _overlay_panels(panels, wrong_order)
        if np.array_equal(wrong_out, out):
            return None

    return out


def _gen_attempt() -> np.ndarray | None:
    counts_sorted, visible_counts = _RNG.choice(FEASIBLE_PLANS)
    layer_order = _RNG.choice(VALID_LAYER_ORDERS)
    layer_masks = _build_layer_masks(counts_sorted, visible_counts)
    if layer_masks is None:
        return None

    panel_masks = [None] * PANEL_COUNT
    for rank, panel_index in enumerate(layer_order):
        panel_masks[panel_index] = layer_masks[rank]

    panel_colors = list(PANEL_COLORS)
    _RNG.shuffle(panel_colors)
    inp = _render_input(panel_masks, panel_colors)
    out = _semantic_checks(inp, layer_order)
    if out is None:
        return None

    out_sig = _arr_sig(out)
    if out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid density-ordered overlay input")
