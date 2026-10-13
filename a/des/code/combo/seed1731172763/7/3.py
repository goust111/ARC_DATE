import random
from collections import Counter

import numpy as np


BG_COLOR = 0
SEP_COLOR = 5
FOREGROUND_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
INTERIOR_MARGIN = 1

# Chosen so W + 1 = n * (side + 1) has a unique valid panel count among small divisors.
VALID_PANEL_SIDES = {
    4: (6, 10),
    5: (6, 8, 10),
    6: (6, 8, 10),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_layout(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int]] = []
    for panel_count in range(2, w + 1):
        if (w - (panel_count - 1)) % panel_count != 0:
            continue
        panel_w = (w - (panel_count - 1)) // panel_count
        if panel_w <= 0:
            continue

        sep_cols = [(i + 1) * panel_w + i for i in range(panel_count - 1)]
        if sep_cols and sep_cols[-1] >= w:
            continue
        if not all(np.all(grid[:, c] == grid[0, c]) for c in sep_cols):
            continue

        sep_colors = {int(grid[0, c]) for c in sep_cols}
        if len(sep_colors) != 1:
            continue
        sep = next(iter(sep_colors))
        if sep == bg:
            continue
        candidates.append((panel_count, panel_w, sep))

    assert candidates, "no valid separator layout found"

    square_candidates = [cand for cand in candidates if cand[1] == h]
    if square_candidates:
        candidates = square_candidates

    assert len(candidates) == 1, f"ambiguous panel layout: {candidates}"
    return candidates[0]


def _rotation_order(mask: np.ndarray) -> int:
    if mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask, 1)):
        return 4
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _render_panels(panels: list[np.ndarray], sep_color: int) -> np.ndarray:
    panel_h, panel_w = map(int, panels[0].shape)
    out = np.full((panel_h, len(panels) * panel_w + len(panels) - 1), BG_COLOR, dtype=int)
    for idx, panel in enumerate(panels):
        c0 = idx * (panel_w + 1)
        out[:, c0 : c0 + panel_w] = panel
        if idx < len(panels) - 1:
            out[:, c0 + panel_w] = sep_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort equal-sized panels by the rotational symmetry order of their internal pattern.

    Input structure:
    - Black background.
    - One row of equal-sized panels separated by full-height grey columns.
    - Each panel contains a monochrome pattern.

    Rule:
    - Compute each panel's rotational symmetry order on the panel mask:
      4 if invariant under 90 degrees, 2 if invariant under 180 degrees,
      otherwise 1.
    - Sort panels from high order to low order.
    - Keep the original left-to-right order for ties.
    - Reassemble the sorted panels with the same separator color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    panel_count, panel_w, sep = _find_layout(g, bg)

    panels = []
    orders = []
    for idx in range(panel_count):
        c0 = idx * (panel_w + 1)
        panel = g[:, c0 : c0 + panel_w].copy()
        panels.append(panel)
        orders.append(_rotation_order(panel != bg))

    sorted_indices = sorted(range(panel_count), key=lambda i: -orders[i])
    return _render_panels([panels[i] for i in sorted_indices], sep)


def _rot90_pos(side: int, r: int, c: int) -> tuple[int, int]:
    return c, side - 1 - r


def _rot180_pos(side: int, r: int, c: int) -> tuple[int, int]:
    return side - 1 - r, side - 1 - c


def _unique_orbits(side: int, orbit_order: int) -> list[list[tuple[int, int]]]:
    seen = set()
    orbits = []
    interior = range(INTERIOR_MARGIN, side - INTERIOR_MARGIN)
    for r in interior:
        for c in interior:
            orbit = {(r, c)}
            if orbit_order == 4:
                cur = (r, c)
                for _ in range(3):
                    cur = _rot90_pos(side, *cur)
                    orbit.add(cur)
            else:
                orbit.add(_rot180_pos(side, r, c))
            key = tuple(sorted(orbit))
            if key not in seen:
                seen.add(key)
                orbits.append(list(key))
    random.shuffle(orbits)
    return orbits


def _mask_from_orbits(side: int, chosen_orbits: list[list[tuple[int, int]]]) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    for orbit in chosen_orbits:
        for r, c in orbit:
            mask[r, c] = True
    return mask


def _make_order4_mask(side: int) -> np.ndarray:
    orbits = _unique_orbits(side, 4)
    max_pick = max(1, min(len(orbits) - 1, len(orbits) // 2 + 2))
    pick = random.randint(1, max_pick)
    return _mask_from_orbits(side, random.sample(orbits, pick))


def _make_order2_mask(side: int) -> np.ndarray:
    orbits = _unique_orbits(side, 2)
    max_pick = max(2, min(len(orbits) - 1, len(orbits) // 2 + 3))
    for _ in range(200):
        pick = random.randint(2, max_pick)
        mask = _mask_from_orbits(side, random.sample(orbits, pick))
        if _rotation_order(mask) == 2:
            return mask
    raise RuntimeError("failed to create an order-2 pattern")


def _make_order1_mask(side: int) -> np.ndarray:
    for _ in range(200):
        mask = _make_order2_mask(side)
        cells = [
            (r, c)
            for r in range(INTERIOR_MARGIN, side - INTERIOR_MARGIN)
            for c in range(INTERIOR_MARGIN, side - INTERIOR_MARGIN)
            if not mask[r, c]
        ]
        random.shuffle(cells)
        for r, c in cells:
            rr, cc = _rot180_pos(side, r, c)
            if mask[rr, cc]:
                continue
            mask2 = mask.copy()
            mask2[r, c] = True
            if _rotation_order(mask2) == 1:
                return mask2
    raise RuntimeError("failed to create an order-1 pattern")


def _make_panel(side: int, order: int, color: int) -> np.ndarray:
    panel = np.full((side, side), BG_COLOR, dtype=int)
    if order == 4:
        mask = _make_order4_mask(side)
    elif order == 2:
        mask = _make_order2_mask(side)
    else:
        mask = _make_order1_mask(side)
    panel[mask] = color
    return panel


def _positive_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(1, total - parts + 2):
        for rest in _positive_compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def _expected_output(panels: list[np.ndarray], orders: list[int]) -> np.ndarray:
    sorted_indices = sorted(range(len(panels)), key=lambda i: -orders[i])
    return _render_panels([panels[i] for i in sorted_indices], SEP_COLOR)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(4, 6)
    side = random.choice(VALID_PANEL_SIDES[panel_count])
    fg = random.choice(FOREGROUND_POOL)

    count4, count2, count1 = random.choice(_positive_compositions(panel_count, 3))
    orders = [4] * count4 + [2] * count2 + [1] * count1
    random.shuffle(orders)
    if orders == sorted(orders, reverse=True):
        return None

    panels = [_make_panel(side, order, fg) for order in orders]
    inp = _render_panels(panels, SEP_COLOR)
    out = _expected_output(panels, orders)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    seen_orders = Counter(_rotation_order(panel != BG_COLOR) for panel in panels)
    if set(seen_orders) != {1, 2, 4}:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
