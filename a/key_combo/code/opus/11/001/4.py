from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PANEL_COLORS = (BLUE, RED, GREEN, YELLOW)
MAX_RETRIES = 500


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _neighbors4(cur[0], cur[1], h, w):
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _grow_connected(
    allowed: set[tuple[int, int]],
    seeds: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    if not seeds or not seeds.issubset(allowed):
        return None
    cells = set(seeds)
    while len(cells) < target_size:
        frontier = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c, 100, 100):
                if (rr, cc) in allowed and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            return None
        scored = []
        for cell in frontier:
            support = sum((nbr in cells) for nbr in _neighbors4(cell[0], cell[1], 100, 100))
            scored.append((support, random.random(), cell))
        scored.sort(reverse=True)
        take_count = min(len(scored), 4)
        cells.add(scored[random.randrange(take_count)][2])
    return cells if _is_connected(cells) else None


def _parse_input(grid: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    assert len(sep_rows) == 1, "expected exactly one horizontal separator row"
    sep_row = sep_rows[0]
    assert 2 <= sep_row < h - 2, "separator row must leave both instruction and panel areas"

    top = g[:sep_row, :]
    thumb_mask = np.isin(top, PANEL_COLORS)
    assert int(thumb_mask.sum()) == 4, "expected exactly four thumbnail pixels"
    thumb_box = _bbox_from_mask(thumb_mask)
    assert thumb_box is not None
    r0, c0, r1, c1 = thumb_box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "thumbnail must be a 2x2 block"
    thumb = top[r0 : r1 + 1, c0 : c1 + 1]
    seq = [int(v) for v in thumb.reshape(-1)]
    assert sorted(seq) == [1, 2, 3, 4], "thumbnail must contain blue/red/green/yellow exactly once"

    lower = g[sep_row + 1 :, :]
    sep_cols = [c for c in range(w) if np.all(lower[:, c] == GREY)]
    assert len(sep_cols) == 3, "expected three vertical separator columns"
    bounds = [-1] + sep_cols + [w]
    widths = [bounds[i + 1] - bounds[i] - 1 for i in range(4)]
    assert len(set(widths)) == 1 and widths[0] >= 2, "panels must have equal positive width"

    panels = []
    for i, color in enumerate(PANEL_COLORS):
        c_start = bounds[i] + 1
        c_end = bounds[i + 1]
        panel = lower[:, c_start:c_end]
        assert panel.shape == (lower.shape[0], widths[0])
        extras = set(map(int, np.unique(panel))) - {BACKGROUND}
        assert extras <= {color}, f"panel {i + 1} must use only its designated color"
        mask = panel == color
        assert 5 <= int(mask.sum()) <= 12, "each panel must have 5-12 foreground cells"
        panels.append(mask)
    return seq, panels


def _simulate_order(order: tuple[int, ...] | list[int], masks: dict[int, np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    sample = next(iter(masks.values()))
    out = np.zeros(sample.shape, dtype=int)
    accepted: list[int] = []
    for color in order:
        mask = masks[int(color)]
        if not np.any(out[mask] != BACKGROUND):
            out[mask] = int(color)
            accepted.append(int(color))
    return out, tuple(accepted)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the 2x2 corner thumbnail as a panel-priority schedule and overlay only non-overlapping panels.

    Input structure:
    - Black background with a top instruction area above one solid grey horizontal separator row.
    - The instruction area contains exactly one 2x2 thumbnail made of blue, red, green, and yellow, one pixel each.
    - Below the separator, three solid grey separator columns split the space into four equal panels from left to right.
    - Panel 1 uses blue, panel 2 red, panel 3 green, and panel 4 yellow; each panel contains one foreground mask on black.

    Rule:
    - Read the thumbnail in row-major order: top-left, top-right, bottom-left, bottom-right.
    - Start with an empty black canvas the size of one panel.
    - For each color in that order, look up the correspondingly colored panel mask.
    - If that mask would overlap any already placed non-black pixel on the canvas, skip it; otherwise copy its colored pixels onto the canvas.

    Key insight:
    - The thumbnail is not a spatial minimap of the panels; it is a time-order code that decides which masks get first claim on contested cells.
    """
    seq, panels = _parse_input(grid)
    masks = {color: panel for color, panel in zip(PANEL_COLORS, panels)}
    out, _ = _simulate_order(tuple(seq), masks)
    return out


def _make_conflict_masks(panel_h: int, panel_w: int) -> list[np.ndarray] | None:
    for _ in range(200):
        free_side = random.choice(("left", "right"))
        if free_side == "right":
            free_band = {panel_w - 2, panel_w - 1}
            main_cols = [c for c in range(panel_w) if c not in free_band]
            hub_col = max(main_cols)
            side_col = min(main_cols)
        else:
            free_band = {0, 1}
            main_cols = [c for c in range(panel_w) if c not in free_band]
            hub_col = min(main_cols)
            side_col = max(main_cols)

        top_overlap_r = random.randint(2, panel_h // 2 - 1)
        bottom_overlap_r = random.randint(panel_h // 2 + 1, panel_h - 3)
        if bottom_overlap_r - top_overlap_r < 2:
            continue

        hub = {(r, hub_col) for r in range(top_overlap_r, bottom_overlap_r + 1)}
        branch_pool = [
            (top_overlap_r, side_col),
            ((top_overlap_r + bottom_overlap_r) // 2, side_col),
            (bottom_overlap_r, side_col),
            (min(bottom_overlap_r, top_overlap_r + 1), side_col),
            (max(top_overlap_r, bottom_overlap_r - 1), side_col),
        ]
        random.shuffle(branch_pool)
        hub_target = random.randint(5, min(9, len(hub) + len(branch_pool)))
        for cell in branch_pool:
            if len(hub) >= hub_target:
                break
            hub.add(cell)

        top_overlap = (top_overlap_r, hub_col)
        bottom_overlap = (bottom_overlap_r, hub_col)

        top_allowed = {
            (r, c)
            for r in range(0, top_overlap_r + 1)
            for c in main_cols
            if (r, c) not in (hub - {top_overlap})
        }
        bottom_allowed = {
            (r, c)
            for r in range(bottom_overlap_r, panel_h)
            for c in main_cols
            if (r, c) not in (hub - {bottom_overlap})
        }
        free_allowed = {
            (r, c)
            for r in range(1, panel_h - 1)
            for c in free_band
        }

        top_size = random.randint(5, min(8, len(top_allowed)))
        bottom_size = random.randint(5, min(8, len(bottom_allowed)))
        free_size = random.randint(5, min(8, len(free_allowed)))
        top_mask = _grow_connected(top_allowed, {top_overlap}, top_size)
        bottom_mask = _grow_connected(bottom_allowed, {bottom_overlap}, bottom_size)
        free_seed = {(panel_h // 2, min(free_band)), (panel_h // 2, max(free_band))}
        free_seed = {cell for cell in free_seed if cell in free_allowed}
        free_seed = {random.choice(tuple(free_seed))} if free_seed else {(panel_h // 2, min(free_band))}
        free_mask = _grow_connected(free_allowed, free_seed, free_size)
        if top_mask is None or bottom_mask is None or free_mask is None:
            continue
        if not _is_connected(hub):
            continue

        masks = []
        for cells in (hub, top_mask, bottom_mask, free_mask):
            mask = np.zeros((panel_h, panel_w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if not (5 <= int(mask.sum()) <= 12):
                break
            masks.append(mask)
        else:
            overlap_counts = []
            for i in range(4):
                for j in range(i + 1, 4):
                    overlap_counts.append(int(np.sum(masks[i] & masks[j])))
            positive_pairs = sum(v > 0 for v in overlap_counts)
            if positive_pairs != 2:
                continue
            if np.sum(masks[1] & masks[2]) != 0:
                continue
            if np.sum(masks[0] & masks[3]) != 0 or np.sum(masks[1] & masks[3]) != 0 or np.sum(masks[2] & masks[3]) != 0:
                continue
            return masks
    return None


def _apply_symmetry(masks: list[np.ndarray]) -> list[np.ndarray]:
    op = random.choice(("id", "flipud", "fliplr", "both"))
    transformed = []
    for mask in masks:
        arr = mask
        if op in ("flipud", "both"):
            arr = np.flipud(arr)
        if op in ("fliplr", "both"):
            arr = np.fliplr(arr)
        transformed.append(arr.copy())
    return transformed


def _choose_sequence(masks_by_color: dict[int, np.ndarray], desired_count: int) -> tuple[int, ...] | None:
    natural_out, _ = _simulate_order(PANEL_COLORS, masks_by_color)
    perm_data = []
    for seq in itertools.permutations(PANEL_COLORS):
        out, accepted = _simulate_order(seq, masks_by_color)
        perm_data.append((seq, out, accepted))
    counts = Counter(len(accepted) for _, _, accepted in perm_data)
    if not ({2, 3} <= set(counts)):
        return None

    candidates = []
    for seq, out, accepted in perm_data:
        if len(accepted) != desired_count:
            continue
        a, b, c, d = seq
        col_major = (a, c, b, d)
        alt_out_1, _ = _simulate_order(col_major, masks_by_color)
        if np.array_equal(out, natural_out):
            continue
        if np.array_equal(out, alt_out_1):
            continue
        candidates.append(seq)
    if not candidates:
        return None
    return random.choice(candidates)


def _assemble_input(sequence: tuple[int, ...], panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.zeros((panel_h + 4, panel_w * 4 + 3), dtype=int)
    grid[3, :] = GREY

    c0 = 0
    for i, (color, mask) in enumerate(zip(PANEL_COLORS, panels)):
        block = np.zeros_like(mask, dtype=int)
        block[mask] = color
        grid[4:, c0 : c0 + panel_w] = block
        c0 += panel_w
        if i < 3:
            grid[4:, c0] = GREY
            c0 += 1

    thumb = np.array(sequence, dtype=int).reshape(2, 2)
    grid[0:2, 0:2] = thumb
    return grid


def _validate_case(inp: np.ndarray) -> bool:
    seq, panels = _parse_input(inp)
    masks_by_color = {color: panel for color, panel in zip(PANEL_COLORS, panels)}
    expected, accepted = _simulate_order(tuple(seq), masks_by_color)
    if not np.array_equal(main(inp), expected):
        return False

    conflict_pairs = 0
    for i in range(4):
        for j in range(i + 1, 4):
            if np.any(panels[i] & panels[j]):
                conflict_pairs += 1
    if conflict_pairs < 2 or conflict_pairs == 6:
        return False

    all_counts = set()
    outputs = set()
    for order in itertools.permutations(PANEL_COLORS):
        out, acc = _simulate_order(order, masks_by_color)
        all_counts.add(len(acc))
        outputs.add((out.shape, out.tobytes()))
    if not ({2, 3} <= all_counts):
        return False
    if len(outputs) < 2:
        return False
    if len(accepted) not in {2, 3}:
        return False

    natural_out, _ = _simulate_order(PANEL_COLORS, masks_by_color)
    row_major = tuple(seq)
    col_major = (row_major[0], row_major[2], row_major[1], row_major[3])
    col_out, _ = _simulate_order(col_major, masks_by_color)
    if np.array_equal(expected, natural_out):
        return False
    if np.array_equal(expected, col_out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        panel_h = random.randint(8, 12)
        panel_w = random.randint(4, 5)
        base_masks = _make_conflict_masks(panel_h, panel_w)
        if base_masks is None:
            continue
        base_masks = _apply_symmetry(base_masks)

        desired_count = random.choice((2, 3))
        role_orders = list(itertools.permutations(range(4)))
        random.shuffle(role_orders)
        if desired_count == 3:
            role_orders.sort(key=lambda order: 0 if order[0] == 0 else 1)

        for roles in role_orders:
            panels = [base_masks[idx] for idx in roles]
            masks_by_color = {color: mask for color, mask in zip(PANEL_COLORS, panels)}
            sequence = _choose_sequence(masks_by_color, desired_count)
            if sequence is None:
                continue
            inp = _assemble_input(sequence, panels)
            if max(inp.shape) > 30:
                continue
            if _validate_case(inp):
                return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
