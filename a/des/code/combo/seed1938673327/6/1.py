import random
from collections import deque

import numpy as np


BACKGROUND = 0
BORDER_COLOR = 4
PALETTE = tuple(color for color in range(10) if color not in (BACKGROUND, BORDER_COLOR))

GEN_RETRIES = 400
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (18, 28)
RULE_COUNT_RANGE = (3, 6)
TARGET_H_RANGE = (4, 9)
TARGET_W_RANGE = (5, 11)
LEGEND_GAP_RANGE = (2, 6)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_rules_to_color(color: int, rules: list[tuple[int, int]]) -> tuple[int, int]:
    current = color
    changes = 0
    for src, dst in rules:
        if current == src:
            current = dst
            changes += 1
    return current, changes


def _extract_parts(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    comps = _components4(g != BACKGROUND)
    assert len(comps) == 2, "expected exactly two foreground regions"

    infos = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": len(coords),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )

    legend_info = None
    for info in infos:
        if info["width"] == 2 and info["area"] == 2 * info["height"]:
            if legend_info is None or info["bbox"][1] < legend_info["bbox"][1]:
                legend_info = info
    assert legend_info is not None, "failed to find the 2-column legend"

    target_info = next(info for info in infos if info is not legend_info)
    tr0, tc0, tr1, tc1 = target_info["bbox"]
    target = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert target_info["area"] == target.size, "target region must be a filled rectangle"
    assert np.all(target != BACKGROUND), "target rectangle must be fully foreground"

    lr0, lc0, lr1, _ = legend_info["bbox"]
    legend = g[lr0 : lr1 + 1, lc0 : lc0 + 2]
    rules = [(int(row[0]), int(row[1])) for row in legend]
    assert all(src != BACKGROUND and dst != BACKGROUND for src, dst in rules)
    assert target_info["bbox"][1] > legend_info["bbox"][3], "target must lie to the right of the legend"
    return legend_info["bbox"], target_info["bbox"], rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the ordered color-substitution legend to the target rectangle, then add a yellow
    8-neighborhood border around pixels replaced an odd number of times.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_bbox, target_bbox, rules = _extract_parts(g)
    _ = legend_bbox

    tr0, tc0, tr1, tc1 = target_bbox
    target = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    out = g.copy()

    final_target = target.copy()
    odd_mask = np.zeros(target.shape, dtype=bool)
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            final_color, changes = _apply_rules_to_color(int(target[r, c]), rules)
            final_target[r, c] = final_color
            odd_mask[r, c] = (changes % 2) == 1

    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = final_target

    for r, c in np.argwhere(odd_mask):
        gr = tr0 + int(r)
        gc = tc0 + int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BACKGROUND:
                    out[nr, nc] = BORDER_COLOR
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_rules() -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]]]:
    n_rules = random.randint(*RULE_COUNT_RANGE)
    split = random.randint(1, n_rules - 1)
    lengths = [split, n_rules - split]
    colors = random.sample(PALETTE, n_rules + 2)

    chains = []
    idx = 0
    for length in lengths:
        chain = colors[idx : idx + length + 1]
        idx += length + 1
        chains.append(chain)

    rules: list[tuple[int, int]] = []
    order = [0, 1]
    random.shuffle(order)
    for chain_idx in order:
        chain = chains[chain_idx]
        rules.extend(list(zip(chain[:-1], chain[1:])))

    stats = {color: _apply_rules_to_color(color, rules) for color in PALETTE}
    odd_colors = [color for color, (_final, count) in stats.items() if count % 2 == 1]
    productive_even = [color for color, (_final, count) in stats.items() if count > 0 and count % 2 == 0]
    deep_colors = [color for color, (_final, count) in stats.items() if count >= 2]
    changed_finals = {final for final, count in stats.values() if count > 0}

    assert odd_colors
    assert productive_even
    assert deep_colors
    assert len(changed_finals) >= 2
    return rules, stats


def _choose_active_colors(stats: dict[int, tuple[int, int]]) -> tuple[list[int], int, int, int]:
    odd_colors = [color for color, (_final, count) in stats.items() if count % 2 == 1]
    productive_even = [color for color, (_final, count) in stats.items() if count > 0 and count % 2 == 0]
    deep_colors = [color for color, (_final, count) in stats.items() if count >= 2]
    zero_colors = [color for color, (_final, count) in stats.items() if count == 0]

    odd_color = random.choice(odd_colors)
    even_color = random.choice(productive_even)
    deep_color = random.choice(deep_colors)

    active: list[int] = []
    for color in (odd_color, even_color, deep_color):
        if color not in active:
            active.append(color)

    if zero_colors and random.random() < 0.7:
        zero_color = random.choice(zero_colors)
        if zero_color not in active:
            active.append(zero_color)

    finals = {stats[color][0] for color in active}
    for color in random.sample(list(PALETTE), len(PALETTE)):
        if len(finals) >= 2 and len(active) >= 3:
            break
        if color not in active:
            active.append(color)
            finals.add(stats[color][0])

    target_count = random.randint(3, min(6, len(PALETTE)))
    for color in random.sample(list(PALETTE), len(PALETTE)):
        if len(active) >= target_count:
            break
        if color not in active:
            active.append(color)

    return active, odd_color, even_color, deep_color


def _make_target(height: int, width: int, active_colors: list[int], odd_color: int, even_color: int, deep_color: int) -> np.ndarray:
    band_rows = random.randint(2, min(4, height))
    band_cols = random.randint(2, min(4, width))
    row_sizes = _random_partition(height, band_rows)
    col_sizes = _random_partition(width, band_cols)

    blocks = np.empty((band_rows, band_cols), dtype=int)
    all_positions = [(r, c) for r in range(band_rows) for c in range(band_cols)]
    boundary_positions = [
        (r, c)
        for r, c in all_positions
        if r == 0 or c == 0 or r == band_rows - 1 or c == band_cols - 1
    ]

    odd_pos = random.choice(boundary_positions)
    even_choices = [pos for pos in boundary_positions if pos != odd_pos]
    even_pos = random.choice(even_choices)

    assigned: dict[tuple[int, int], int] = {
        odd_pos: odd_color,
        even_pos: even_color,
    }
    if deep_color not in assigned.values():
        remaining = [pos for pos in all_positions if pos not in assigned]
        assigned[random.choice(remaining)] = deep_color

    for pos, color in assigned.items():
        blocks[pos] = color
    for r, c in all_positions:
        if (r, c) not in assigned:
            blocks[r, c] = random.choice(active_colors)

    target = np.full((height, width), BACKGROUND, dtype=int)
    r0 = 0
    for br, rh in enumerate(row_sizes):
        c0 = 0
        for bc, cw in enumerate(col_sizes):
            target[r0 : r0 + rh, c0 : c0 + cw] = blocks[br, bc]
            c0 += cw
        r0 += rh
    return target


def _validate_sample(grid: np.ndarray) -> bool:
    try:
        legend_bbox, target_bbox, rules = _extract_parts(grid)
        out = main(grid)
    except AssertionError:
        return False

    lr0, lc0, lr1, lc1 = legend_bbox
    tr0, tc0, tr1, tc1 = target_bbox
    if not np.array_equal(grid[lr0 : lr1 + 1, lc0 : lc1 + 1], out[lr0 : lr1 + 1, lc0 : lc1 + 1]):
        return False
    if np.array_equal(grid[tr0 : tr1 + 1, tc0 : tc1 + 1], out[tr0 : tr1 + 1, tc0 : tc1 + 1]):
        return False

    target = grid[tr0 : tr1 + 1, tc0 : tc1 + 1]
    visible_odd = False
    visible_even = False
    final_colors = set()
    for r in range(target.shape[0]):
        for c in range(target.shape[1]):
            final_color, changes = _apply_rules_to_color(int(target[r, c]), rules)
            final_colors.add(final_color)
            if r in (0, target.shape[0] - 1) or c in (0, target.shape[1] - 1):
                if changes % 2 == 1:
                    visible_odd = True
                elif changes > 0:
                    visible_even = True
    return visible_odd and visible_even and len(final_colors) >= 2


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rules, stats = _build_rules()
        active_colors, odd_color, even_color, deep_color = _choose_active_colors(stats)

        target_h = random.randint(*TARGET_H_RANGE)
        target_w = random.randint(*TARGET_W_RANGE)
        target = _make_target(target_h, target_w, active_colors, odd_color, even_color, deep_color)

        legend_h = len(rules)
        min_h = max(legend_h, target_h) + 2
        min_w = 2 + target_w + LEGEND_GAP_RANGE[0] + 3
        max_h = max(GRID_H_RANGE[1], min_h)
        max_w = max(GRID_W_RANGE[1], min_w)
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)

        legend_r0 = random.randint(1, h - legend_h - 1)
        legend_c0 = random.randint(1, min(3, w - target_w - 4))
        gap = random.randint(*LEGEND_GAP_RANGE)
        min_target_c0 = legend_c0 + 2 + gap
        max_target_c0 = w - target_w - 1
        if min_target_c0 > max_target_c0:
            continue
        target_r0 = random.randint(1, h - target_h - 1)
        target_c0 = random.randint(min_target_c0, max_target_c0)

        grid = np.full((h, w), BACKGROUND, dtype=int)
        legend = np.array(rules, dtype=int)
        grid[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + 2] = legend
        grid[target_r0 : target_r0 + target_h, target_c0 : target_c0 + target_w] = target

        if _validate_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid sample")
