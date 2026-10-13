from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)
GEN_RETRIES = 400
ORTH_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the compressed top color-band order as a hidden priority list that arbitrates competitive filling after lower-panel walls are erased.

    Input structure:
    - A full-width grey divider row splits the grid into a jagged horizontal-band top panel and a lower panel.
    - The top panel is fully tiled by 3-4 non-black colors whose row-majority sequence compresses to one ordered color list.
    - The lower panel contains black background, magenta wall pixels, and one singleton seed for each top-band color.

    Rule:
    - Read the top panel row by row, take each row's unique majority color, and compress adjacent duplicates to get the priority order from highest to lowest.
    - In the lower panel, first flood each original non-magenta region from its seed color, stopping at magenta walls and the grid boundary.
    - Erase all magenta walls to black, then for every remaining black connected component assign the highest-priority color among its colored 4-neighbors and fill that whole component.
    - Remove the top panel and divider entirely; output only the solved lower panel.

    Key insight:
    - The upper bands do not directly appear in the output; their compressed top-to-bottom order only decides which lower seed wins whenever erased walls merge competing territories.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(divider_rows) == 1, "expected exactly one full-width grey divider"
    divider = divider_rows[0]
    assert 0 < divider < g.shape[0] - 1, "divider must split non-empty top and bottom panels"

    top = g[:divider]
    bottom = g[divider + 1 :]
    priority = _extract_priority(top)
    return _solve_bottom(bottom, priority)


def _extract_priority(top: np.ndarray) -> list[int]:
    assert top.ndim == 2 and top.shape[0] >= 3
    majorities: list[int] = []
    for row in top:
        counts = Counter(int(v) for v in row)
        color, count = counts.most_common(1)[0]
        assert color not in (BLACK, GREY, MAGENTA), "top panel must be fully colored by seed colors"
        winners = [c for c, n in counts.items() if n == count]
        assert len(winners) == 1, "row-majority colors must be unambiguous"
        majorities.append(color)

    priority: list[int] = []
    for color in majorities:
        if not priority or priority[-1] != color:
            priority.append(color)
    assert 3 <= len(priority) <= 4, "expected 3-4 compressed color bands"
    assert len(set(priority)) == len(priority), "each band color should appear once in the compressed order"
    return priority


def _solve_bottom(bottom: np.ndarray, priority: list[int]) -> np.ndarray:
    h, w = bottom.shape
    seed_locs: dict[int, tuple[int, int]] = {}
    allowed = set(priority)
    for r in range(h):
        for c in range(w):
            color = int(bottom[r, c])
            if color in allowed:
                assert color not in seed_locs, "each priority color must appear as a single seed"
                seed_locs[color] = (r, c)
            else:
                assert color in (BLACK, MAGENTA), "lower panel may only contain black, magenta walls, and seed colors"

    assert set(seed_locs) == allowed, "seed colors must match the top-band colors exactly"
    rank = {color: idx for idx, color in enumerate(priority)}

    passable = bottom != MAGENTA
    region_ids = -np.ones((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or region_ids[r, c] >= 0:
                continue
            comp = _bfs_component(passable, (r, c), region_ids, len(components))
            components.append(comp)

    out = np.where(bottom == MAGENTA, MAGENTA, BLACK).astype(int)
    for comp in components:
        colors = {int(bottom[r, c]) for r, c in comp if int(bottom[r, c]) in allowed}
        assert len(colors) <= 1, "each original region must contain at most one seed"
        if not colors:
            continue
        fill = next(iter(colors))
        for r, c in comp:
            out[r, c] = fill

    out[out == MAGENTA] = BLACK
    zero_mask = out == BLACK
    zero_ids = -np.ones((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not zero_mask[r, c] or zero_ids[r, c] >= 0:
                continue
            comp = _bfs_component(zero_mask, (r, c), zero_ids, 0)
            neighbor_colors = set()
            for rr, cc in comp:
                for nr, nc in _neighbors4(rr, cc, h, w):
                    color = int(out[nr, nc])
                    if color in allowed:
                        neighbor_colors.add(color)
            if not neighbor_colors:
                continue
            winner = min(neighbor_colors, key=rank.get)
            for rr, cc in comp:
                out[rr, cc] = winner

    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ORTH_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bfs_component(mask: np.ndarray, start: tuple[int, int], labels: np.ndarray, label: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    q = deque([start])
    labels[start] = label
    comp = [start]
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and labels[nr, nc] < 0:
                labels[nr, nc] = label
                q.append((nr, nc))
                comp.append((nr, nc))
    return comp


def _random_thicknesses(parts: int, total: int, low: int, high: int) -> list[int]:
    while True:
        vals = [low] * parts
        remaining = total - low * parts
        for _ in range(remaining):
            vals[random.randrange(parts)] += 1
        if all(v <= high for v in vals):
            return vals


def _make_top_panel(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    thicknesses = _random_thicknesses(len(colors), height, 1, 3)
    base_rows: list[int] = []
    for color, thick in zip(colors, thicknesses):
        base_rows.extend([color] * thick)
    if len(base_rows) != height:
        return None

    grid = np.array(base_rows, dtype=int)[:, None] * np.ones((1, width), dtype=int)
    for boundary in range(height - 1):
        upper = int(base_rows[boundary])
        lower = int(base_rows[boundary + 1])
        if upper == lower:
            continue
        for target_row, source_color, target_color in (
            (boundary, upper, lower),
            (boundary + 1, lower, upper),
        ):
            max_notch = max(1, width // 4)
            for _ in range(random.randint(1, 3)):
                length = random.randint(1, max_notch)
                start = random.randint(0, width - length)
                if np.count_nonzero(grid[target_row] == source_color) - length <= width // 2:
                    continue
                grid[target_row, start : start + length] = target_color

    for row in range(height):
        band = int(base_rows[row])
        neighbors = []
        if row > 0:
            neighbors.append(int(base_rows[row - 1]))
        if row + 1 < height:
            neighbors.append(int(base_rows[row + 1]))
        if not neighbors:
            continue
        extra = random.randint(0, 2)
        for _ in range(extra):
            if np.count_nonzero(grid[row] == band) <= width // 2 + 1:
                break
            c = random.randrange(width)
            grid[row, c] = random.choice(neighbors)

    try:
        if _extract_priority(grid) != colors:
            return None
    except AssertionError:
        return None
    if np.array_equal(grid, np.array(base_rows, dtype=int)[:, None] * np.ones((1, width), dtype=int)):
        return None
    return grid


def _region_size(rect: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _adjacent_regions(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(rects):
        ar0, ar1, ac0, ac1 = a
        for j in range(i + 1, len(rects)):
            br0, br1, bc0, bc1 = rects[j]
            if ac1 + 2 == bc0 or bc1 + 2 == ac0:
                overlap = min(ar1, br1) - max(ar0, br0) + 1
                if overlap > 0:
                    pairs.append((i, j))
                    continue
            if ar1 + 2 == br0 or br1 + 2 == ar0:
                overlap = min(ac1, bc1) - max(ac0, bc0) + 1
                if overlap > 0:
                    pairs.append((i, j))
    return pairs


def _split_rectangles(height: int, width: int, wanted: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]] | None:
    rects = [(0, height - 1, 0, width - 1)]
    walls: list[tuple[str, int, int, int]] = []
    min_span = 2

    for _ in range(wanted - 1):
        choices: list[tuple[int, str]] = []
        for idx, (r0, r1, c0, c1) in enumerate(rects):
            if c1 - c0 + 1 >= min_span * 2 + 1:
                choices.append((idx, "V"))
            if r1 - r0 + 1 >= min_span * 2 + 1:
                choices.append((idx, "H"))
        if not choices:
            return None

        idx, orient = random.choice(choices)
        r0, r1, c0, c1 = rects.pop(idx)
        if orient == "V":
            cut = random.randint(c0 + min_span, c1 - min_span)
            rects.extend([(r0, r1, c0, cut - 1), (r0, r1, cut + 1, c1)])
            walls.append(("V", cut, r0, r1))
        else:
            cut = random.randint(r0 + min_span, r1 - min_span)
            rects.extend([(r0, cut - 1, c0, c1), (cut + 1, r1, c0, c1)])
            walls.append(("H", cut, c0, c1))

    rects.sort(key=lambda rect: (rect[2], rect[0], rect[3], rect[1]))
    return rects, walls


def _region_seed_cell(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = rect
    row_margin = 1 if r1 - r0 + 1 >= 3 else 0
    col_margin = 1 if c1 - c0 + 1 >= 3 else 0
    rr0, rr1 = r0 + row_margin, r1 - row_margin
    cc0, cc1 = c0 + col_margin, c1 - col_margin
    if rr0 > rr1:
        rr0, rr1 = r0, r1
    if cc0 > cc1:
        cc0, cc1 = c0, c1
    return random.randint(rr0, rr1), random.randint(cc0, cc1)


def _build_bottom_panel(height: int, width: int, priority: list[int], extra_region: bool) -> np.ndarray | None:
    wanted = len(priority) + int(extra_region)
    split = _split_rectangles(height, width, wanted)
    if split is None:
        return None
    rects, walls = split
    if len(rects) < len(priority):
        return None

    adjacency = _adjacent_regions(rects)
    if not adjacency:
        return None

    pair = random.choice(adjacency)
    seed_regions = {pair[0], pair[1]}
    remaining = [idx for idx in range(len(rects)) if idx not in seed_regions]
    random.shuffle(remaining)
    while len(seed_regions) < len(priority) and remaining:
        seed_regions.add(remaining.pop())
    if len(seed_regions) != len(priority):
        return None

    grid = np.zeros((height, width), dtype=int)
    for orient, fixed, start, end in walls:
        if orient == "V":
            grid[start : end + 1, fixed] = MAGENTA
        else:
            grid[fixed, start : end + 1] = MAGENTA

    regions = list(seed_regions)
    random.shuffle(regions)
    for color, region_idx in zip(random.sample(priority, len(priority)), regions):
        r, c = _region_seed_cell(rects[region_idx])
        grid[r, c] = color

    if not _bottom_is_informative(grid, priority, require_expansion=extra_region):
        return None
    return grid


def _bottom_is_informative(bottom: np.ndarray, priority: list[int], require_expansion: bool) -> bool:
    try:
        initial = _initial_region_fill(bottom, priority)
    except AssertionError:
        return False

    erased = initial.copy()
    erased[erased == MAGENTA] = BLACK
    zero_mask = erased == BLACK
    labels = -np.ones_like(erased, dtype=int)
    competition = False
    expansion = False
    for r in range(erased.shape[0]):
        for c in range(erased.shape[1]):
            if not zero_mask[r, c] or labels[r, c] >= 0:
                continue
            comp = _bfs_component(zero_mask, (r, c), labels, 0)
            neighbor_colors = set()
            touches_plain_bg = False
            for rr, cc in comp:
                if bottom[rr, cc] == BLACK:
                    touches_plain_bg = True
                for nr, nc in _neighbors4(rr, cc, *erased.shape):
                    color = int(erased[nr, nc])
                    if color in priority:
                        neighbor_colors.add(color)
            competition |= len(neighbor_colors) >= 2
            expansion |= touches_plain_bg and bool(neighbor_colors)

    if not competition:
        return False
    if require_expansion and not expansion:
        return False
    solved = _solve_bottom(bottom, priority)
    return len(set(int(v) for v in np.unique(solved)) - {BLACK}) >= 2


def _initial_region_fill(bottom: np.ndarray, priority: list[int]) -> np.ndarray:
    h, w = bottom.shape
    allowed = set(priority)
    out = np.where(bottom == MAGENTA, MAGENTA, BLACK).astype(int)
    passable = bottom != MAGENTA
    labels = -np.ones((h, w), dtype=int)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] >= 0:
                continue
            comp = _bfs_component(passable, (r, c), labels, next_label)
            next_label += 1
            colors = {int(bottom[rr, cc]) for rr, cc in comp if int(bottom[rr, cc]) in allowed}
            assert len(colors) <= 1
            if colors:
                fill = next(iter(colors))
                for rr, cc in comp:
                    out[rr, cc] = fill
    return out


def _generate_case() -> np.ndarray | None:
    n_colors = random.choice((3, 4))
    priority = random.sample(PALETTE, n_colors)
    top_h = random.randint(max(5, n_colors), min(8, 3 * n_colors))
    bottom_h = random.randint(7, 10)
    width = random.randint(13, 18)

    top = _make_top_panel(top_h, width, priority)
    if top is None:
        return None

    extra_region = bool(random.getrandbits(1))
    bottom = _build_bottom_panel(bottom_h, width, priority, extra_region)
    if bottom is None:
        return None

    grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    grid[:top_h] = top
    grid[top_h] = GREY
    grid[top_h + 1 :] = bottom

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != bottom.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid priority-fill ARC input")
