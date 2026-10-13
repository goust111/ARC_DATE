from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _is_bridge_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> bool:
    rows = {r for r, _ in coords}
    colors = {int(grid[r, c]) for r, c in coords}
    return (
        len(coords) >= 2
        and len(rows) == 1
        and len(colors) == 1
        and int(next(iter(colors))) not in (bg, MAROON)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore complementarily occluded mirror twins, then recolor bridge-colored recoveries to grey.

    Input structure:
    - The background is the most common color.
    - Two connected multicolor figures sit on the left and right sides of the grid and are horizontal mirrors of each other.
    - Some figure cells are replaced by maroon occluder pixels; each hidden cell is visible in the mirrored position of the opposite figure.
    - A separate one-cell-thick horizontal bridge line of one non-background, non-maroon color lies in the gap between the twins.

    Rule:
    - Identify the unique bridge line and the two non-bridge figure components.
    - For every maroon cell in one twin, copy the visible color from the horizontally mirrored cell of the opposite twin.
    - If a newly restored color matches the bridge-line color, recolor that restored cell to grey.
    - Remove the bridge line to background and leave all originally visible non-bridge figure pixels unchanged.

    Key insight:
    - Maroon marks erasures rather than payload; the opposite mirrored twin always reveals the hidden color, while the bridge line supplies the only parameter for the second recoloring pass.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    comps = _components(g != bg)
    assert len(comps) == 3, "expected exactly two twin components and one bridge line"

    bridge_candidates = [coords for coords in comps if _is_bridge_component(g, coords, bg)]
    assert len(bridge_candidates) == 1, "expected one isolated horizontal bridge line"
    bridge = bridge_candidates[0]
    bridge_color = int(g[bridge[0]])

    twins = [coords for coords in comps if coords is not bridge]
    assert len(twins) == 2
    twins.sort(key=lambda coords: _bbox(coords)[1])
    left_coords, right_coords = twins
    lr0, lc0, lr1, lc1 = _bbox(left_coords)
    rr0, rc0, rr1, rc1 = _bbox(right_coords)

    assert (lr0, lr1) == (rr0, rr1), "mirror twins must share the same row span"
    assert (lr1 - lr0) == (rr1 - rr0), "mirror twins must share the same height"
    assert (lc1 - lc0) == (rc1 - rc0), "mirror twins must share the same width"
    assert lc1 < rc0, "left twin must lie strictly left of the right twin"

    left_patch = g[lr0 : lr1 + 1, lc0 : lc1 + 1].copy()
    right_patch = g[rr0 : rr1 + 1, rc0 : rc1 + 1].copy()
    right_as_left = np.fliplr(right_patch)
    shape_mask = left_patch != bg
    assert np.array_equal(shape_mask, right_as_left != bg), "twins must share the same occupancy mask"

    left_out = left_patch.copy()
    right_out_as_left = right_as_left.copy()
    h, w = left_patch.shape
    for r in range(h):
        for c in range(w):
            if not shape_mask[r, c]:
                continue
            a = int(left_patch[r, c])
            b = int(right_as_left[r, c])
            assert not (a == MAROON and b == MAROON), "occlusions must be complementary"
            if a != MAROON and b != MAROON:
                assert a == b, "visible mirrored colors must agree"
                continue
            if a == MAROON:
                assert b not in (bg, MAROON)
                left_out[r, c] = GREY if b == bridge_color else b
            if b == MAROON:
                assert a not in (bg, MAROON)
                right_out_as_left[r, c] = GREY if a == bridge_color else a

    out = g.copy()
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = left_out
    out[rr0 : rr1 + 1, rc0 : rc1 + 1] = np.fliplr(right_out_as_left)
    for r, c in bridge:
        out[r, c] = bg
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    target = random.randint(max(h + w, int(area * 0.42)), max(h + w + 1, int(area * 0.72)))

    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(anchors)

    mask = np.zeros((h, w), dtype=bool)
    current = anchors[0]
    mask[current] = True
    for nxt in anchors[1:]:
        r, c = current
        while (r, c) != nxt:
            steps: list[tuple[int, int]] = []
            if r < nxt[0]:
                steps.append((r + 1, c))
            elif r > nxt[0]:
                steps.append((r - 1, c))
            if c < nxt[1]:
                steps.append((r, c + 1))
            elif c > nxt[1]:
                steps.append((r, c - 1))
            r, c = random.choice(steps)
            mask[r, c] = True
        current = nxt

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in zip(*np.where(mask)):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    return mask


def _colorize_mask(mask: np.ndarray) -> tuple[np.ndarray, int] | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    color_count = random.randint(2, min(4, len(coords)))
    palette = random.sample(SHAPE_COLORS, color_count)
    seed_coords = random.sample(coords, color_count)
    color_grid = np.full(mask.shape, BG, dtype=int)

    for r, c in coords:
        ranked: list[tuple[int, float, int]] = []
        for idx, (sr, sc) in enumerate(seed_coords):
            dist = abs(r - sr) + abs(c - sc)
            noise = random.random() * 0.15
            ranked.append((dist, noise, idx))
        _, _, idx = min(ranked)
        color_grid[r, c] = palette[idx]

    counts = {color: int(np.sum(color_grid == color)) for color in palette}
    bridge_candidates = [color for color, count in counts.items() if 2 <= count <= len(coords) - 2]
    if not bridge_candidates:
        return None
    bridge_color = random.choice(bridge_candidates)
    return color_grid, bridge_color


def _occlusion_states(color_grid: np.ndarray, mask: np.ndarray, bridge_color: int) -> dict[tuple[int, int], int] | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    bridge_cells = [pos for pos in coords if int(color_grid[pos]) == bridge_color]
    other_cells = [pos for pos in coords if int(color_grid[pos]) != bridge_color]
    if not bridge_cells or not other_cells:
        return None

    states = {pos: 0 for pos in coords}
    bridge_pick = random.choice(bridge_cells)
    other_pick = random.choice(other_cells)
    first_side = random.choice((1, 2))
    states[bridge_pick] = first_side
    states[other_pick] = 3 - first_side

    for pos in coords:
        if pos == bridge_pick or pos == other_pick:
            continue
        roll = random.random()
        if roll < 0.42:
            states[pos] = 0
        elif roll < 0.71:
            states[pos] = 1
        else:
            states[pos] = 2

    return states


def _render_case(
    h: int,
    w: int,
    top: int,
    left: int,
    gap: int,
    mask: np.ndarray,
    color_grid: np.ndarray,
    bridge_color: int,
    states: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    shape_h, shape_w = mask.shape
    right = left + shape_w + gap
    grid = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for r in range(shape_h):
        for c in range(shape_w):
            if not mask[r, c]:
                continue
            color = int(color_grid[r, c])
            state = states[(r, c)]
            rr = top + r
            lc = left + c
            rc = right + (shape_w - 1 - c)

            grid[rr, lc] = MAROON if state == 1 else color
            grid[rr, rc] = MAROON if state == 2 else color

            out[rr, lc] = GREY if state == 1 and color == bridge_color else color
            out[rr, rc] = GREY if state == 2 and color == bridge_color else color

    bridge_row = top + random.randrange(shape_h)
    bridge_start_min = left + shape_w + 1
    bridge_end_max = right - 2
    bridge_start = random.randint(bridge_start_min, bridge_end_max - 1)
    bridge_end = random.randint(bridge_start + 1, bridge_end_max)
    grid[bridge_row, bridge_start : bridge_end + 1] = bridge_color

    return grid, out


def _gen_attempt() -> np.ndarray | None:
    for _ in range(60):
        shape_h = random.randint(4, 8)
        shape_w = random.randint(4, 8)
        top = random.randint(1, 8)
        left = random.randint(1, 5)
        bottom = random.randint(1, 8)
        right_margin = random.randint(1, 5)
        max_gap = 30 - (left + shape_w + right_margin + shape_w)
        if max_gap < 4:
            continue
        gap = random.randint(4, min(12, max_gap))
        h = top + shape_h + bottom
        w = left + shape_w + gap + shape_w + right_margin
        if h > 30 or w > 30:
            continue

        mask = _random_connected_mask(shape_h, shape_w)
        colored = _colorize_mask(mask)
        if colored is None:
            continue
        color_grid, bridge_color = colored
        states = _occlusion_states(color_grid, mask, bridge_color)
        if states is None:
            continue

        grid, oracle = _render_case(h, w, top, left, gap, mask, color_grid, bridge_color, states)
        if _most_common_color(grid) != BG:
            continue

        comps = _components(grid != BG)
        if len(comps) != 3:
            continue
        bridge_like = [coords for coords in comps if _is_bridge_component(grid, coords, BG)]
        if len(bridge_like) != 1:
            continue

        if not np.any(grid == MAROON):
            continue
        if not np.any(oracle == GREY):
            continue
        if np.array_equal(grid, oracle):
            continue

        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, oracle):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-occlusion ARC input")
