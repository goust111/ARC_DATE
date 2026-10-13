from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLANK = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

# 5x5 diamond outline, following the shared ARC (r, c) convention from arc_utils.
DIAMOND = (
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
)
OVERLAP_STEPS = ((1, 0), (0, 1), (1, 1))
MAX_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _table_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == BLANK)
    assert coords.size > 0, "expected a grey table with blanks"
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    assert r1 - r0 == c1 - c0, "grey bbox must be square"
    return box


def _shifted_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(int(r0 + dr), int(c0 + dc)) for dr, dc in DIAMOND]


def _candidate_translations(
    grid: np.ndarray, coords: np.ndarray, left_limit: int
) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in coords}
    min_r, min_c = coords.min(axis=0)
    H, W = grid.shape
    out: list[tuple[int, int]] = []
    for r0 in range(int(min_r) - 4, int(min_r) + 1):
        for c0 in range(int(min_c) - 4, int(min_c) + 1):
            cells = _shifted_cells((r0, c0))
            if any(r < 0 or c < 0 or r >= H or c >= left_limit for r, c in cells):
                continue
            cell_set = set(cells)
            if not pts.issubset(cell_set):
                continue
            values = [int(grid[r, c]) for r, c in cells]
            if any(v in (BG, BLANK) for v in values):
                continue
            out.append((int(r0), int(c0)))
    return out


def _detect_diamonds(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    _, table_c0, _, _ = _table_bbox(grid)
    left = grid[:, :table_c0]
    placements: dict[int, tuple[int, int]] = {}
    for color in sorted(int(v) for v in np.unique(left) if int(v) not in (BG, BLANK)):
        coords = np.argwhere(left == color)
        if int(coords.shape[0]) < 3:
            continue
        candidates = _candidate_translations(grid, coords, table_c0)
        if len(candidates) == 1:
            placements[int(color)] = candidates[0]
    assert 3 <= len(placements) <= 4, "expected 3-4 detectable diamonds"
    return placements


def _topological_cycle(
    grid: np.ndarray, placements: dict[int, tuple[int, int]]
) -> list[int]:
    colors = sorted(placements)
    above_to_below = {color: set() for color in colors}
    for lower, pos in placements.items():
        for r, c in _shifted_cells(pos):
            seen = int(grid[r, c])
            if seen in placements and seen != lower:
                above_to_below[seen].add(lower)

    indeg = {color: 0 for color in colors}
    for upper, lowers in above_to_below.items():
        for lower in lowers:
            indeg[lower] += 1

    order: list[int] = []
    ready = sorted(color for color, deg in indeg.items() if deg == 0)
    while ready:
        assert len(ready) == 1, "occlusion relation must define a unique total order"
        cur = ready.pop(0)
        order.append(cur)
        for nxt in sorted(above_to_below[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort()

    assert len(order) == len(colors), "occlusion graph must be acyclic"
    return order


def _table_pattern(cycle: list[int], side: int) -> np.ndarray:
    n = len(cycle)
    out = np.empty((side, side), dtype=int)
    for r in range(side):
        for c in range(side):
            idx = c % n if r <= c else (c + 1) % n
            out[r, c] = int(cycle[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the occlusion order of overlapping diamonds and use it as the cyclic legend to fill a sparse diagonal-phase table.

    Input structure:
    - Black background surrounds two disjoint visual regions with no explicit separator.
    - On the left, 3-4 colored 5x5 diamond-outline templates are partially overlapped so only upper layers stay visible; small off-template single-pixel distractors may also appear.
    - On the right, a square grey-backed table contains some visible colored cells and many grey blanks, using the same color set as the diamonds.

    Rule:
    - Identify the left-side diamond colors by finding color sets whose visible pixels fit exactly inside one translated diamond template, ignoring isolated off-template noise.
    - For each diamond, any template cell currently occupied by another diamond color means that other color lies above it; topologically sort these relations to obtain the unique top-to-bottom layer order.
    - Treat that top-to-bottom order as a cycle. Inside the right square table, with local coordinates (r, c), the intended color is cycle[c mod N] when r <= c and cycle[(c + 1) mod N] when r > c.
    - Replace only grey table cells with the inferred colors and preserve every other cell.

    Key insight:
    - The sparse table does not uniquely reveal the color cycle by itself; the left occlusion stack provides the missing cyclic order, turning a reconstruct-from-fragments problem into a cross-region legend lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    placements = _detect_diamonds(g)
    cycle = _topological_cycle(g, placements)

    r0, c0, r1, c1 = _table_bbox(g)
    side = r1 - r0 + 1
    full = _table_pattern(cycle, side)

    out = g.copy()
    region = out[r0 : r1 + 1, c0 : c1 + 1]
    mask = region == BLANK
    region[mask] = full[mask]
    return out


def _sample_positions(n: int) -> list[tuple[int, int]]:
    for _ in range(200):
        pos = [(0, 0)]
        for _ in range(n - 1):
            dr, dc = random.choice(OVERLAP_STEPS)
            nr, nc = pos[-1][0] + dr, pos[-1][1] + dc
            pos.append((nr, nc))

        if random.random() < 0.5:
            max_r = max(r for r, _ in pos)
            pos = [(max_r - r, c) for r, c in pos]
        if random.random() < 0.5:
            max_c = max(c for _, c in pos)
            pos = [(r, max_c - c) for r, c in pos]
        if random.random() < 0.5:
            pos = list(reversed(pos))

        min_r = min(r for r, _ in pos)
        min_c = min(c for _, c in pos)
        pos = [(r - min_r, c - min_c) for r, c in pos]
        if max(r for r, _ in pos) <= 4 and max(c for _, c in pos) <= 4:
            return pos
    raise RuntimeError("failed to sample overlapping diamond placements")


def _render_left_patch(
    cycle: list[int], positions: list[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, int]]:
    max_r = max(r for r, _ in positions) + 5
    max_c = max(c for _, c in positions) + 5
    patch = np.zeros((max_r, max_c), dtype=int)
    visible_counts = {color: 0 for color in cycle}
    for color, pos in reversed(list(zip(cycle, positions))):
        for r, c in _shifted_cells(pos):
            patch[r, c] = int(color)
    for color in cycle:
        visible_counts[color] = int(np.sum(patch == color))
    return patch, visible_counts


def _place_noise(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    left_limit: int,
    forbidden_colors: set[int],
) -> None:
    noise_colors = [c for c in PALETTE if c not in forbidden_colors]
    random.shuffle(noise_colors)
    target = random.randint(0, min(3, len(noise_colors)))
    placed = 0
    H, _ = grid.shape
    attempts = 0
    while placed < target and attempts < 200:
        attempts += 1
        r = random.randrange(H)
        c = random.randrange(left_limit)
        if grid[r, c] != BG:
            continue
        ok = True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        color = noise_colors[placed]
        grid[r, c] = int(color)
        occupied.add((r, c))
        placed += 1


def _masked_table(cycle: list[int], side: int) -> tuple[np.ndarray, set[int]]:
    full = _table_pattern(cycle, side)
    masked = np.full((side, side), BLANK, dtype=int)
    reveal_count = 1 if len(cycle) == 3 else 2
    reveal_indices = set(random.sample(range(len(cycle)), reveal_count))

    for idx in sorted(reveal_indices):
        top_cols = [c for c in range(1, side - 1) if c % len(cycle) == idx]
        if top_cols:
            keep = top_cols if random.random() < 0.5 else random.sample(top_cols, random.randint(1, len(top_cols)))
            for c in keep:
                masked[0, c] = full[0, c]

        lower_cols = [c for c in range(0, side - 1) if (c + 1) % len(cycle) == idx]
        if lower_cols:
            keep = (
                lower_cols
                if random.random() < 0.5
                else random.sample(lower_cols, random.randint(1, len(lower_cols)))
            )
            for c in keep:
                masked[c + 1, c] = full[c + 1, c]

    masked[0, 0] = BLANK
    masked[0, side - 1] = BLANK
    masked[side - 1, 0] = BLANK
    masked[side - 1, side - 1] = BLANK
    assert np.mean(masked == BLANK) >= 0.4
    assert np.any(masked != BLANK)
    return masked, reveal_indices


def _cycle_matches(table: np.ndarray, cycle: tuple[int, ...]) -> bool:
    full = _table_pattern(list(cycle), int(table.shape[0]))
    visible = table != BLANK
    return bool(np.array_equal(table[visible], full[visible]))


def _compatible_cycles(table: np.ndarray, colors: list[int]) -> list[tuple[int, ...]]:
    out = []
    for cycle in itertools.permutations(colors):
        if _cycle_matches(table, cycle):
            out.append(tuple(int(v) for v in cycle))
    return out


def _build_candidate() -> np.ndarray | None:
    n = random.choice((3, 4))
    colors = random.sample(PALETTE, n)
    cycle = random.sample(colors, n)  # top -> bottom
    positions = _sample_positions(n)
    left_patch, visible_counts = _render_left_patch(cycle, positions)
    if min(visible_counts.values()) < 3:
        return None

    side = random.randint(8, 11)
    table, _reveal_indices = _masked_table(cycle, side)

    left_h, left_w = map(int, left_patch.shape)
    H = max(left_h, side) + random.randint(4, 8)
    left_r = random.randint(1, H - left_h - 1)
    table_r = random.randint(1, H - side - 1)
    left_c = random.randint(1, 2)
    gap = random.randint(2, 4)
    table_c = left_c + left_w + gap
    W = table_c + side + random.randint(1, 2)
    if H > 30 or W > 30:
        return None

    grid = np.zeros((H, W), dtype=int)
    for r in range(left_h):
        for c in range(left_w):
            if left_patch[r, c] != BG:
                grid[left_r + r, left_c + c] = int(left_patch[r, c])
    grid[table_r : table_r + side, table_c : table_c + side] = table

    occupied = {
        (left_r + r, left_c + c)
        for r in range(left_h)
        for c in range(left_w)
        if left_patch[r, c] != BG
    }
    _place_noise(grid, occupied, table_c, set(colors) | {BLANK, BG})

    try:
        placements = _detect_diamonds(grid)
        inferred_cycle = _topological_cycle(grid, placements)
    except Exception:
        return None
    if inferred_cycle != cycle:
        return None

    diamond_colors = sorted(placements)
    if diamond_colors != sorted(colors):
        return None

    for color, pos in placements.items():
        coords = np.argwhere(grid[:, :table_c] == color)
        if len(_candidate_translations(grid, coords, table_c)) != 1:
            return None
        cells = _shifted_cells(pos)
        if any(grid[r, c] in (BG, BLANK) for r, c in cells):
            return None

    table_cycles = _compatible_cycles(table, colors)
    if tuple(cycle) not in table_cycles or len(table_cycles) < 2:
        return None

    solved = main(grid)
    expected = grid.copy()
    fill = _table_pattern(cycle, side)
    region = expected[table_r : table_r + side, table_c : table_c + side]
    mask = region == BLANK
    region[mask] = fill[mask]
    if not np.array_equal(solved, expected):
        return None

    alternatives = [cy for cy in table_cycles if cy != tuple(cycle)]
    if not alternatives:
        return None
    alt_fill = _table_pattern(list(alternatives[0]), side)
    if np.array_equal(alt_fill, fill):
        return None
    if np.array_equal(alt_fill[table == BLANK], fill[table == BLANK]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-cycle ARC input")
