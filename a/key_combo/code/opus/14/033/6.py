from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COUNTER_COLOR = 5
RING_COLORS = (1, 2, 3, 4, 6)
RING_SIZE = 6
RING: tuple[tuple[int, int], ...] = (
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 4),
    (2, 0),
    (2, 5),
    (3, 0),
    (3, 5),
    (4, 1),
    (4, 4),
    (5, 2),
    (5, 3),
)
RING_SET = frozenset(RING)

GRID_ROWS_RANGE = (16, 20)
GRID_COLS_RANGE = (16, 20)
RING_COUNT_RANGE = (3, 5)
K_EQ_COUNT_PROB = 0.15
MIN_VISIBLE = 4
MAX_VISIBLE = 10
NOISE_PER_COLOR_PROB = 0.6
MAX_NOISE_PER_COLOR = 2
COUNTER_MARGIN = 2
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _ring_cells(offset: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = offset
    return {(r0 + dr, c0 + dc) for dr, dc in RING}


def _offset_bbox(offset: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = offset
    return r0, c0, r0 + RING_SIZE - 1, c0 + RING_SIZE - 1


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _exact_matching_offsets(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    offset_counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in RING:
            offset = (r - dr, c - dc)
            offset_counts[offset] = offset_counts.get(offset, 0) + 1
    return [offset for offset, count in offset_counts.items() if count == len(coords)]


def _fit_ring_fragment(
    coords: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]], int]:
    assert coords, "expected visible pixels for a ring color"
    offset_counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in RING:
            offset = (r - dr, c - dc)
            offset_counts[offset] = offset_counts.get(offset, 0) + 1

    best_count = max(offset_counts.values())
    best_offsets = [offset for offset, count in offset_counts.items() if count == best_count]
    assert len(best_offsets) == 1, "ring fragment must determine a unique template position"
    offset = best_offsets[0]

    matched = {(r, c) for r, c in coords if (r - offset[0], c - offset[1]) in RING_SET}
    assert len(matched) == best_count
    assert best_count >= MIN_VISIBLE, "ring fragment must expose enough template pixels"
    assert _exact_matching_offsets(matched) == [offset], "matched fragment must identify one offset"
    return offset, matched, best_count


def _find_fragments(
    grid: np.ndarray,
) -> dict[int, tuple[tuple[int, int], set[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    fragments: dict[int, tuple[tuple[int, int], set[tuple[int, int]]]] = {}
    for color in np.unique(g):
        color = int(color)
        if color not in RING_COLORS:
            continue
        coords = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        if not coords:
            continue
        offset, matched, _ = _fit_ring_fragment(coords)
        fragments[color] = (offset, matched)
    assert RING_COUNT_RANGE[0] <= len(fragments) <= RING_COUNT_RANGE[1], "expected 3-5 ring colors"
    return fragments


def _layer_order(
    grid: np.ndarray,
    fragments: dict[int, tuple[tuple[int, int], set[tuple[int, int]]]],
) -> tuple[list[int], dict[int, set[int]]]:
    g = np.asarray(grid, dtype=int)
    ring_colors = set(fragments)
    above_to_below = {color: set() for color in ring_colors}
    indegree = {color: 0 for color in ring_colors}

    for color, (offset, visible) in fragments.items():
        for r, c in _ring_cells(offset) - visible:
            cover = int(g[r, c])
            assert cover in ring_colors and cover != color, "missing ring pixels must be covered by another ring"
            if color not in above_to_below[cover]:
                above_to_below[cover].add(color)
                indegree[color] += 1

    order: list[int] = []
    remaining = set(ring_colors)
    while remaining:
        top = [color for color in remaining if indegree[color] == 0]
        assert len(top) == 1, "occlusion relations must determine a unique layer order"
        color = top[0]
        order.append(color)
        remaining.remove(color)
        for below in above_to_below[color]:
            indegree[below] -= 1
    return order, above_to_below


def _render_complete_rings(colors: list[int]) -> np.ndarray:
    out = np.full((RING_SIZE, RING_SIZE * len(colors)), BG, dtype=int)
    for i, color in enumerate(colors):
        c0 = i * RING_SIZE
        for dr, dc in RING:
            out[dr, c0 + dc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapped ring sprites, rank them by occlusion depth, then tile the first K complete rings horizontally.

    Input structure:
    - The background is black, with 3-5 colored 6x6 hollow-ring sprites drawn in distinct colors from a fixed ring palette.
    - The rings overlap, so lower rings are only partially visible; some same-colored stray pixels may appear off the ring template as noise.
    - A separate 3x3 grey-on-black counter sprite sits in one corner and does not intersect the ring scene.

    Rule:
    - For each ring color, find the unique translated 6x6 ring template that explains the largest subset of that color's pixels and ignore same-colored off-template noise.
    - Any expected ring pixel that is missing from a fragment must currently show the color of a higher ring, which gives a directed occlusion edge; topologically sort these edges from top to bottom.
    - Count the grey pixels in the corner 3x3 sprite to get K, reconstruct every ring completely, and output the first K rings in top-to-bottom order as a 6 x (6K) horizontal tiling.

    Key insight:
    - The visible ring fragments reveal both each ring's full shape and the depth order, while the separate grey mini-sprite independently says how many of the sorted reconstructed rings to keep.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fragments = _find_fragments(g)
    order, _ = _layer_order(g, fragments)

    k = int(np.count_nonzero(g == COUNTER_COLOR))
    assert 1 <= k <= len(order), "grey counter must request 1..ring_count output rings"
    return _render_complete_rings(order[:k])


def _corner_bbox(rows: int, cols: int, corner: str, size: int = 3) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, size - 1, size - 1
    if corner == "tr":
        return 0, cols - size, size - 1, cols - 1
    if corner == "bl":
        return rows - size, 0, rows - 1, size - 1
    if corner == "br":
        return rows - size, cols - size, rows - 1, cols - 1
    raise ValueError(f"unknown corner {corner}")


def _expanded_corner_block(rows: int, cols: int, corner: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _corner_bbox(rows, cols, corner, size=3)
    if corner == "tl":
        return 0, 0, min(rows - 1, r1 + COUNTER_MARGIN), min(cols - 1, c1 + COUNTER_MARGIN)
    if corner == "tr":
        return 0, max(0, c0 - COUNTER_MARGIN), min(rows - 1, r1 + COUNTER_MARGIN), cols - 1
    if corner == "bl":
        return max(0, r0 - COUNTER_MARGIN), 0, rows - 1, min(cols - 1, c1 + COUNTER_MARGIN)
    return max(0, r0 - COUNTER_MARGIN), max(0, c0 - COUNTER_MARGIN), rows - 1, cols - 1


def _all_scene_offsets(
    rows: int, cols: int, forbidden_box: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    offsets = []
    for r0 in range(rows - RING_SIZE + 1):
        for c0 in range(cols - RING_SIZE + 1):
            if not _boxes_intersect(_offset_bbox((r0, c0)), forbidden_box):
                offsets.append((r0, c0))
    return offsets


def _candidate_offsets(
    rows: int,
    cols: int,
    forbidden_box: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
    higher_cells: list[set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    covered_by_higher = set().union(*higher_cells)
    covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
    visible_prev = higher_cells[-1] - covered_before_prev

    candidates: list[tuple[int, int]] = []
    for offset in _all_scene_offsets(rows, cols, forbidden_box):
        if offset in offsets:
            continue
        cells = _ring_cells(offset)
        visible = cells - covered_by_higher
        if not (MIN_VISIBLE <= len(visible) <= MAX_VISIBLE):
            continue
        if not (cells & visible_prev):
            continue
        if _exact_matching_offsets(visible) != [offset]:
            continue
        candidates.append(offset)
    return candidates


def _build_offsets(
    rows: int,
    cols: int,
    forbidden_box: tuple[int, int, int, int],
    ring_count: int,
) -> list[tuple[int, int]] | None:
    valid_offsets = _all_scene_offsets(rows, cols, forbidden_box)
    if not valid_offsets:
        return None

    top_offset = random.choice(valid_offsets)
    offsets = [top_offset]
    higher_cells = [_ring_cells(top_offset)]

    for _ in range(1, ring_count):
        candidates = _candidate_offsets(rows, cols, forbidden_box, offsets, higher_cells)
        if not candidates:
            return None
        offset = random.choice(candidates)
        offsets.append(offset)
        higher_cells.append(_ring_cells(offset))
    return offsets


def _render_scene(
    rows: int,
    cols: int,
    order: list[int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((rows, cols), BG, dtype=int)
    for color, offset in reversed(list(zip(order, offsets))):
        for r, c in _ring_cells(offset):
            grid[r, c] = color
    return grid


def _place_counter(grid: np.ndarray, corner: str, k: int) -> None:
    r0, c0, _r1, _c1 = _corner_bbox(*grid.shape, corner, size=3)
    cells = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
    for r, c in random.sample(cells, k):
        grid[r, c] = COUNTER_COLOR


def _noise_candidates(
    grid: np.ndarray,
    ring_offsets: list[tuple[int, int]],
    counter_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    ring_cells_union = set().union(*(_ring_cells(offset) for offset in ring_offsets))
    candidates = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != BG:
                continue
            if (r, c) in ring_cells_union:
                continue
            if _boxes_intersect((r, c, r, c), counter_box):
                continue
            candidates.append((r, c))
    return candidates


def _pick_isolated_cells(
    candidates: list[tuple[int, int]],
    count: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        r, c = cell
        if cell in blocked:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _add_noise(
    grid: np.ndarray,
    order: list[int],
    offsets: list[tuple[int, int]],
    counter_box: tuple[int, int, int, int],
) -> bool:
    candidates = _noise_candidates(grid, offsets, counter_box)
    blocked: set[tuple[int, int]] = set()
    noisy_colors = [color for color in order if random.random() < NOISE_PER_COLOR_PROB]
    if not noisy_colors and random.random() < 0.8:
        noisy_colors = [random.choice(order)]

    for color in noisy_colors:
        count = random.randint(1, MAX_NOISE_PER_COLOR)
        cells = _pick_isolated_cells(candidates, count, blocked)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
        blocked.update(cells)
        candidates = [cell for cell in candidates if cell not in blocked]
    return True


def _expected_output(order: list[int], k: int) -> np.ndarray:
    return _render_complete_rings(order[:k])


def _semantic_checks(
    grid: np.ndarray,
    expected_order: list[int],
    expected_k: int,
    offsets: list[tuple[int, int]],
) -> bool:
    try:
        fragments = _find_fragments(grid)
        order, edges = _layer_order(grid, fragments)
        out = main(grid)
    except Exception:
        return False

    if order != expected_order:
        return False
    if int(np.count_nonzero(grid == COUNTER_COLOR)) != expected_k:
        return False
    if not np.array_equal(out, _expected_output(expected_order, expected_k)):
        return False

    for color, expected_offset in zip(expected_order, offsets):
        found_offset, visible = fragments[color]
        if found_offset != expected_offset:
            return False
        if color != expected_order[0] and len(visible) == len(RING):
            return False

    for above, below in zip(expected_order, expected_order[1:]):
        if below not in edges[above]:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    ring_count = random.randint(*RING_COUNT_RANGE)
    corner = random.choice(("tl", "tr", "bl", "br"))
    counter_box = _corner_bbox(rows, cols, corner, size=3)
    forbidden_box = _expanded_corner_block(rows, cols, corner)

    order = random.sample(list(RING_COLORS), ring_count)
    if ring_count > 1 and random.random() > K_EQ_COUNT_PROB:
        k = random.randint(1, ring_count - 1)
    else:
        k = ring_count

    offsets = _build_offsets(rows, cols, forbidden_box, ring_count)
    if offsets is None:
        return None

    grid = _render_scene(rows, cols, order, offsets)
    _place_counter(grid, corner, k)
    if not _add_noise(grid, order, offsets, counter_box):
        return None

    if not _semantic_checks(grid, order, k, offsets):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
