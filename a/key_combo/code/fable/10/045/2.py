import random
from collections import Counter

import numpy as np


RING_SIZE = 5
RING_CELLS = tuple(
    (r, c)
    for r in range(RING_SIZE)
    for c in range(RING_SIZE)
    if r in (0, RING_SIZE - 1) or c in (0, RING_SIZE - 1)
)

PALETTE = tuple(range(10))
RETRY_LIMIT = 800
RING_COUNT_RANGE = (2, 4)
HEIGHT_RANGE = (12, 24)
WIDTH_RANGE = (12, 20)
VISIBLE_RANGE = (5, 12)


def _background_color(grid: np.ndarray) -> int:
    values = grid[:-1].reshape(-1) if grid.shape[0] > 1 else grid.reshape(-1)
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _ring_points(offset: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = offset
    return {(r0 + dr, c0 + dc) for dr, dc in RING_CELLS}


def _matching_offsets(coords: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    if not coords:
        return []
    candidates = None
    for r, c in coords:
        offsets = {(r - dr, c - dc) for dr, dc in RING_CELLS}
        candidates = offsets if candidates is None else candidates & offsets
        if not candidates:
            return []

    h, w = shape
    out = []
    for r0, c0 in sorted(candidates):
        if not (0 <= r0 <= h - RING_SIZE and 0 <= c0 <= w - RING_SIZE):
            continue
        if coords.issubset(_ring_points((r0, c0))):
            out.append((r0, c0))
    return out


def _find_fragments(grid: np.ndarray, bg: int, floor: int) -> dict[int, tuple[int, int]]:
    fragments: dict[int, tuple[int, int]] = {}
    for color0 in np.unique(grid):
        color = int(color0)
        if color in (bg, floor):
            continue
        coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
        matches = _matching_offsets(coords, grid.shape)
        assert len(matches) == 1, "each color fragment must fit one unique ring offset"
        fragments[color] = matches[0]
    assert fragments, "expected at least one ring fragment"
    return fragments


def _layer_order(
    grid: np.ndarray, fragments: dict[int, tuple[int, int]], bg: int, floor: int
) -> list[int]:
    colors = list(fragments)
    above: dict[int, set[int]] = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}

    for lower, offset in fragments.items():
        for r, c in _ring_points(offset):
            seen = int(grid[r, c])
            if seen == lower:
                continue
            assert seen not in (bg, floor), "hidden ring pixels must be covered by another ring"
            if lower not in above[seen]:
                above[seen].add(lower)
                indeg[lower] += 1

    order: list[int] = []
    free = sorted(color for color in colors if indeg[color] == 0)
    while free:
        assert len(free) == 1, "occlusion must determine a unique total order"
        color = free.pop()
        order.append(color)
        for nxt in sorted(above[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                free.append(nxt)
        free.sort()

    assert len(order) == len(colors), "occlusion graph must be acyclic"
    return order


def _drop_rings(
    shape: tuple[int, int],
    bg: int,
    floor: int,
    fragments: dict[int, tuple[int, int]],
    drop_order: list[int],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[-1, :] = floor
    occupied = np.zeros((h, w), dtype=bool)
    occupied[-1, :] = True
    final_offsets: dict[int, tuple[int, int]] = {}

    for color in drop_order:
        row, col = fragments[color]
        while True:
            next_row = row + 1
            next_cells = {(next_row + dr, col + dc) for dr, dc in RING_CELLS}
            if any(r >= h or occupied[r, c] for r, c in next_cells):
                break
            row = next_row
        for r, c in _ring_points((row, col)):
            out[r, c] = color
            occupied[r, c] = True
        final_offsets[color] = (row, col)

    return out, final_offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapping hollow square rings, then drop them in back-to-front order until they stack on the floor.

    Input structure:
    - The dominant color is the background, and the bottom row is one uniform floor color.
    - Several distinct non-background colors appear as visible fragments of equal 5x5 hollow square rings.
    - Fragments overlap because higher rings overwrite expected border pixels of lower rings.

    Rule:
    - Recover the unique 5x5 ring placement for each color from its visible border pixels.
    - Whenever one ring expects a border pixel but another color is currently visible there, that visible color is above it; use all such covers to derive a unique top-to-bottom order.
    - Clear the original scene, then from the bottom-most ring upward drop each full ring straight down until one more step would hit the floor row or an already settled ring.

    Key insight:
    - The overlap order is not the drawing order of the output image; it becomes the time order of gravity, with deeper rings moving first and claiming the lower resting places.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= RING_SIZE + 2 and g.shape[1] >= RING_SIZE

    floor = int(g[-1, 0])
    assert np.all(g[-1] == floor), "bottom row must be a uniform floor"
    bg = _background_color(g)
    assert bg != floor, "background and floor must differ"

    fragments = _find_fragments(g, bg, floor)
    top_to_bottom = _layer_order(g, fragments, bg, floor)
    out, _ = _drop_rings(g.shape, bg, floor, fragments, list(reversed(top_to_bottom)))
    return out


def _render_input(
    shape: tuple[int, int],
    bg: int,
    floor: int,
    top_to_bottom: list[int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), bg, dtype=int)
    grid[-1, :] = floor
    for color, offset in reversed(list(zip(top_to_bottom, offsets))):
        for r, c in _ring_points(offset):
            grid[r, c] = color
    return grid


def _build_offsets(height: int, width: int, ring_count: int) -> list[tuple[int, int]] | None:
    row_max = max(1, min(height - RING_SIZE - 1, height - 10))
    col_max = width - RING_SIZE
    top = (random.randint(0, row_max), random.randint(0, col_max))
    offsets = [top]
    higher_cells = [_ring_points(top)]

    for _ in range(1, ring_count):
        covered_by_higher = set().union(*higher_cells)
        covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
        visible_prev = higher_cells[-1] - covered_before_prev
        candidates: list[tuple[int, int]] = []
        for r0 in range(row_max + 1):
            for c0 in range(col_max + 1):
                if (r0, c0) in offsets:
                    continue
                cells = _ring_points((r0, c0))
                visible = cells - covered_by_higher
                if not (VISIBLE_RANGE[0] <= len(visible) <= VISIBLE_RANGE[1]):
                    continue
                if not (cells & visible_prev):
                    continue
                if _matching_offsets(visible, (height, width)) != [(r0, c0)]:
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        offset = random.choice(candidates)
        offsets.append(offset)
        higher_cells.append(_ring_points(offset))
    return offsets


def _semantic_ok(
    inp: np.ndarray,
    bg: int,
    floor: int,
    top_to_bottom: list[int],
    fragments: dict[int, tuple[int, int]],
) -> bool:
    recovered = _find_fragments(inp, bg, floor)
    if recovered != fragments:
        return False
    recovered_order = _layer_order(inp, recovered, bg, floor)
    if recovered_order != top_to_bottom:
        return False

    direct_edges = set()
    for i in range(len(top_to_bottom) - 1):
        direct_edges.add((top_to_bottom[i], top_to_bottom[i + 1]))
    found_edges = set()
    for lower, offset in recovered.items():
        for r, c in _ring_points(offset):
            seen = int(inp[r, c])
            if seen not in (bg, floor, lower):
                found_edges.add((seen, lower))
    if not direct_edges.issubset(found_edges):
        return False

    correct_drop = list(reversed(top_to_bottom))
    correct_out, final_offsets = _drop_rings(inp.shape, bg, floor, recovered, correct_drop)
    movement = [final_offsets[color][0] - recovered[color][0] for color in correct_drop]
    if sum(movement) < len(correct_drop) + 2 or max(movement) < 2:
        return False

    y_sorted_drop = sorted(correct_drop, key=lambda color: recovered[color][0], reverse=True)
    if y_sorted_drop == correct_drop:
        return False
    y_out, _ = _drop_rings(inp.shape, bg, floor, recovered, y_sorted_drop)
    if np.array_equal(y_out, correct_out):
        return False

    wrong_out, _ = _drop_rings(inp.shape, bg, floor, recovered, top_to_bottom)
    if np.array_equal(wrong_out, correct_out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    ring_count = random.randint(*RING_COUNT_RANGE)

    colors = random.sample(PALETTE, ring_count + 2)
    bg, floor = colors[:2]
    ring_colors = colors[2:]

    offsets = _build_offsets(height, width, ring_count)
    if offsets is None:
        return None

    top_to_bottom = ring_colors[:]
    fragments = dict(zip(top_to_bottom, offsets))
    inp = _render_input((height, width), bg, floor, top_to_bottom, offsets)

    visible_counts = [int(np.count_nonzero(inp == color)) for color in top_to_bottom]
    if min(visible_counts) < VISIBLE_RANGE[0] or visible_counts[0] != len(RING_CELLS):
        return None
    if max(visible_counts[1:], default=VISIBLE_RANGE[0]) == len(RING_CELLS):
        return None

    if not _semantic_ok(inp, bg, floor, top_to_bottom, fragments):
        return None
    if not np.array_equal(main(inp), _drop_rings(inp.shape, bg, floor, fragments, list(reversed(top_to_bottom)))[0]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-drop input")
