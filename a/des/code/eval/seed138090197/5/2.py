import random
from typing import Iterable

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
GRID_MARGIN_RANGE = (2, 5)
PATTERN_SIZE_RANGE = (7, 13)
PATTERN_FILL_FRAC = (0.24, 0.46)
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_MAX = 4
RECT_PLACEMENT_RETRIES = 160
TEMPLATE_GROWTH_EXTRA_MAX = 4
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one marked cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rot90_cell(r: int, c: int, size: int) -> tuple[int, int]:
    return c, size - 1 - r


def _all_orbits(size: int) -> tuple[list[tuple[tuple[int, int], ...]], dict[tuple[int, int], int]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    cell_to_orbit: dict[tuple[int, int], int] = {}

    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = []
            rr, cc = r, c
            for _ in range(4):
                if (rr, cc) not in orbit:
                    orbit.append((rr, cc))
                rr, cc = _rot90_cell(rr, cc, size)
            orbit_tuple = tuple(sorted(orbit))
            orbit_id = len(orbits)
            orbits.append(orbit_tuple)
            for cell in orbit_tuple:
                seen.add(cell)
                cell_to_orbit[cell] = orbit_id

    return orbits, cell_to_orbit


def _orbit_graph(size: int) -> tuple[list[tuple[tuple[int, int], ...]], list[set[int]], dict[tuple[int, int], int]]:
    orbits, cell_to_orbit = _all_orbits(size)
    adjacency = [set() for _ in orbits]
    for (r, c), orbit_id in cell_to_orbit.items():
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size:
                other_id = cell_to_orbit[(rr, cc)]
                if other_id != orbit_id:
                    adjacency[orbit_id].add(other_id)
    return orbits, adjacency, cell_to_orbit


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _infer_template_color(local: np.ndarray, bg: int, orbits: Iterable[tuple[tuple[int, int], ...]]) -> int:
    candidates = [int(color) for color in np.unique(local) if int(color) != bg]
    assert candidates, "expected at least one non-background color"

    best_key = None
    best_color = None
    for color in candidates:
        contradictions = 0
        orbit_support = 0
        visible_count = int(np.count_nonzero(local == color))
        for orbit in orbits:
            values = [int(local[r, c]) for r, c in orbit]
            if color in values:
                orbit_support += 1
                if bg in values:
                    contradictions += 1
        key = (contradictions, -orbit_support, -visible_count, color)
        if best_key is None or key < best_key:
            best_key = key
            best_color = color

    assert best_color is not None
    return best_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a 90-degree rotationally symmetric monochrome template hidden under colored rectangles.

    The input contains one square template region with black background. The
    true template uses one foreground color and is invariant under 90-degree
    rotation; colored rectangles overwrite some cells inside that square. The
    output removes the rectangles and restores the original foreground/background
    pattern from rotational orbit agreement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    non_bg = g != bg
    assert non_bg.any(), "expected a non-background template region"
    r0, c0, r1, c1 = _bbox_from_mask(non_bg)
    size = r1 - r0 + 1
    assert size == c1 - c0 + 1, "90-degree rotational template must live in a square bbox"

    local = g[r0 : r1 + 1, c0 : c1 + 1]
    orbits, _ = _all_orbits(size)
    fg = _infer_template_color(local, bg, orbits)

    out = np.full_like(g, bg)
    for orbit in orbits:
        values = [int(local[r, c]) for r, c in orbit]
        has_fg = fg in values
        has_bg = bg in values
        assert has_fg or has_bg, "every rotational orbit must expose at least one unoccluded sample"
        assert not (has_fg and has_bg), "visible samples in one orbit must agree on the hidden template"
        fill = fg if has_fg else bg
        for r, c in orbit:
            out[r0 + r, c0 + c] = fill

    return out


def _make_template(rng: random.Random, size: int) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    orbits, adjacency, cell_to_orbit = _orbit_graph(size)
    center_cell = (size // 2, size // 2) if size % 2 else (size // 2 - 1, size // 2 - 1)
    center_orbit = cell_to_orbit[center_cell]
    boundary_orbits = {
        orbit_id
        for orbit_id, orbit in enumerate(orbits)
        if any(r in (0, size - 1) or c in (0, size - 1) for r, c in orbit)
    }

    target_area = rng.randint(
        max(10, int(size * size * PATTERN_FILL_FRAC[0])),
        max(12, int(size * size * PATTERN_FILL_FRAC[1])),
    )

    active = {center_orbit}
    frontier = set(adjacency[center_orbit])
    while frontier and (
        sum(len(orbits[idx]) for idx in active) < target_area or not (active & boundary_orbits)
    ):
        candidates = sorted(frontier)
        weights = []
        for orbit_id in candidates:
            bonus = 2.0 if orbit_id in boundary_orbits and not (active & boundary_orbits) else 1.0
            weights.append(bonus + rng.random())
        chosen = rng.choices(candidates, weights=weights, k=1)[0]
        active.add(chosen)
        frontier.remove(chosen)
        frontier.update(adjacency[chosen] - active)

    for _ in range(rng.randint(0, TEMPLATE_GROWTH_EXTRA_MAX)):
        if not frontier:
            break
        chosen = rng.choice(sorted(frontier))
        active.add(chosen)
        frontier.remove(chosen)
        frontier.update(adjacency[chosen] - active)

    mask = np.zeros((size, size), dtype=bool)
    for orbit_id in active:
        for r, c in orbits[orbit_id]:
            mask[r, c] = True

    return mask, orbits


def _place_rectangles(
    rng: random.Random, mask: np.ndarray, orbits: list[tuple[tuple[int, int], ...]]
) -> list[tuple[int, int, int, int]]:
    size = mask.shape[0]
    rect_count = rng.randint(*RECT_COUNT_RANGE)

    for _ in range(RECT_PLACEMENT_RETRIES):
        covered = np.zeros_like(mask, dtype=bool)
        blocked = np.zeros_like(mask, dtype=bool)
        rectangles: list[tuple[int, int, int, int]] = []

        ok = True
        for _ in range(rect_count):
            placed = False
            for _ in range(80):
                h = rng.randint(1, max(1, min(RECT_SIDE_MAX, size - 3)))
                w = rng.randint(1, max(1, min(RECT_SIDE_MAX, size - 3)))
                r0 = rng.randint(1, size - h - 1)
                c0 = rng.randint(1, size - w - 1)
                r1, c1 = r0 + h, c0 + w

                pad_r0 = max(0, r0 - 1)
                pad_c0 = max(0, c0 - 1)
                pad_r1 = min(size, r1 + 1)
                pad_c1 = min(size, c1 + 1)
                if blocked[pad_r0:pad_r1, pad_c0:pad_c1].any():
                    continue

                covered[r0:r1, c0:c1] = True
                blocked[pad_r0:pad_r1, pad_c0:pad_c1] = True
                rectangles.append((r0, c0, h, w))
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if not np.any(covered & mask):
            continue
        if not np.any(covered & ~mask):
            continue
        if any(all(covered[r, c] for r, c in orbit) for orbit in orbits):
            continue

        visible_fg = int(np.count_nonzero(mask & ~covered))
        if visible_fg <= max(h * w for _, _, h, w in rectangles):
            continue

        return rectangles

    raise RuntimeError("failed to place non-degenerate rectangles")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.randint(*PATTERN_SIZE_RANGE)
    mask, orbits = _make_template(rng, size)
    if not _is_connected(mask):
        return None
    if _bbox_from_mask(mask) != (0, 0, size - 1, size - 1):
        return None
    if int(mask.sum()) < 12 or int((~mask).sum()) < 8:
        return None

    fg = rng.choice([color for color in PALETTE if color != 0])
    remaining = [color for color in PALETTE if color not in (0, fg)]

    try:
        rectangles = _place_rectangles(rng, mask, orbits)
    except RuntimeError:
        return None

    top = rng.randint(*GRID_MARGIN_RANGE)
    bottom = rng.randint(*GRID_MARGIN_RANGE)
    left = rng.randint(*GRID_MARGIN_RANGE)
    right = rng.randint(*GRID_MARGIN_RANGE)
    height = size + top + bottom
    width = size + left + right

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    local = grid[top : top + size, left : left + size]
    local_expected = expected[top : top + size, left : left + size]
    local[mask] = fg
    local_expected[mask] = fg

    rect_colors = rng.sample(remaining, len(rectangles))
    for color, (r0, c0, h, w) in zip(rect_colors, rectangles):
        local[r0 : r0 + h, c0 : c0 + w] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-occlusion ARC input")
