import random
from collections import Counter

import numpy as np


BACKGROUND = 0
EQUAL_FILL = 5
RING_COLORS = (1, 2, 3, 4, 6, 7)
RING_SIZE = 3
OUTLINE_RADIUS = 2
MIN_GRID_SIZE = 13
MAX_GRID_SIZE = 22
MIN_RING_COUNT = 3
MAX_RING_COUNT = 4
GEN_ATTEMPTS = 400

CORNER_OFFSETS = ((0, 0), (0, 2), (2, 0), (2, 2))
SIDE_OFFSETS = ((0, 1), (1, 2), (2, 1), (1, 0))
RING_OFFSETS = tuple((dr, dc) for dr in range(3) for dc in range(3) if (dr, dc) != (1, 1))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mode_by_reading_order(values: list[int]) -> int:
    counts = Counter(values)
    best = max(counts.values())
    for value in values:
        if counts[value] == best:
            return int(value)
    raise AssertionError("mode selection failed")


def _ring_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BACKGROUND:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BACKGROUND:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            components.append({"coords": coords, "bbox": (r0, c0, r1, c1), "area": len(coords)})

    return components


def _extract_ring(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = bbox
    assert r1 == r0 + 2 and c1 == c0 + 2, "expected 3x3 rings"
    assert grid[r0 + 1, c0 + 1] == BACKGROUND, "expected hollow ring center"

    expected = {(r0 + dr, c0 + dc) for dr, dc in RING_OFFSETS}
    corners = [int(grid[r0 + dr, c0 + dc]) for dr, dc in CORNER_OFFSETS]
    sides = [int(grid[r0 + dr, c0 + dc]) for dr, dc in SIDE_OFFSETS]
    actual = {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3) if grid[r0 + dr, c0 + dc] != BACKGROUND}
    assert actual == expected, "component bbox must be a full 3x3 ring"
    assert all(color in RING_COLORS for color in corners + sides), "unexpected ring color"
    return corners, sides


def _draw_outline(out: np.ndarray, center_r: int, center_c: int, color: int) -> None:
    h, w = out.shape
    r0, r1 = center_r - OUTLINE_RADIUS, center_r + OUTLINE_RADIUS
    c0, c1 = center_c - OUTLINE_RADIUS, center_c + OUTLINE_RADIUS
    assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "5x5 outline must fit"

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr in (r0, r1) or cc in (c0, c1):
                if out[rr, cc] == BACKGROUND:
                    out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Resolve each 3x3 ring by corner-vs-side color diversity, filling the center and sometimes drawing a 5x5 outline.

    Input structure:
    - A black background contains several separated 3x3 foreground rings.
    - Each ring has eight colored pixels around a black center.
    - Within each ring, the four corner cells and four side-center cells may repeat colors differently.

    Rule:
    - For each ring, count how many distinct colors appear in the corner group and in the side-center group.
    - If corner diversity is higher, fill the center with the corner-group mode and draw a 5x5 hollow outline in the side-group mode.
    - If side diversity is higher, fill the center with the side-group mode and draw a 5x5 hollow outline in the corner-group mode.
    - If the diversities are equal, fill the center with grey and draw no outline. Original ring pixels stay unchanged, and outlines only replace black background cells.

    Key insight:
    - Corners and side-centers are not transformed independently; they compete, and the more diverse group wins the center while the less diverse group supplies the outer outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    components = _ring_components(g)
    assert components, "expected at least one ring"

    for comp in components:
        bbox = comp["bbox"]
        coords = {tuple(coord) for coord in comp["coords"]}
        r0, c0, r1, c1 = bbox
        expected = {(r0 + dr, c0 + dc) for dr, dc in RING_OFFSETS}

        assert comp["area"] == 8, "ring area must be 8"
        assert coords == expected, "foreground components must be complete 3x3 rings"

        corners, sides = _extract_ring(g, bbox)
        corner_diversity = len(set(corners))
        side_diversity = len(set(sides))
        center_r, center_c = r0 + 1, c0 + 1

        if corner_diversity > side_diversity:
            out[center_r, center_c] = _mode_by_reading_order(corners)
            _draw_outline(out, center_r, center_c, _mode_by_reading_order(sides))
        elif corner_diversity < side_diversity:
            out[center_r, center_c] = _mode_by_reading_order(sides)
            _draw_outline(out, center_r, center_c, _mode_by_reading_order(corners))
        else:
            out[center_r, center_c] = EQUAL_FILL

    return out


def _gap_separated(a: tuple[int, int], b: tuple[int, int], gap: int = 3) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + RING_SIZE - 1, ac0 + RING_SIZE - 1
    br1, bc1 = br0 + RING_SIZE - 1, bc0 + RING_SIZE - 1
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _outline_boxes_disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a[0] - 1, a[1] - 1
    br0, bc0 = b[0] - 1, b[1] - 1
    ar1, ac1 = ar0 + 4, ac0 + 4
    br1, bc1 = br0 + 4, bc0 + 4
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _sample_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - 3) for c in range(1, width - 3)]
    random.shuffle(candidates)

    def rec(chosen: list[tuple[int, int]], remaining: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen
        if len(remaining) < count - len(chosen):
            return None

        order = remaining[:]
        random.shuffle(order)
        for pos in order:
            if all(_gap_separated(pos, prev) and _outline_boxes_disjoint(pos, prev) for prev in chosen):
                next_remaining = [cand for cand in remaining if cand != pos]
                result = rec(chosen + [pos], next_remaining)
                if result is not None:
                    return result
        return None

    return rec([], candidates)


def _positive_partitions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    partitions: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for tail in _positive_partitions(total - first, parts - 1):
            partitions.append((first,) + tail)
    return partitions


_COUNT_PARTITIONS = {k: _positive_partitions(4, k) for k in range(1, 5)}


def _build_group(unique_count: int) -> list[int]:
    colors = random.sample(RING_COLORS, unique_count)
    counts = random.choice(_COUNT_PARTITIONS[unique_count])
    values: list[int] = []
    for color, count in zip(colors, counts):
        values.extend([color] * count)
    random.shuffle(values)
    return values


def _sample_groups(branch: str) -> tuple[list[int], list[int]]:
    assert branch in {"gt", "lt", "eq"}
    if branch == "gt":
        pairs = [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]
    elif branch == "lt":
        pairs = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)]
    else:
        pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]

    for _ in range(200):
        corner_unique, side_unique = random.choice(pairs)
        corners = _build_group(corner_unique)
        sides = _build_group(side_unique)

        if branch != "eq" and _mode_by_reading_order(corners) == _mode_by_reading_order(sides):
            continue
        if len(set(corners + sides)) < 2:
            continue
        return corners, sides
    raise RuntimeError(f"failed to sample group colors for branch {branch}")


def _place_ring(grid: np.ndarray, top_left: tuple[int, int], corners: list[int], sides: list[int]) -> None:
    r0, c0 = top_left
    for (dr, dc), color in zip(CORNER_OFFSETS, corners):
        grid[r0 + dr, c0 + dc] = color
    for (dr, dc), color in zip(SIDE_OFFSETS, sides):
        grid[r0 + dr, c0 + dc] = color


def _branch_assignment(count: int) -> list[str]:
    branches = ["gt", "lt", "eq"]
    if count == 4:
        branches.append(random.choice(branches))
    random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    ring_count = random.randint(MIN_RING_COUNT, MAX_RING_COUNT)

    positions = _sample_positions(height, width, ring_count)
    if positions is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for top_left, branch in zip(positions, _branch_assignment(ring_count)):
        corners, sides = _sample_groups(branch)
        _place_ring(grid, top_left, corners, sides)

    try:
        output = main(grid)
    except Exception:
        return None

    if np.array_equal(output, grid):
        return None

    branch_hits = {"gt": 0, "lt": 0, "eq": 0}
    for comp in _ring_components(grid):
        corners, sides = _extract_ring(grid, comp["bbox"])
        corner_diversity = len(set(corners))
        side_diversity = len(set(sides))
        if corner_diversity > side_diversity:
            branch_hits["gt"] += 1
        elif corner_diversity < side_diversity:
            branch_hits["lt"] += 1
        else:
            branch_hits["eq"] += 1

    if min(branch_hits.values()) < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
