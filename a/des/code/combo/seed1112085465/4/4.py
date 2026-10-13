from __future__ import annotations

import random

import numpy as np


BG = 8
MARKER = 4
PALETTE = (0, 1, 2, 3, 5, 6, 7, 9)
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MIN_SIZE = 12
MAX_SIZE = 22
COUNT_CHOICES = (1, 2, 3, 4, 5)
MAX_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each yellow marker into a solid square whose color and size are
    encoded by the marker's 8-neighborhood.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for rr, cc in np.argwhere(g == MARKER):
        r = int(rr)
        c = int(cc)
        counts: dict[int, int] = {}
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                color = int(g[nr, nc])
                if color != BG and color != MARKER:
                    counts[color] = counts.get(color, 0) + 1

        assert counts, "every marker must see at least one sprite pixel"
        size_code = max(counts.values())
        winners = [color for color, count in counts.items() if count == size_code]
        assert len(winners) == 1, "majority color must be unique"
        fill = winners[0]
        radius = size_code - 1

        for nr in range(max(0, r - radius), min(h, r + radius + 1)):
            for nc in range(max(0, c - radius), min(w, c + radius + 1)):
                if (nr == r and nc == c) or (int(g[nr, nc]) == BG and int(out[nr, nc]) == BG):
                    out[nr, nc] = fill

    return out


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _bbox_around(r: int, c: int, radius: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, r - radius),
        max(0, c - radius),
        min(h - 1, r + radius),
        min(w - 1, c + radius),
    )


def _adjacent_offsets(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _pick_cluster(options: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    assert 0 < size <= len(options)
    chosen = [random.choice(options)]
    remaining = [cell for cell in options if cell != chosen[0]]

    while len(chosen) < size:
        frontier = [cell for cell in remaining if any(_adjacent_offsets(cell, other) for other in chosen)]
        pool = frontier if frontier else remaining
        cell = random.choice(pool)
        chosen.append(cell)
        remaining.remove(cell)

    return chosen


def _partition_total(total: int, max_part: int) -> list[int]:
    if total == 0:
        return []

    min_parts = (total + max_part - 1) // max_part
    max_parts = min(total, 3)
    if min_parts > max_parts:
        raise ValueError("cannot partition total under the majority constraint")

    parts = random.randint(min_parts, max_parts)
    while True:
        cuts = [0] * parts
        remaining = total
        for i in range(parts):
            tail = parts - i - 1
            lo = 1
            hi = min(max_part, remaining - tail)
            if hi < lo:
                break
            value = random.randint(lo, hi)
            cuts[i] = value
            remaining -= value
        else:
            if remaining == 0:
                random.shuffle(cuts)
                return cuts


def _marker_positions(
    shape: tuple[int, int],
    radius: int,
    clipped: bool,
    reserved: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    reserve_radius = max(radius, 2)
    positions: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            square_clipped = r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w
            if square_clipped != clipped:
                continue
            bbox = _bbox_around(r, c, reserve_radius, shape)
            if any(_boxes_overlap(bbox, other, margin=1) for other in reserved):
                continue
            positions.append((r, c))
    return positions


def _sample_marker_layout(count: int) -> tuple[list[int], list[bool]]:
    counts = [random.choice(COUNT_CHOICES) for _ in range(count)]
    if all(n == 1 for n in counts):
        counts[random.randrange(count)] = random.choice(COUNT_CHOICES[1:])

    clipped = [False] * count
    eligible = [i for i, n in enumerate(counts) if n >= 3]
    for i in eligible:
        clipped[i] = random.random() < 0.5
    if len(eligible) >= 2:
        if not any(clipped[i] for i in eligible):
            clipped[random.choice(eligible)] = True
        if all(clipped[i] for i in eligible):
            clipped[random.choice(eligible)] = False

    return counts, clipped


def _paint_marker_neighborhood(
    grid: np.ndarray,
    center: tuple[int, int],
    majority_color: int,
    count: int,
    color_pool: list[int],
) -> None:
    r, c = center
    available = list(NEIGHBORS8)
    majority_offsets = _pick_cluster(available, count)
    color_map: dict[int, list[tuple[int, int]]] = {majority_color: majority_offsets}

    remaining = [cell for cell in available if cell not in majority_offsets]
    other_total = 0 if count == 1 else random.randint(0, min(3, len(remaining)))
    other_counts = _partition_total(other_total, count - 1) if other_total else []
    other_colors = random.sample([color for color in color_pool if color != majority_color], len(other_counts))

    for color, size in zip(other_colors, other_counts):
        picked = _pick_cluster(remaining, size)
        color_map[color] = picked
        remaining = [cell for cell in remaining if cell not in picked]

    grid[r, c] = MARKER
    for color, offsets in color_map.items():
        for dr, dc in offsets:
            grid[r + dr, c + dc] = color
            if random.random() < 0.45:
                er = r + 2 * dr
                ec = c + 2 * dc
                if 0 <= er < grid.shape[0] and 0 <= ec < grid.shape[1] and int(grid[er, ec]) == BG:
                    grid[er, ec] = color


def _add_extra_objects(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
) -> None:
    extra_count = random.randint(0, 2)
    h, w = grid.shape
    for _ in range(extra_count):
        color = random.choice(PALETTE)
        size = random.randint(1, 3)
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if int(grid[r, c]) == BG
            and not any(_boxes_overlap((r, c, r, c), box, margin=0) for box in reserved)
        ]
        if not candidates:
            return
        start = random.choice(candidates)
        cells = [start]
        frontier = [start]
        while len(cells) < size and frontier:
            base = random.choice(frontier)
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = base[0] + dr
                nc = base[1] + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG:
                    cell = (nr, nc)
                    if any(_boxes_overlap((nr, nc, nr, nc), box, margin=0) for box in reserved):
                        continue
                    if cell not in cells:
                        neighbors.append(cell)
            if not neighbors:
                frontier.remove(base)
                continue
            cell = random.choice(neighbors)
            cells.append(cell)
            frontier.append(cell)
        for r, c in cells:
            grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    grid = np.full((h, w), BG, dtype=int)

    max_markers = 2
    if min(h, w) >= 15:
        max_markers = 3
    if min(h, w) >= 19:
        max_markers = 4
    marker_count = random.randint(1, max_markers)

    counts, clipped = _sample_marker_layout(marker_count)
    majority_colors = random.sample(list(PALETTE), marker_count)
    specs = [
        {
            "count": count,
            "radius": count - 1,
            "clipped": clip,
            "color": color,
        }
        for count, clip, color in zip(counts, clipped, majority_colors)
    ]
    specs.sort(key=lambda spec: max(spec["radius"], 2), reverse=True)

    reserved: list[tuple[int, int, int, int]] = []
    for spec in specs:
        positions = _marker_positions((h, w), spec["radius"], spec["clipped"], reserved)
        if not positions:
            return None
        center = random.choice(positions)
        reserved.append(_bbox_around(center[0], center[1], max(spec["radius"], 2), (h, w)))
        _paint_marker_neighborhood(grid, center, spec["color"], spec["count"], list(PALETTE))

    _add_extra_objects(grid, reserved)

    if not np.any(grid == MARKER):
        return None
    if not np.any((grid != BG) & (grid != MARKER)):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid != BG)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
