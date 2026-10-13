from __future__ import annotations

import random

import numpy as np


BG = 0
FENCE = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 25
GEN_RETRIES = 400

_COUNT_CYCLE = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    fences: list[tuple[int, int, int, int]] = []
    for comp in _components(g == FENCE):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 5 and c1 - c0 >= 5
        assert set(comp) == _perimeter_cells(r0, c0, r1, c1)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.shape[0] >= 4 and interior.shape[1] >= 4
        assert not np.any(interior == FENCE)
        fences.append((r0, c0, r1, c1))
    fences.sort()
    assert fences
    return fences


def _collect_seeds(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = box
    seeds: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color != BG:
                assert color != FENCE
                seeds.append((r, c, color))
    assert 1 <= len(seeds) <= 4
    assert len({color for _, _, color in seeds}) == len(seeds)
    return seeds


def _nearest_seed_color(
    seeds: list[tuple[int, int, int]], corner_r: int, corner_c: int
) -> int:
    sr, sc, color = min(
        seeds,
        key=lambda item: (
            abs(item[0] - corner_r) + abs(item[1] - corner_c),
            item[0],
            item[1],
        ),
    )
    assert isinstance(sr, int) and isinstance(sc, int)
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior with a pattern chosen by its seed count.

    1 seed  -> solid fill with that color.
    2 seeds -> horizontal stripes, colors ordered by upper seed first.
    3 seeds -> vertical stripes, colors ordered by left-to-right seeds.
    4 seeds -> quadrant fill, each quadrant takes the seed nearest to its corner.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for box in _find_fences(g):
        r0, c0, r1, c1 = box
        seeds = _collect_seeds(g, box)
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = inner.shape

        if len(seeds) == 1:
            inner[:, :] = seeds[0][2]
            continue

        if len(seeds) == 2:
            ordered = sorted(seeds, key=lambda item: (item[0], item[1]))
            colors = [ordered[0][2], ordered[1][2]]
            for rr in range(inner_h):
                inner[rr, :] = colors[rr % 2]
            continue

        if len(seeds) == 3:
            ordered = sorted(seeds, key=lambda item: (item[1], item[0]))
            colors = [ordered[0][2], ordered[1][2], ordered[2][2]]
            for cc in range(inner_w):
                inner[:, cc] = colors[cc % 3]
            continue

        assert len(seeds) == 4
        rel_seeds = [(r - (r0 + 1), c - (c0 + 1), color) for r, c, color in seeds]
        mid_r = inner_h // 2
        mid_c = inner_w // 2
        tl = _nearest_seed_color(rel_seeds, 0, 0)
        tr = _nearest_seed_color(rel_seeds, 0, inner_w - 1)
        bl = _nearest_seed_color(rel_seeds, inner_h - 1, 0)
        br = _nearest_seed_color(rel_seeds, inner_h - 1, inner_w - 1)

        inner[:mid_r, :mid_c] = tl
        inner[:mid_r, mid_c:] = tr
        inner[mid_r:, :mid_c] = bl
        inner[mid_r:, mid_c:] = br

    return out


def _next_seed_counts(rng: random.Random) -> list[int]:
    global _COUNT_CYCLE
    idx = _COUNT_CYCLE
    _COUNT_CYCLE += 1

    fence_count = 3 + (idx % 3)
    ordered = [1, 2, 3, 4]
    shift = idx % 4
    ordered = ordered[shift:] + ordered[:shift]

    if fence_count == 3:
        counts = ordered[:3]
    elif fence_count == 4:
        counts = ordered[:]
    else:
        counts = ordered + [ordered[(idx // 3) % 4]]

    rng.shuffle(counts)
    return counts


def _sample_inner_dims(
    rng: random.Random, max_inner_h: int, max_inner_w: int, seed_count: int
) -> tuple[int, int]:
    inner_h = rng.randint(4, max_inner_h)
    inner_w = rng.randint(4, max_inner_w)

    mode = rng.random()
    if mode < 0.25 and max_inner_w >= 5:
        inner_w = rng.randint(5, max_inner_w)
    elif mode < 0.50 and max_inner_h >= 5:
        inner_h = rng.randint(5, max_inner_h)

    if seed_count == 4:
        odd_h = [v for v in range(4, max_inner_h + 1) if v % 2 == 1]
        odd_w = [v for v in range(4, max_inner_w + 1) if v % 2 == 1]
        if odd_h and rng.random() < 0.55:
            inner_h = rng.choice(odd_h)
        if odd_w and rng.random() < 0.55:
            inner_w = rng.choice(odd_w)

    return inner_h, inner_w


def _random_lengths(
    rng: random.Random, total: int, parts: int, min_size: int
) -> list[int]:
    lengths = [min_size] * parts
    for _ in range(total - parts * min_size):
        lengths[rng.randrange(parts)] += 1
    return lengths


def _layout_shape(rng: random.Random, fence_count: int) -> tuple[int, int]:
    if fence_count == 3:
        return rng.choice([(1, 3), (3, 1), (2, 2)])
    if fence_count == 4:
        return 2, 2
    return 2, 3


def _build_slots(
    rng: random.Random, h: int, w: int, rows: int, cols: int
) -> list[tuple[int, int, int, int]] | None:
    usable_h = h - 2 - (rows - 1)
    usable_w = w - 2 - (cols - 1)
    if usable_h < rows * 6 or usable_w < cols * 6:
        return None

    row_heights = _random_lengths(rng, usable_h, rows, 6)
    col_widths = _random_lengths(rng, usable_w, cols, 6)

    slots: list[tuple[int, int, int, int]] = []
    r = 1
    for rh in row_heights:
        c = 1
        for cw in col_widths:
            slots.append((r, c, r + rh - 1, c + cw - 1))
            c += cw + 1
        r += rh + 1
    return slots


def _build_specs(
    rng: random.Random, counts: list[int]
) -> tuple[int, int, list[dict[str, int]]] | None:
    rows, cols = _layout_shape(rng, len(counts))
    min_h = max(GRID_MIN, 2 + (rows - 1) + rows * 6)
    min_w = max(GRID_MIN, 2 + (cols - 1) + cols * 6)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)
    slots = _build_slots(rng, h, w, rows, cols)
    if slots is None:
        return None

    rng.shuffle(slots)
    used_slots = slots[: len(counts)]
    specs: list[dict[str, int]] = []
    for seed_count, slot in zip(counts, used_slots):
        sr0, sc0, sr1, sc1 = slot
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        inner_h, inner_w = _sample_inner_dims(rng, slot_h - 2, slot_w - 2, seed_count)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        r0 = rng.randint(sr0, sr1 - outer_h + 1)
        c0 = rng.randint(sc0, sc1 - outer_w + 1)
        specs.append(
            {
                "seed_count": seed_count,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "r0": r0,
                "c0": c0,
                "r1": r0 + outer_h - 1,
                "c1": c0 + outer_w - 1,
            }
        )
    return h, w, specs


def _quadrant_seed_positions(
    rng: random.Random, inner_h: int, inner_w: int
) -> list[tuple[int, int]]:
    mid_r = inner_h // 2
    mid_c = inner_w // 2

    def near(start: int, end: int, from_high: bool) -> int:
        span = end - start + 1
        offset = rng.randint(0, min(1, span - 1))
        return end - offset if from_high else start + offset

    return [
        (near(0, mid_r - 1, False), near(0, mid_c - 1, False)),
        (near(0, mid_r - 1, False), near(mid_c, inner_w - 1, True)),
        (near(mid_r, inner_h - 1, True), near(0, mid_c - 1, False)),
        (near(mid_r, inner_h - 1, True), near(mid_c, inner_w - 1, True)),
    ]


def _place_seeds(
    grid: np.ndarray, spec: dict[str, int], rng: random.Random
) -> None:
    seed_count = spec["seed_count"]
    r0 = spec["r0"] + 1
    c0 = spec["c0"] + 1
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    colors = rng.sample(SEED_COLORS, seed_count)

    if seed_count == 1:
        positions = [(rng.randrange(inner_h), rng.randrange(inner_w))]
    elif seed_count == 2:
        rows = sorted(rng.sample(range(inner_h), 2))
        positions = [(rows[0], rng.randrange(inner_w)), (rows[1], rng.randrange(inner_w))]
    elif seed_count == 3:
        cols = sorted(rng.sample(range(inner_w), 3))
        positions = [(rng.randrange(inner_h), cols[0]), (rng.randrange(inner_h), cols[1]), (rng.randrange(inner_h), cols[2])]
    else:
        positions = _quadrant_seed_positions(rng, inner_h, inner_w)

    for (dr, dc), color in zip(positions, colors):
        grid[r0 + dr, c0 + dc] = color


def _draw_fence(grid: np.ndarray, spec: dict[str, int]) -> None:
    r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
    grid[r0, c0 : c1 + 1] = FENCE
    grid[r1, c0 : c1 + 1] = FENCE
    grid[r0 : r1 + 1, c0] = FENCE
    grid[r0 : r1 + 1, c1] = FENCE


def _validate_generated(grid: np.ndarray, expected_counts: list[int]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if grid.shape[0] < GRID_MIN or grid.shape[1] < GRID_MIN:
        return False

    try:
        boxes = _find_fences(grid)
    except AssertionError:
        return False

    if len(boxes) != len(expected_counts):
        return False

    found_counts: list[int] = []
    for box in boxes:
        seeds = _collect_seeds(grid, box)
        found_counts.append(len(seeds))
        if len(seeds) == 2 and len({r for r, _, _ in seeds}) != 2:
            return False
        if len(seeds) == 3 and len({c for _, c, _ in seeds}) != 3:
            return False
        if len(seeds) == 4:
            r0, c0, r1, c1 = box
            rel = [(r - (r0 + 1), c - (c0 + 1), color) for r, c, color in seeds]
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            corners = [
                min(rel, key=lambda item: (abs(item[0]) + abs(item[1]), item[0], item[1]))[:2],
                min(rel, key=lambda item: (abs(item[0]) + abs(item[1] - (inner_w - 1)), item[0], item[1]))[:2],
                min(rel, key=lambda item: (abs(item[0] - (inner_h - 1)) + abs(item[1]), item[0], item[1]))[:2],
                min(rel, key=lambda item: (abs(item[0] - (inner_h - 1)) + abs(item[1] - (inner_w - 1)), item[0], item[1]))[:2],
            ]
            if len(set(corners)) != 4:
                return False

    if sorted(found_counts) != sorted(expected_counts):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if not np.array_equal(out[grid == FENCE], grid[grid == FENCE]):
        return False
    return True


def _gen_attempt(rng: random.Random, counts: list[int]) -> np.ndarray | None:
    built = _build_specs(rng, counts)
    if built is None:
        return None

    h, w, placed = built
    grid = np.full((h, w), BG, dtype=int)
    for spec in placed:
        _draw_fence(grid, spec)
    for spec in placed:
        _place_seeds(grid, spec, rng)

    if not _validate_generated(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    counts = _next_seed_counts(rng)
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng, counts)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
