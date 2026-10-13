import itertools
import random
from collections import defaultdict

import numpy as np


BG = 0
AZURE = 8
MAROON = 9
CONTROL_COLORS = {AZURE, MAROON}
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7)


def _marker_info(g: np.ndarray) -> dict[int, dict[str, int | bool]]:
    info: dict[int, dict[str, int | bool]] = {}
    marker_rows = [r for r in range(g.shape[0]) if int(g[r, 0]) in CONTROL_COLORS]
    assert marker_rows, "expected at least one left-edge control marker"

    for rank, r in enumerate(marker_rows):
        control = int(g[r, 0])
        color = int(g[r, 1])
        assert color not in CONTROL_COLORS | {BG}, "marker must point to a path color"
        assert color not in info, "each path color must have exactly one marker"
        info[color] = {"rank": rank, "row": r, "azure": control == AZURE}
    return info


def _crossing_descriptors(
    g: np.ndarray, info: dict[int, dict[str, int | bool]]
) -> list[tuple[int, int, int, str, int, str]]:
    h, w = g.shape
    out: list[tuple[int, int, int, str, int, str]] = []
    for r in range(1, h - 1):
        for c in range(2, w - 1):
            center = int(g[r, c])
            if center not in info:
                continue

            horiz_center = int(g[r, c - 1]) == center and int(g[r, c + 1]) == center
            vert_center = int(g[r - 1, c]) == center and int(g[r + 1, c]) == center
            if horiz_center == vert_center:
                continue

            if horiz_center:
                other = int(g[r - 1, c])
                if other == int(g[r + 1, c]) and other in info and other != center:
                    out.append((r, c, center, "h", other, "v"))
            else:
                other = int(g[r, c - 1])
                if other == int(g[r, c + 1]) and other in info and other != center:
                    out.append((r, c, center, "v", other, "h"))
    return out


def _priority(info: dict[int, dict[str, int | bool]], color: int) -> tuple[int, int]:
    meta = info[color]
    return (1 if bool(meta["azure"]) else 0, -int(meta["rank"]))


def _top_color(info: dict[int, dict[str, int | bool]], a: int, b: int) -> int:
    return a if _priority(info, a) > _priority(info, b) else b


def _erase_triplet(
    erased: set[tuple[int, int]], shape: tuple[int, int], r: int, c: int, orient: str
) -> None:
    h, w = shape
    if orient == "h":
        deltas = ((0, -1), (0, 0), (0, 1))
    else:
        deltas = ((-1, 0), (0, 0), (1, 0))

    for dr, dc in deltas:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and cc > 1:
            erased.add((rr, cc))


def main(grid: np.ndarray) -> np.ndarray:
    """Weave colored polylines by opening a three-cell gap on the under-passing path at each crossing."""
    g = np.asarray(grid, dtype=int)
    info = _marker_info(g)

    erased: set[tuple[int, int]] = set()
    top_pixels: dict[tuple[int, int], int] = {}
    for r, c, color_a, orient_a, color_b, orient_b in _crossing_descriptors(g, info):
        top = _top_color(info, color_a, color_b)
        under = color_b if top == color_a else color_a
        under_orient = orient_b if under == color_b else orient_a
        _erase_triplet(erased, g.shape, r, c, under_orient)
        top_pixels[(r, c)] = top

    out = g.copy()
    for r, c in erased:
        out[r, c] = BG
    for (r, c), color in top_pixels.items():
        out[r, c] = int(color)
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        return [(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)]
    assert c0 == c1
    return [(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)]


def _choose_statuses(n_paths: int, pairs: set[tuple[int, int]]) -> tuple[bool, ...]:
    assignments = [
        bits
        for bits in itertools.product((False, True), repeat=n_paths)
        if any(bits) and not all(bits)
    ]
    assert assignments

    def has_same(bits: tuple[bool, ...]) -> bool:
        return any(bits[a] == bits[b] for a, b in pairs)

    def has_mixed(bits: tuple[bool, ...]) -> bool:
        return any(bits[a] != bits[b] for a, b in pairs)

    if len(pairs) >= 2:
        both = [bits for bits in assignments if has_same(bits) and has_mixed(bits)]
        if both:
            return random.choice(both)

    if pairs:
        want_same = random.random() < 0.5
        filtered = [bits for bits in assignments if (has_same(bits) if want_same else has_mixed(bits))]
        if filtered:
            return random.choice(filtered)

    return random.choice(assignments)


def _build_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(24, 34)
    lane_rows = list(range(2, h - 2, 3))
    max_paths = min(len(PATH_COLORS), len(lane_rows) - 1, 5)
    if max_paths < 3:
        return None

    n_paths = random.randint(3, max_paths)
    start_rows = sorted(random.sample(lane_rows, n_paths))
    colors = random.sample(PATH_COLORS, n_paths)

    col_pool = list(range(4, w - 4, 4))
    if len(col_pool) < 3:
        return None
    n_events = random.randint(3, min(5, len(col_pool)))
    event_cols = sorted(random.sample(col_pool, n_events))

    current_rows = {idx: row for idx, row in enumerate(start_rows)}
    points: dict[int, list[tuple[int, int]]] = {
        idx: [(row, 1)] for idx, row in enumerate(start_rows)
    }
    raw_crossings: list[tuple[int, int, int, int]] = []
    last_mover = None

    for col in event_cols:
        candidates: list[tuple[int, int, tuple[int, ...]]] = []
        for idx in range(n_paths):
            old_row = current_rows[idx]
            occupied = {current_rows[j] for j in range(n_paths) if j != idx}
            for new_row in lane_rows:
                if new_row == old_row or new_row in occupied:
                    continue
                crossed = tuple(
                    j
                    for j in range(n_paths)
                    if j != idx and min(old_row, new_row) < current_rows[j] < max(old_row, new_row)
                )
                if 1 <= len(crossed) <= 2:
                    candidates.append((idx, new_row, crossed))

        if last_mover is not None:
            alt = [cand for cand in candidates if cand[0] != last_mover]
            if alt and random.random() < 0.7:
                candidates = alt
        if not candidates:
            return None

        mover, new_row, crossed = random.choice(candidates)
        old_row = current_rows[mover]
        points[mover].append((old_row, col))
        points[mover].append((new_row, col))
        for other in crossed:
            raw_crossings.append((current_rows[other], col, mover, other))
        current_rows[mover] = new_row
        last_mover = mover

    if len(raw_crossings) < 2:
        return None

    end_min = event_cols[-1] + 2
    if end_min > w - 2:
        return None
    for idx in range(n_paths):
        points[idx].append((current_rows[idx], random.randint(end_min, w - 2)))

    crossing_pairs = {tuple(sorted((a, b))) for _r, _c, a, b in raw_crossings}
    statuses = _choose_statuses(n_paths, crossing_pairs)

    occupancy: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, color in enumerate(colors):
        path = points[idx]
        for a, b in zip(path, path[1:]):
            for cell in _segment_cells(a, b):
                occupancy[cell].add(color)

    grid = np.full((h, w), BG, dtype=int)
    planned_crossings: set[tuple[int, int]] = set()
    for (r, c), cell_colors in occupancy.items():
        if len(cell_colors) == 1:
            grid[r, c] = next(iter(cell_colors))
        elif len(cell_colors) == 2:
            grid[r, c] = random.choice(tuple(cell_colors))
            planned_crossings.add((r, c))
        else:
            return None

    for idx, start_row in enumerate(start_rows):
        grid[start_row, 0] = AZURE if statuses[idx] else MAROON
        grid[start_row, 1] = colors[idx]

    info = _marker_info(grid)
    detected = _crossing_descriptors(grid, info)
    detected_positions = {(r, c) for r, c, _a, _oa, _b, _ob in detected}
    if planned_crossings != detected_positions:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate weaving-path input")
