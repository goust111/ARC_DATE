from __future__ import annotations

from collections import Counter
import math

import numpy as np


AZURE = 8
BLUE = 1
MAROON = 9
GREY = 5

INPUT_COLOR_POOL = [0, 2, 3, 4, 6, 7]
MAX_GEN_ATTEMPTS = 400
MAX_SEARCH_SOLUTIONS = 24

RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_direction(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int], int]:
    corners = [
        ((0, 0), (-1, -1)),
        ((0, grid.shape[1] - 1), (-1, 1)),
        ((grid.shape[0] - 1, 0), (1, -1)),
        ((grid.shape[0] - 1, grid.shape[1] - 1), (1, 1)),
    ]
    hits = [(pos, delta, int(grid[pos])) for pos, delta in corners if int(grid[pos]) != bg]
    if len(hits) != 1:
        raise ValueError("expected exactly one non-background corner marker")
    return hits[0]


def _square_color(grid: np.ndarray, bg: int, marker_pos: tuple[int, int]) -> int:
    counts = Counter(map(int, grid.flatten()))
    counts[int(grid[marker_pos])] -= 1
    counts[bg] = -1
    color, freq = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if freq <= 0:
        raise ValueError("failed to identify foreground square color")
    return int(color)


def _fg_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected foreground cells")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_map_from_placements(shape: tuple[int, int], size: int, placements: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0 in placements:
        out[r0 : r0 + size, c0 : c0 + size] += 1
    return out


def _candidate_cover_data(mask: np.ndarray, size: int) -> tuple[list[tuple[int, int]], list[int], list[list[int]], int]:
    h, w = mask.shape
    if size > h or size > w:
        return [], [], [], 0

    fg_coords = np.argwhere(mask)
    n_fg = int(fg_coords.shape[0])
    if n_fg < size * size:
        return [], [], [], 0

    index_grid = -np.ones((h, w), dtype=int)
    for idx, (r, c) in enumerate(fg_coords):
        index_grid[int(r), int(c)] = int(idx)

    placements: list[tuple[int, int]] = []
    cover_masks: list[int] = []
    cover_lists = [[] for _ in range(n_fg)]

    for r0 in range(h - size + 1):
        for c0 in range(w - size + 1):
            window = mask[r0 : r0 + size, c0 : c0 + size]
            if not bool(window.all()):
                continue
            bits = 0
            local_indices = index_grid[r0 : r0 + size, c0 : c0 + size].ravel()
            for cell_idx in local_indices:
                bits |= 1 << int(cell_idx)
            cand_idx = len(placements)
            placements.append((int(r0), int(c0)))
            cover_masks.append(bits)
            for cell_idx in local_indices:
                cover_lists[int(cell_idx)].append(cand_idx)

    full_bits = (1 << n_fg) - 1
    union_bits = 0
    for bits in cover_masks:
        union_bits |= bits
    if union_bits != full_bits:
        return [], [], [], 0

    return placements, cover_masks, cover_lists, full_bits


def _minimum_covers_for_size(mask: np.ndarray, size: int, global_best: int | float) -> list[tuple[tuple[int, int], ...]]:
    placements, cover_masks, cover_lists, full_bits = _candidate_cover_data(mask, size)
    if not placements:
        return []

    best = global_best
    solutions: list[tuple[tuple[int, int], ...]] = []
    memo: dict[int, int] = {}

    def search(uncovered: int, chosen: list[int]) -> None:
        nonlocal best, solutions
        if not uncovered:
            k = len(chosen)
            solution = tuple(sorted((placements[idx] for idx in chosen)))
            if k < best:
                best = k
                solutions = [solution]
            elif k == best and solution not in solutions and len(solutions) < MAX_SEARCH_SOLUTIONS:
                solutions.append(solution)
            return

        if len(chosen) >= best:
            return

        prev = memo.get(uncovered)
        if prev is not None and prev <= len(chosen):
            return
        memo[uncovered] = len(chosen)

        remaining = uncovered.bit_count()
        best_gain = 0
        pixel = None
        options: list[int] | None = None
        bits = uncovered
        while bits:
            lsb = bits & -bits
            cell = lsb.bit_length() - 1
            cell_options = [idx for idx in cover_lists[cell] if cover_masks[idx] & uncovered]
            if not cell_options:
                return
            if options is None or len(cell_options) < len(options):
                options = cell_options
                pixel = cell
                if len(options) == 1:
                    break
            bits ^= lsb

        assert pixel is not None and options is not None
        for idx in options:
            gain = (cover_masks[idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
        if best_gain == 0:
            return

        lower_bound = math.ceil(remaining / best_gain)
        if len(chosen) + lower_bound > best:
            return

        options = sorted(
            options,
            key=lambda idx: (
                -int((cover_masks[idx] & uncovered).bit_count()),
                placements[idx][0],
                placements[idx][1],
            ),
        )
        for idx in options:
            chosen.append(idx)
            search(uncovered & ~cover_masks[idx], chosen)
            chosen.pop()

    search(full_bits, [])
    return solutions


def _recover_square_cover(mask: np.ndarray) -> tuple[int, tuple[tuple[int, int], ...], np.ndarray]:
    r0, c0, r1, c1 = _fg_bbox(mask)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = cropped.shape

    best_k = float("inf")
    candidates: list[tuple[int, tuple[tuple[int, int], ...], np.ndarray]] = []

    for size in range(2, min(h, w) + 1):
        covers = _minimum_covers_for_size(cropped, size, best_k)
        if not covers:
            continue
        k = len(covers[0])
        if k > best_k:
            continue
        rebased = []
        for cover in covers:
            placements = tuple((r0 + rr, c0 + cc) for rr, cc in cover)
            count_map = _count_map_from_placements(mask.shape, size, placements)
            rebased.append((size, placements, count_map))
        if k < best_k:
            best_k = k
            candidates = rebased
        else:
            candidates.extend(rebased)

    if not candidates:
        raise ValueError("failed to recover a square decomposition")

    by_count_map: dict[bytes, tuple[int, tuple[tuple[int, int], ...], np.ndarray]] = {}
    for size, placements, count_map in candidates:
        key = count_map.tobytes()
        incumbent = by_count_map.get(key)
        if incumbent is None or (size, placements) > (incumbent[0], incumbent[1]):
            by_count_map[key] = (size, placements, count_map)

    if len(by_count_map) == 1:
        return next(iter(by_count_map.values()))

    return max(
        by_count_map.values(),
        key=lambda item: (item[0], len(item[1]), item[1]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor overlapped equal squares by stack depth and cast a one-step diagonal shadow from a corner marker.

    Input structure:
    - The grid has a dominant background color, one singleton corner marker, and one foreground color.
    - The foreground is the union of several same-sized solid squares of that single color, with possible overlaps.
    - The unique non-background corner pixel indicates the shadow direction: its corner chooses one diagonal step.

    Rule:
    - Remove the corner marker and infer the foreground squares as a minimum cover by equal-size solid squares.
    - Count how many recovered squares cover each foreground cell.
    - Recolor coverage depth 1 as azure, depth 2 as blue, and depth 3 or more as maroon.
    - Shift the whole foreground mask by one cell along the marker's diagonal and paint grey shadow only onto background cells outside the foreground.

    Key insight:
    - The lonely corner pixel is not part of the object set; it encodes a global diagonal shadow direction, while the single-color foreground must be decomposed back into overlapping equal squares to recover stack depth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    marker_pos, (dr, dc), _marker_color = _corner_direction(g, bg)
    fg_color = _square_color(g, bg, marker_pos)

    mask = g == fg_color
    if not bool(mask.any()):
        raise ValueError("expected square foreground")

    _size, _placements, count_map = _recover_square_cover(mask)

    out = np.full(g.shape, bg, dtype=int)
    out[count_map == 1] = AZURE
    out[count_map == 2] = BLUE
    out[count_map >= 3] = MAROON

    shadow_mask = np.zeros_like(mask, dtype=bool)
    src = np.argwhere(count_map > 0)
    for r, c in src:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and count_map[rr, cc] == 0:
            shadow_mask[rr, cc] = True
    out[shadow_mask] = GREY
    return out


def _sample_colors() -> tuple[int, int, int]:
    chosen = RNG.choice(INPUT_COLOR_POOL, size=3, replace=False)
    return int(chosen[0]), int(chosen[1]), int(chosen[2])


def _placements_count_map(shape: tuple[int, int], size: int, placements: list[tuple[int, int]]) -> np.ndarray:
    return _count_map_from_placements(shape, size, tuple(placements))


def _square_has_unique_pixel(count_map: np.ndarray, size: int, placement: tuple[int, int]) -> bool:
    r0, c0 = placement
    return bool((count_map[r0 : r0 + size, c0 : c0 + size] == 1).any())


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            nxt = (rr, cc)
            if (
                0 <= rr < mask.shape[0]
                and 0 <= cc < mask.shape[1]
                and bool(mask[rr, cc])
                and nxt not in seen
            ):
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _sample_pair_family(size: int) -> list[tuple[int, int]] | None:
    k = int(RNG.integers(2, 4))
    placements = [(0, 0)]
    for _ in range(1, k):
        placed = False
        for _ in range(40):
            base_r, base_c = placements[int(RNG.integers(len(placements)))]
            dr = int(RNG.integers(-(size - 1), size))
            dc = int(RNG.integers(-(size - 1), size))
            if dr == 0 and dc == 0:
                continue
            if abs(dr) >= size or abs(dc) >= size:
                continue
            nr = base_r + dr
            nc = base_c + dc
            if (nr, nc) in placements:
                continue
            placements.append((nr, nc))
            count_map = _placements_count_map((size * 4, size * 4), size, placements)
            if int(count_map.max()) <= 2:
                placed = True
                break
            placements.pop()
        if not placed:
            return None
    return placements


def _sample_triple_family(size: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        offsets = [(0, 0)]
        while len(offsets) < 3:
            dr = int(RNG.integers(-(size - 1), size))
            dc = int(RNG.integers(-(size - 1), size))
            if (dr, dc) == (0, 0) or (dr, dc) in offsets:
                continue
            offsets.append((dr, dc))

        rows = [r for r, _ in offsets]
        cols = [c for _, c in offsets]
        if max(rows) - min(rows) >= size or max(cols) - min(cols) >= size:
            continue

        placements = offsets
        count_map = _placements_count_map((size * 4, size * 4), size, placements)
        if int(count_map.max()) != 3:
            continue
        return placements
    return None


def _normalize_placements(placements: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in placements)
    min_c = min(c for _, c in placements)
    return [(r - min_r, c - min_c) for r, c in placements]


def _layout_in_grid(size: int, placements: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    placements = _normalize_placements(placements)
    max_r = max(r for r, _ in placements) + size
    max_c = max(c for _, c in placements) + size

    top = int(RNG.integers(2, 6))
    left = int(RNG.integers(2, 6))
    bottom = int(RNG.integers(2, 6))
    right = int(RNG.integers(2, 6))

    h = max_r + top + bottom
    w = max_c + left + right
    if h > 30 or w > 30:
        return None

    shifted = [(top + r, left + c) for r, c in placements]
    return h, w, shifted


def _direction_marker(shape: tuple[int, int], direction_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    options = [
        ((0, 0), (-1, -1)),
        ((0, w - 1), (-1, 1)),
        ((h - 1, 0), (1, -1)),
        ((h - 1, w - 1), (1, 1)),
    ]
    pos, delta = options[direction_index]
    return (int(pos[0]), int(pos[1])), delta


def _build_input_from_placements(
    h: int,
    w: int,
    size: int,
    placements: list[tuple[int, int]],
    bg: int,
    square: int,
    marker: int,
    direction_index: int,
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for r0, c0 in placements:
        grid[r0 : r0 + size, c0 : c0 + size] = square
    marker_pos, _delta = _direction_marker((h, w), direction_index)
    grid[marker_pos] = marker
    return grid


def _valid_generated_case(grid: np.ndarray, size: int, placements: list[tuple[int, int]], expect_triple: bool) -> bool:
    if max(grid.shape) > 30:
        return False

    bg = _most_common_color(grid)
    marker_pos, _delta, _marker_color = _corner_direction(grid, bg)
    fg_color = _square_color(grid, bg, marker_pos)
    mask = grid == fg_color

    if not _connected(mask):
        return False

    actual_count = _count_map_from_placements(grid.shape, size, tuple(placements))
    if int(actual_count.max()) < 2:
        return False
    if expect_triple != (int(actual_count.max()) >= 3):
        return False
    if int(actual_count.max()) > 3:
        return False
    if not all(_square_has_unique_pixel(actual_count, size, placement) for placement in placements):
        return False

    recovered_size, recovered_placements, recovered_count = _recover_square_cover(mask)
    if recovered_size != size:
        return False
    if recovered_placements != tuple(sorted(placements)):
        return False
    if not np.array_equal(actual_count, recovered_count):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = int(RNG.integers(3, 6))
    expect_triple = bool(RNG.integers(2))
    direction_index = int(RNG.integers(4))
    bg, square, marker = _sample_colors()

    sampler = _sample_triple_family if expect_triple else _sample_pair_family
    placements = sampler(size)
    if placements is None:
        return None

    placements = _normalize_placements(placements)
    count_probe = _placements_count_map((size * 4, size * 4), size, placements)
    if expect_triple and int(count_probe.max()) != 3:
        return None
    if not expect_triple and int(count_probe.max()) != 2:
        return None

    layout = _layout_in_grid(size, placements)
    if layout is None:
        return None
    h, w, shifted = layout

    grid = _build_input_from_placements(h, w, size, shifted, bg, square, marker, direction_index)
    if not _valid_generated_case(grid, size, shifted, expect_triple):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadowed-overlap puzzle")
