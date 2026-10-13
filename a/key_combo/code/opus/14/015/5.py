from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


GRID_SIZE = 14
SEPARATOR_ROWS = (4, 9)
SEPARATOR_COLS = (4, 9)
REGION_SIZE = 4
REGION_STARTS = (0, 5, 10)
PALETTE = (1, 2, 3, 5, 6, 7, 9)
GEN_ATTEMPTS = 400

NON_SIGNAL_COUNT_PATTERNS = (
    (4, 4),
    (3, 5),
    (2, 6),
    (2, 2, 4),
    (2, 3, 3),
    (2, 2, 2, 2),
)

HOLE_PATTERNS_BY_SIZE = {
    1: (
        ((1, 1),),
        ((1, 2),),
        ((2, 1),),
        ((2, 2),),
    ),
    2: (
        ((1, 1), (1, 2)),
        ((2, 1), (2, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (2, 2)),
        ((1, 1), (2, 2)),
        ((1, 2), (2, 1)),
    ),
    3: (
        ((1, 1), (1, 2), (2, 1)),
        ((1, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (2, 2)),
        ((1, 2), (2, 1), (2, 2)),
    ),
}

_GEN_BRANCH_TOGGLE = 0
_RECENT_OUTPUT_SIGNATURES: list[bytes] = []


def _unique_extreme_color(counts: Counter[int], *, choose_max: bool, message: str) -> int:
    assert counts, message
    target = max(counts.values()) if choose_max else min(counts.values())
    winners = [color for color, count in counts.items() if count == target]
    assert len(winners) == 1, message
    return int(winners[0])


def _partition_segments(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + separators + [size]
    segments = []
    for lo, hi in zip(bounds, bounds[1:]):
        if hi - lo <= 1:
            continue
        segments.append((lo + 1, hi))
    return segments


def _find_regions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = map(int, grid.shape)
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == 0)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == 0)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 lattice split by two black rows and two black columns"

    row_segments = _partition_segments(h, sep_rows)
    col_segments = _partition_segments(w, sep_cols)
    assert len(row_segments) == 3 and len(col_segments) == 3, "expected exactly nine regions"
    assert all(r1 - r0 == REGION_SIZE for r0, r1 in row_segments), "each region must be 4 rows tall"
    assert all(c1 - c0 == REGION_SIZE for c0, c1 in col_segments), "each region must be 4 cols wide"

    regions = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            regions.append((r0, r1, c0, c1))
    return regions


def _hole_component_is_valid(region: np.ndarray, cells: list[tuple[int, int]], main_color: int) -> bool:
    h, w = map(int, region.shape)
    comp = set(cells)
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
        return False

    for r, c in comp:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in comp:
                continue
            if not (0 <= nr < h and 0 <= nc < w):
                return False
            if int(region[nr, nc]) != main_color:
                return False
    return True


def _has_internal_hole(region: np.ndarray, main_color: int) -> bool:
    h, w = map(int, region.shape)
    zeros = region == 0
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not zeros[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and zeros[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if _hole_component_is_valid(region, cells, main_color):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the 3x3 region lattice while replacing holey regions by the unique signal color.

    Input structure:
    - A 14x14 grid is split by two full black separator rows and two full black separator columns.
    - The separators divide the grid into nine 4x4 regions arranged as a 3x3 lattice.
    - Each region is mostly one non-black main color, optionally with internal black hole pixels.

    Rule:
    - Detect the nine 4x4 regions from the separator lattice and find each region's main non-black color.
    - Mark a region as holey iff it contains a 4-connected black component that stays away from the region border and is fully enclosed by the main color.
    - Find the unique main color that appears in exactly one region; this is the signal color.
    - Output a 3x3 grid in region order: signal-color region becomes black, non-signal holey regions become the signal color, and non-holey non-signal regions keep their main color.

    Key insight:
    - Two independent analyses meet in the compression step: hole detection decides which regions are rewritten, while global main-color rarity decides which color acts as the replacement signal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _find_regions(g)
    main_colors: list[int] = []
    has_hole: list[bool] = []
    for r0, r1, c0, c1 in regions:
        region = g[r0:r1, c0:c1]
        counts = Counter(map(int, region[region != 0].flat))
        main_color = _unique_extreme_color(counts, choose_max=True, message="each region must have a unique main non-black color")
        main_colors.append(main_color)
        has_hole.append(_has_internal_hole(region, main_color))

    signal_counts = Counter(main_colors)
    signal_color = _unique_extreme_color(signal_counts, choose_max=False, message="expected a unique signal color appearing in exactly one region")
    assert signal_counts[signal_color] == 1, "signal color must appear in exactly one region"

    out = np.zeros((3, 3), dtype=int)
    for index, (color, holey) in enumerate(zip(main_colors, has_hole)):
        rr, cc = divmod(index, 3)
        if color == signal_color:
            out[rr, cc] = 0
        elif holey:
            out[rr, cc] = signal_color
        else:
            out[rr, cc] = color
    return out


def _hole_count_sequence(n_holes: int) -> list[int]:
    base = [1, 2, 3]
    random.shuffle(base)
    seq = [base[i % len(base)] for i in range(n_holes)]
    random.shuffle(seq)
    return seq


def _choose_signal_hole_flag() -> bool:
    global _GEN_BRANCH_TOGGLE
    flag = bool(_GEN_BRANCH_TOGGLE % 2)
    _GEN_BRANCH_TOGGLE += 1
    return flag


def _choose_region_colors() -> tuple[list[int], int]:
    signal_idx = random.randrange(9)
    non_signal_counts = random.choice(NON_SIGNAL_COUNT_PATTERNS)
    chosen = random.sample(PALETTE, 1 + len(non_signal_counts))
    signal_color = chosen[0]
    other_colors = chosen[1:]

    colors = [signal_color] * 9
    remaining = [idx for idx in range(9) if idx != signal_idx]
    random.shuffle(remaining)
    cursor = 0
    for color, count in zip(other_colors, non_signal_counts):
        for idx in remaining[cursor : cursor + count]:
            colors[idx] = color
        cursor += count
    return colors, signal_idx


def _choose_hole_flags(region_colors: list[int], signal_idx: int) -> list[bool] | None:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, color in enumerate(region_colors):
        if idx != signal_idx:
            groups[color].append(idx)

    mixed_candidates = [indices for indices in groups.values() if len(indices) >= 2]
    if not mixed_candidates:
        return None

    flags = [False] * 9
    flags[signal_idx] = _choose_signal_hole_flag()

    mixed_group = random.choice(mixed_candidates)
    mixed_pair = random.sample(mixed_group, 2)
    flags[mixed_pair[0]] = True
    flags[mixed_pair[1]] = False

    max_non_signal_holes = 6 if flags[signal_idx] else 7
    target_non_signal_holes = random.randint(2, max_non_signal_holes)
    remaining = [idx for idx in range(9) if idx not in (signal_idx, mixed_pair[0], mixed_pair[1])]
    random.shuffle(remaining)
    for idx in remaining[: target_non_signal_holes - 1]:
        flags[idx] = True

    non_signal_holes = sum(flags[idx] for idx in range(9) if idx != signal_idx)
    no_hole_total = sum(not flag for flag in flags)
    if non_signal_holes < 2 or no_hole_total < 2:
        return None

    has_mixed_color = any(any(flags[i] for i in idxs) and any(not flags[i] for i in idxs) for idxs in groups.values())
    if not has_mixed_color:
        return None
    return flags


def _render_input(region_colors: list[int], hole_flags: list[bool]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in SEPARATOR_ROWS:
        grid[r, :] = 0
    for c in SEPARATOR_COLS:
        grid[:, c] = 0

    hole_regions = [idx for idx, flag in enumerate(hole_flags) if flag]
    hole_sizes = _hole_count_sequence(len(hole_regions))
    hole_size_by_region = {idx: size for idx, size in zip(hole_regions, hole_sizes)}

    for idx, (r0, c0) in enumerate((r, c) for r in REGION_STARTS for c in REGION_STARTS):
        color = region_colors[idx]
        grid[r0 : r0 + REGION_SIZE, c0 : c0 + REGION_SIZE] = color
        if hole_flags[idx]:
            pattern = random.choice(HOLE_PATTERNS_BY_SIZE[hole_size_by_region[idx]])
            for dr, dc in pattern:
                grid[r0 + dr, c0 + dc] = 0
    return grid


def _expected_output(region_colors: list[int], hole_flags: list[bool], signal_idx: int) -> np.ndarray:
    signal_color = region_colors[signal_idx]
    out = np.zeros((3, 3), dtype=int)
    for idx, (color, holey) in enumerate(zip(region_colors, hole_flags)):
        r, c = divmod(idx, 3)
        if idx == signal_idx:
            out[r, c] = 0
        elif holey:
            out[r, c] = signal_color
        else:
            out[r, c] = color
    return out


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    region_colors: list[int],
    hole_flags: list[bool],
    signal_idx: int,
) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not (np.all(grid[list(SEPARATOR_ROWS), :] == 0) and np.all(grid[:, list(SEPARATOR_COLS)] == 0)):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    parsed_regions = _find_regions(grid)
    parsed_colors: list[int] = []
    parsed_holes: list[bool] = []
    hole_sizes: list[int] = []
    for r0, r1, c0, c1 in parsed_regions:
        region = grid[r0:r1, c0:c1]
        if np.any(region[0, :] == 0) or np.any(region[-1, :] == 0) or np.any(region[:, 0] == 0) or np.any(region[:, -1] == 0):
            return False
        nz = region[region != 0]
        counts = Counter(map(int, nz.flat))
        if len(counts) != 1:
            return False
        parsed_colors.append(int(nz[0]))
        parsed_holes.append(_has_internal_hole(region, int(nz[0])))
        if parsed_holes[-1]:
            hole_sizes.append(int((region == 0).sum()))

    if parsed_colors != region_colors or parsed_holes != hole_flags:
        return False

    color_counts = Counter(parsed_colors)
    signal_candidates = [color for color, count in color_counts.items() if count == 1]
    if len(signal_candidates) != 1 or signal_candidates[0] != region_colors[signal_idx]:
        return False

    non_signal_holes = sum(parsed_holes[i] and parsed_colors[i] != signal_candidates[0] for i in range(9))
    if non_signal_holes < 2:
        return False
    if sum(not hole for hole in parsed_holes) < 2:
        return False
    if len(set(hole_sizes)) < 2:
        return False

    by_color: dict[int, list[int]] = defaultdict(list)
    for idx, color in enumerate(parsed_colors):
        by_color[color].append(idx)
    if not any(len(idxs) >= 2 and any(parsed_holes[i] for i in idxs) and any(not parsed_holes[i] for i in idxs) for idxs in by_color.values()):
        return False

    signal_color = signal_candidates[0]
    signal_cell = divmod(signal_idx, 3)
    if expected[signal_cell] != 0:
        return False
    if int((expected == signal_color).sum()) < 2:
        return False
    if not np.any((expected != 0) & (expected != signal_color)):
        return False
    return True


def _output_signature(out: np.ndarray) -> bytes:
    return out.tobytes() + str(out.shape).encode()


def _remember_output_signature(signature: bytes) -> None:
    _RECENT_OUTPUT_SIGNATURES.append(signature)
    if len(_RECENT_OUTPUT_SIGNATURES) > 16:
        del _RECENT_OUTPUT_SIGNATURES[0]


def _gen_attempt() -> np.ndarray | None:
    region_colors, signal_idx = _choose_region_colors()
    hole_flags = _choose_hole_flags(region_colors, signal_idx)
    if hole_flags is None:
        return None

    grid = _render_input(region_colors, hole_flags)
    expected = _expected_output(region_colors, hole_flags, signal_idx)

    signature = _output_signature(expected)
    if signature in _RECENT_OUTPUT_SIGNATURES:
        return None
    if not _validate_semantics(grid, expected, region_colors, hole_flags, signal_idx):
        return None

    _remember_output_signature(signature)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-5 input")
