from __future__ import annotations

from collections import Counter
import random

import numpy as np


Interval = tuple[int, int]
Coord = tuple[int, int]

_COLORS = tuple(range(10))
_GEN_RETRIES = 700


def _runs(indices: list[int]) -> list[Interval]:
    if not indices:
        return []
    ordered = sorted(set(int(i) for i in indices))
    out: list[Interval] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def _content_intervals(length: int, separator_indices: list[int]) -> list[Interval]:
    intervals: list[Interval] = []
    start = 0
    for sep0, sep1 in _runs(separator_indices):
        if start <= sep0 - 1:
            intervals.append((start, sep0 - 1))
        start = sep1 + 1
    if start <= int(length) - 1:
        intervals.append((start, int(length) - 1))
    return intervals


def _partition(grid: np.ndarray) -> tuple[int, list[Interval], list[Interval]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, list[Interval], list[Interval]]] = []

    for color in set(map(int, grid.ravel())):
        full_rows = [r for r in range(height) if np.all(grid[r, :] == int(color))]
        full_cols = [c for c in range(width) if np.all(grid[:, c] == int(color))]
        if not full_rows or not full_cols:
            continue

        row_intervals = _content_intervals(height, full_rows)
        col_intervals = _content_intervals(width, full_cols)
        if len(row_intervals) != 2 or len(col_intervals) != 2:
            continue

        row_sizes = {end - begin + 1 for begin, end in row_intervals}
        col_sizes = {end - begin + 1 for begin, end in col_intervals}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if next(iter(row_sizes)) != next(iter(col_sizes)):
            continue

        candidates.append((int(color), row_intervals, col_intervals))

    assert len(candidates) == 1, "expected one frame color dividing the grid into four equal square regions"
    return candidates[0]


def _patch(grid: np.ndarray, row_interval: Interval, col_interval: Interval) -> np.ndarray:
    r0, r1 = row_interval
    c0, c1 = col_interval
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _rotation_to_match(anchor: np.ndarray, patch: np.ndarray) -> int:
    scores = [int(np.count_nonzero(anchor != np.rot90(patch, k))) for k in range(4)]
    best = min(scores)
    assert scores.count(best) == 1, "expected a unique best relative rotation"
    return int(scores.index(best))


def _majority_grid(aligned: list[np.ndarray]) -> np.ndarray:
    shape = aligned[0].shape
    out = np.empty(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            counts = Counter(int(patch[r, c]) for patch in aligned)
            ranked = counts.most_common()
            assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
            out[r, c] = int(ranked[0][0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair four noisy rotated copies by aligning them and taking their per-cell consensus.

    Input structure:
    - A frame color forms complete rows and complete columns that split the grid into
      four equal square regions.
    - The four regions contain the same hidden square pattern in rotated orientations,
      with a few wrong pixels in every region.
    - The clean pattern does not appear as an intact region, and the region rotations
      are not encoded by their positions.

    Rule:
    - Use pairwise grid comparison to find the unique rotation that aligns every region
      to the upper-left region's orientation.
    - Rotate all four copies into that shared orientation and restore each cell by the
      unique majority color among the aligned copies.
    - Rotate the restored consensus back to each region's original orientation and
      replace the region contents, leaving all frame rows and columns unchanged.

    Key insight:
    - No quadrant is the answer by itself; the clean pattern is the cross-region
      consensus after solving the hidden relative rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _frame_color, row_intervals, col_intervals = _partition(g)

    patches = [
        _patch(g, row_interval, col_interval).copy()
        for row_interval in row_intervals
        for col_interval in col_intervals
    ]
    assert len(patches) == 4
    assert len({patch.shape for patch in patches}) == 1

    rotations = [0] + [_rotation_to_match(patches[0], patch) for patch in patches[1:]]
    aligned = [np.rot90(patch, rot) for patch, rot in zip(patches, rotations)]
    consensus = _majority_grid(aligned)

    out = g.copy()
    idx = 0
    for row_interval in row_intervals:
        for col_interval in col_intervals:
            restored = np.rot90(consensus, (-rotations[idx]) % 4)
            r0, r1 = row_interval
            c0, c1 = col_interval
            out[r0 : r1 + 1, c0 : c1 + 1] = restored
            idx += 1
    return out


# --------------- generator ---------------


def _rotational_distance_ok(base: np.ndarray) -> bool:
    size = int(base.shape[0])
    min_required = max(8, (size * size) // 5)
    for k in range(1, 4):
        if int(np.count_nonzero(base != np.rot90(base, k))) < min_required:
            return False
    return True


def _paint_random_walk(base: np.ndarray, color: int, steps: int) -> None:
    n = int(base.shape[0])
    r = random.randrange(n)
    c = random.randrange(n)
    for _ in range(int(steps)):
        base[r, c] = int(color)
        moves = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                moves.append((nr, nc))
        r, c = random.choice(moves)


def _paint_small_motif(base: np.ndarray, cells: list[Coord], top: int, left: int, color: int) -> None:
    n = int(base.shape[0])
    for dr, dc in cells:
        r = int(top) + int(dr)
        c = int(left) + int(dc)
        if 0 <= r < n and 0 <= c < n:
            base[r, c] = int(color)


def _make_base_pattern(size: int, bg: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(250):
        base = np.full((size, size), int(bg), dtype=int)
        target = random.randint(max(12, size * size // 4), max(13, size * size // 2))

        while int(np.count_nonzero(base != bg)) < target:
            color = random.choice(colors)
            if random.random() < 0.70:
                _paint_random_walk(base, color, random.randint(size, size * 2))
            else:
                r = random.randrange(size)
                c = random.randrange(size)
                base[r, c] = int(color)

        motifs = [
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 2)],
        ]
        random.shuffle(motifs)
        _paint_small_motif(base, motifs[0], random.randrange(0, size - 2), random.randrange(0, size - 1), colors[0])
        _paint_small_motif(base, motifs[1], random.randrange(0, size - 1), random.randrange(0, size - 1), colors[1])
        _paint_small_motif(base, motifs[2], random.randrange(0, size - 1), random.randrange(0, size - 2), random.choice(colors))

        foreground = int(np.count_nonzero(base != bg))
        if not (size * size // 5 <= foreground <= (size * size * 3) // 5):
            continue
        if len({int(v) for v in base.ravel() if int(v) != bg}) < 2:
            continue
        if _rotational_distance_ok(base):
            return base
    return None


def _rotate_coord(coord: Coord, size: int, k: int) -> Coord:
    r, c = coord
    rr, cc = int(r), int(c)
    for _ in range(int(k) % 4):
        rr, cc = size - 1 - cc, rr
    return rr, cc


def _sample_noise_coords(base: np.ndarray, bg: int) -> list[list[Coord]]:
    size = int(base.shape[0])
    all_fg = [tuple(map(int, rc)) for rc in np.argwhere(base != int(bg))]
    all_bg = [tuple(map(int, rc)) for rc in np.argwhere(base == int(bg))]
    if len(all_fg) < 2 or len(all_bg) < 2:
        return []

    used: set[Coord] = set()
    specs: list[list[Coord]] = [[] for _ in range(4)]

    forced = [
        random.choice(all_fg),
        random.choice(all_bg),
        random.choice(all_fg),
        random.choice(all_bg),
    ]
    for idx, coord in enumerate(forced):
        if coord in used:
            pool = all_fg if int(base[coord]) != int(bg) else all_bg
            choices = [cell for cell in pool if cell not in used]
            if not choices:
                return []
            coord = random.choice(choices)
        specs[idx].append(coord)
        used.add(coord)

    max_extra = max(0, min(3, (size * size) // 28))
    all_cells = [tuple(map(int, rc)) for rc in np.argwhere(np.ones(base.shape, dtype=bool))]
    for idx in range(4):
        extra_count = random.randint(0, max_extra)
        choices = [cell for cell in all_cells if cell not in used]
        if len(choices) < extra_count:
            return []
        for coord in random.sample(choices, extra_count):
            specs[idx].append(coord)
            used.add(coord)

    return specs


def _corrupt_copy(clean: np.ndarray, base_noise: list[Coord], angle: int, palette: list[int], bg: int) -> np.ndarray:
    size = int(clean.shape[0])
    out = clean.copy()
    for base_coord in base_noise:
        r, c = _rotate_coord(base_coord, size, angle)
        current = int(out[r, c])
        if current == int(bg):
            choices = [color for color in palette if color != int(bg)]
        else:
            choices = [color for color in palette if color != current]
        out[r, c] = random.choice(choices)
    return out


def _assemble_grid(frame: int, panels: list[np.ndarray], border: bool) -> np.ndarray:
    size = int(panels[0].shape[0])
    if border:
        grid_size = 2 * size + 3
        grid = np.full((grid_size, grid_size), int(frame), dtype=int)
        row_starts = [1, size + 2]
        col_starts = [1, size + 2]
    else:
        grid_size = 2 * size + 1
        grid = np.full((grid_size, grid_size), int(frame), dtype=int)
        row_starts = [0, size + 1]
        col_starts = [0, size + 1]

    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            grid[r0 : r0 + size, c0 : c0 + size] = panels[idx]
            idx += 1
    return grid


def _expected_output(frame: int, base: np.ndarray, angles: list[int], border: bool) -> np.ndarray:
    clean_panels = [np.rot90(base, angle) for angle in angles]
    return _assemble_grid(frame, clean_panels, border)


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(6, 13)
    border = bool(random.getrandbits(1))
    if not border and size > 14:
        return None

    bg, frame = random.sample(_COLORS, 2)
    available = [color for color in _COLORS if color not in (bg, frame)]
    pattern_colors = random.sample(available, random.randint(2, min(4, len(available))))
    palette = [int(bg)] + [int(color) for color in pattern_colors]

    base = _make_base_pattern(size, int(bg), [int(color) for color in pattern_colors])
    if base is None:
        return None

    angles = list(range(4))
    random.shuffle(angles)
    if tuple(angles) == (0, 1, 2, 3):
        return None

    noise_specs = _sample_noise_coords(base, int(bg))
    if not noise_specs or not all(spec for spec in noise_specs):
        return None

    panels: list[np.ndarray] = []
    for angle, spec in zip(angles, noise_specs):
        clean = np.rot90(base, angle)
        panels.append(_corrupt_copy(clean, spec, angle, palette, int(bg)))

    inp = _assemble_grid(int(frame), panels, border)
    expected = _expected_output(int(frame), base, angles, border)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    for panel, angle in zip(panels, angles):
        if np.array_equal(panel, np.rot90(base, angle)):
            return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-consensus repair input")
