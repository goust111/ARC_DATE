from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
FRAME_COLORS = (1, 2, 3)
FRAME_GRID = 3
GAP = 1
MAX_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask_1d.astype(bool).tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _extract_frames(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int, np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_runs = _runs(np.any(g != BACKGROUND, axis=1))
    col_runs = _runs(np.any(g != BACKGROUND, axis=0))
    assert len(row_runs) == FRAME_GRID and len(col_runs) == FRAME_GRID

    frame_h = row_runs[0][1] - row_runs[0][0] + 1
    frame_w = col_runs[0][1] - col_runs[0][0] + 1
    assert frame_h == frame_w and frame_h >= 5
    assert all(r1 - r0 + 1 == frame_h for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == frame_w for c0, c1 in col_runs)

    frames: list[tuple[int, int, int, np.ndarray]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = {int(v) for v in block.flat if int(v) != BACKGROUND}
            assert len(colors) == 1
            color = colors.pop()
            assert np.all(block[0, :] == color)
            assert np.all(block[-1, :] == color)
            assert np.all(block[:, 0] == color)
            assert np.all(block[:, -1] == color)
            frames.append((r0, c0, color, block.copy()))

    counts = Counter(color for _r0, _c0, color, _block in frames)
    assert set(counts) == set(FRAME_COLORS)
    assert all(counts[color] == 3 for color in FRAME_COLORS)
    return frame_h, frame_w, frames


def _complete_by_color(grid: np.ndarray) -> np.ndarray:
    frame_h, frame_w, frames = _extract_frames(grid)
    masks_by_color: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((frame_h, frame_w), dtype=bool))

    for _r0, _c0, color, block in frames:
        masks_by_color[color] |= block == color

    out = np.array(grid, dtype=int, copy=True)
    for r0, c0, color, _block in frames:
        out[r0 : r0 + frame_h, c0 : c0 + frame_w][masks_by_color[color]] = color
    return out


def _column_pack(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    counts = {color: int(np.count_nonzero(g == color)) for color in FRAME_COLORS}
    assert all(counts[color] > 0 for color in FRAME_COLORS)
    assert len(set(counts.values())) == len(FRAME_COLORS)

    ranked_colors = sorted(FRAME_COLORS, key=lambda color: counts[color], reverse=True)
    out = np.zeros_like(g)
    height, width = g.shape
    scan = [(r, c) for c in range(width) for r in range(height - 1, -1, -1)]

    idx = 0
    for color in ranked_colors:
        for _ in range(counts[color]):
            r, c = scan[idx]
            out[r, c] = color
            idx += 1
    return out


def _ranked_colors(counts: dict[int, int]) -> tuple[int, ...]:
    return tuple(sorted(counts, key=lambda color: counts[color], reverse=True))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete same-colored frame motifs, then restack the colors by post-completion frequency.

    Input structure:
    - The grid is a 3x3 lattice of equal-sized colored square frames on a black background.
    - Neighboring frames are separated by one-pixel black gap rows and columns.
    - Each frame has a solid one-pixel border in one of three colors, and may contain same-colored interior motif pixels.
    - Each of the three frame colors appears exactly three times.

    Rule:
    - Group frames by border color.
    - Within each color group, take the union of all same-colored relative pixel positions across the three frames.
    - Paint that union back into every frame of the group to complete the shared motif.
    - Count the total number of pixels of each frame color after completion.
    - Produce a same-size output by packing those color counts in descending-frequency order, filling columns left-to-right and each column bottom-to-top; leave remaining cells black.

    Key insight:
    - The shared-motif completion changes the color totals, so the final column packing must be ranked after propagating each color's union motif to all matching frames.
    """
    completed = _complete_by_color(np.asarray(grid, dtype=int))
    return _column_pack(completed)


def _sample_union_sizes(rng: np.random.Generator, interior_area: int) -> tuple[int, int, int] | None:
    low = 3
    high = min(interior_area - 1, interior_area // 2 + 3)
    if high - low + 1 < 3:
        return None
    sizes = rng.choice(np.arange(low, high + 1), size=3, replace=False)
    return tuple(int(v) for v in sizes)


def _sample_input_totals(
    rng: np.random.Generator, union_sizes: tuple[int, int, int]
) -> tuple[int, int, int] | None:
    after_counts = {idx: union_sizes[idx] for idx in range(3)}
    after_rank = _ranked_colors(after_counts)

    for _ in range(200):
        totals = tuple(int(rng.integers(size, 3 * size)) for size in union_sizes)
        if len(set(totals)) != 3:
            continue
        before_counts = {idx: totals[idx] for idx in range(3)}
        before_rank = _ranked_colors(before_counts)
        if before_rank == after_rank:
            continue
        if before_rank[0] == after_rank[0]:
            continue
        return totals
    return None


def _build_subsets(
    rng: np.random.Generator, positions: list[tuple[int, int]], total_count: int
) -> list[set[tuple[int, int]]] | None:
    if len(positions) < 3:
        return None

    shuffled = positions[:]
    rng.shuffle(shuffled)
    frame_sets = [set([shuffled[0]]), set([shuffled[1]]), set([shuffled[2]])]
    for pos in shuffled[3:]:
        frame_sets[int(rng.integers(3))].add(pos)

    current = sum(len(frame_set) for frame_set in frame_sets)
    needed = total_count - current
    if needed < 0:
        return None

    candidates = [(pos, frame_idx) for pos in shuffled for frame_idx in range(3) if pos not in frame_sets[frame_idx]]
    rng.shuffle(candidates)
    for pos, frame_idx in candidates:
        if needed == 0:
            break
        frame_sets[frame_idx].add(pos)
        needed -= 1

    if needed != 0:
        return None
    if all(len(frame_set) == len(positions) for frame_set in frame_sets):
        return None
    return frame_sets


def _draw_frame(block: np.ndarray, color: int, motif: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(block, copy=True)
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    for r, c in motif:
        out[r, c] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_size = int(rng.integers(5, 9))
    interior_positions = [
        (r, c) for r in range(1, frame_size - 1) for c in range(1, frame_size - 1)
    ]
    union_sizes = _sample_union_sizes(rng, len(interior_positions))
    if union_sizes is None:
        return None

    input_totals = _sample_input_totals(rng, union_sizes)
    if input_totals is None:
        return None

    motifs_by_color: dict[int, list[set[tuple[int, int]]]] = {}
    union_masks: dict[int, set[tuple[int, int]]] = {}
    for color, union_size, input_total in zip(FRAME_COLORS, union_sizes, input_totals):
        chosen_idx = rng.choice(len(interior_positions), size=union_size, replace=False)
        union_positions = [interior_positions[int(i)] for i in chosen_idx]
        subsets = _build_subsets(rng, union_positions, input_total)
        if subsets is None:
            return None
        motifs_by_color[color] = subsets
        union_masks[color] = set(union_positions)

    layout = list(FRAME_COLORS) * 3
    rng.shuffle(layout)
    usage = Counter()
    grid_size = FRAME_GRID * frame_size + (FRAME_GRID - 1) * GAP
    grid = np.zeros((grid_size, grid_size), dtype=int)

    for idx, color in enumerate(layout):
        r0 = (idx // FRAME_GRID) * (frame_size + GAP)
        c0 = (idx % FRAME_GRID) * (frame_size + GAP)
        variant_idx = usage[color]
        usage[color] += 1
        block = _draw_frame(np.zeros((frame_size, frame_size), dtype=int), color, motifs_by_color[color][variant_idx])
        grid[r0 : r0 + frame_size, c0 : c0 + frame_size] = block

    if any(usage[color] != 3 for color in FRAME_COLORS):
        return None

    completed = _complete_by_color(grid)
    before_counts = {color: int(np.count_nonzero(grid == color)) for color in FRAME_COLORS}
    after_counts = {color: int(np.count_nonzero(completed == color)) for color in FRAME_COLORS}
    before_rank = _ranked_colors(before_counts)
    after_rank = _ranked_colors(after_counts)

    if len(set(after_counts.values())) != 3:
        return None
    if before_rank == after_rank:
        return None
    if before_rank[0] == after_rank[0]:
        return None

    border_pixels = 4 * frame_size - 4
    if any(after_counts[color] != 3 * (border_pixels + len(union_masks[color])) for color in FRAME_COLORS):
        return None
    if not np.any(completed != grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after retries")
