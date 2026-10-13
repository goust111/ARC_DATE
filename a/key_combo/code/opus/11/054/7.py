from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
FRAME_SIZE = 5
INTERIOR_SIZE = 3
STRIDE = FRAME_SIZE + 1
COLORS = (1, 2, 3, 4)
GRID_CELLS = 3
MAX_GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    assert idx.size > 0, "expected at least one foreground run"
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for x in map(int, idx[1:]):
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs


def _parse_frames(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    row_runs = _runs(np.any(g != bg, axis=1))
    col_runs = _runs(np.any(g != bg, axis=0))
    frame_h = row_runs[0][1] - row_runs[0][0] + 1
    frame_w = col_runs[0][1] - col_runs[0][0] + 1

    assert frame_h == frame_w >= 3
    assert all(r1 - r0 + 1 == frame_h for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == frame_w for c0, c1 in col_runs)

    frames = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = sorted(set(map(int, block[block != bg])))
            assert len(colors) == 1, "each block must contain exactly one frame color"
            color = colors[0]
            assert np.all(block[0, :] == color)
            assert np.all(block[-1, :] == color)
            assert np.all(block[:, 0] == color)
            assert np.all(block[:, -1] == color)
            interior = block[1:-1, 1:-1]
            assert np.all((interior == bg) | (interior == color))
            frames.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "color": int(color),
                    "motif": (interior == color).copy(),
                }
            )

    return bg, frame_h, frame_w, frames


def main(grid: np.ndarray) -> np.ndarray:
    """Complete same-colored frame motifs, then summarize their filled areas as sorted vertical bars.

    Input structure:
    - The background is black.
    - Foreground objects form a rectangular lattice of equal-size colored square frames separated by black gaps.
    - Each frame has a one-pixel border in one color and a partially filled interior using only that same color.

    Rule:
    - Group frames by border color.
    - For each color group, take the union of all relative filled interior cells and treat that as the completed motif for that color.
    - Count the number of filled cells in that completed interior motif, sort colors by that count ascending, and output one column per color.
    - The output height is the largest completed count; each column is bottom-filled with its color for exactly that many cells.

    Key insight:
    - The quantity to rank is not any original frame's visible fill count, but the area of the shared motif after same-color completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, frame_h, frame_w, frames = _parse_frames(g)
    interior_h = frame_h - 2
    interior_w = frame_w - 2
    motifs_by_color: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((interior_h, interior_w), dtype=bool)
    )
    frame_counts: Counter[int] = Counter()

    for frame in frames:
        color = int(frame["color"])
        motifs_by_color[color] |= frame["motif"]
        frame_counts[color] += 1

    assert len(motifs_by_color) >= 2, "expected multiple color groups"
    assert all(2 <= count <= 3 for count in frame_counts.values())

    fill_counts = {color: int(mask.sum()) for color, mask in motifs_by_color.items()}
    counts = list(fill_counts.values())
    assert counts and len(counts) == len(set(counts)), "completed fill counts must be unique"

    ranked = sorted(fill_counts.items(), key=lambda item: item[1])
    out = np.zeros((max(counts), len(ranked)), dtype=int)
    for col, (color, count) in enumerate(ranked):
        if count:
            out[-count:, col] = int(color)
    return out


def _output_from_fill_counts(fill_counts: dict[int, int]) -> np.ndarray:
    ranked = sorted(fill_counts.items(), key=lambda item: item[1])
    out = np.zeros((max(fill_counts.values()), len(ranked)), dtype=int)
    for col, (color, count) in enumerate(ranked):
        out[-count:, col] = int(color)
    return out


def _sample_fill_counts(group_sizes: dict[int, int]) -> dict[int, int]:
    branch = random.randrange(3)
    if branch == 0:
        counts = [3, 9, *random.sample(range(4, 9), 2)]
    elif branch == 1:
        start = random.randint(3, 6)
        counts = list(range(start, start + 4))
    else:
        counts = random.sample(range(3, 10), 4)
    random.shuffle(counts)
    return {color: count for color, count in zip(COLORS, counts)}


def _sample_canonical_mask(fill_count: int) -> np.ndarray:
    mask = np.zeros((INTERIOR_SIZE, INTERIOR_SIZE), dtype=bool)
    for idx in random.sample(range(INTERIOR_SIZE * INTERIOR_SIZE), fill_count):
        mask[idx // INTERIOR_SIZE, idx % INTERIOR_SIZE] = True
    return mask


def _owner_subsets(group_size: int) -> list[tuple[int, ...]]:
    subsets = []
    for bits in range(1, 1 << group_size):
        if bits == (1 << group_size) - 1:
            continue
        subsets.append(tuple(i for i in range(group_size) if bits & (1 << i)))
    return subsets


def _distribute_motif(canonical: np.ndarray, group_size: int) -> list[np.ndarray]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(canonical)]
    random.shuffle(cells)
    masks = [np.zeros_like(canonical, dtype=bool) for _ in range(group_size)]

    if group_size == 2:
        assert len(cells) >= 3
        base = [(0,), (1,), (0, 1)]
    else:
        assert group_size == 3 and len(cells) >= 3
        if len(cells) == 3:
            base = [(0,), (1, 2), (0, 1)]
        else:
            base = [(0,), (1,), (2,), (0, 1)]

    for (r, c), owners in zip(cells, base):
        for idx in owners:
            masks[idx][r, c] = True

    subsets = _owner_subsets(group_size)
    for r, c in cells[len(base) :]:
        owners = random.choice(subsets)
        for idx in owners:
            masks[idx][r, c] = True

    sizes = [int(mask.sum()) for mask in masks]
    assert all(0 < size < len(cells) for size in sizes)
    assert len({mask.tobytes() for mask in masks}) == group_size
    multiplicities = [sum(int(mask[r, c]) for mask in masks) for r, c in cells]
    assert 1 in multiplicities and max(multiplicities) > 1
    return masks


def _paint_frame(grid: np.ndarray, r0: int, c0: int, color: int, motif: np.ndarray) -> None:
    r1 = r0 + FRAME_SIZE
    c1 = c0 + FRAME_SIZE
    grid[r0:r1, c0:c1] = BG
    grid[r0, c0:c1] = color
    grid[r1 - 1, c0:c1] = color
    grid[r0:r1, c0] = color
    grid[r0:r1, c1 - 1] = color
    grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1][motif] = color


def _sample_color_assignment(group_sizes: dict[int, int]) -> list[int] | None:
    colors = [color for color, size in group_sizes.items() for _ in range(size)]
    for _ in range(80):
        random.shuffle(colors)
        rows = [colors[i * GRID_CELLS : (i + 1) * GRID_CELLS] for i in range(GRID_CELLS)]
        cols = [[rows[r][c] for r in range(GRID_CELLS)] for c in range(GRID_CELLS)]
        if all(len(set(line)) >= 2 for line in rows + cols):
            return colors.copy()
    return None


def _dependency_ok(fill_counts: dict[int, int], group_masks: dict[int, list[np.ndarray]]) -> bool:
    union_order = tuple(color for color, _ in sorted(fill_counts.items(), key=lambda item: item[1]))
    raw_totals = {color: sum(int(mask.sum()) for mask in masks) for color, masks in group_masks.items()}
    raw_max = {color: max(int(mask.sum()) for mask in masks) for color, masks in group_masks.items()}

    def uniquely_matches(proxy: dict[int, int]) -> bool:
        if len(set(proxy.values())) != len(proxy):
            return False
        proxy_order = tuple(color for color, _ in sorted(proxy.items(), key=lambda item: item[1]))
        return proxy_order == union_order

    return not uniquely_matches(raw_totals) and not uniquely_matches(raw_max)


def _build_attempt() -> np.ndarray | None:
    triple_color = random.choice(COLORS)
    group_sizes = {color: (3 if color == triple_color else 2) for color in COLORS}
    fill_counts = _sample_fill_counts(group_sizes)

    canonical_by_color = {color: _sample_canonical_mask(fill_counts[color]) for color in COLORS}
    group_masks = {
        color: _distribute_motif(canonical_by_color[color], group_sizes[color]) for color in COLORS
    }
    if not _dependency_ok(fill_counts, group_masks):
        return None

    assignment = _sample_color_assignment(group_sizes)
    if assignment is None:
        return None

    pad_top = random.randint(0, 4)
    pad_bottom = random.randint(0, 4)
    pad_left = random.randint(0, 4)
    pad_right = random.randint(0, 4)
    height = GRID_CELLS * FRAME_SIZE + (GRID_CELLS - 1) + pad_top + pad_bottom
    width = GRID_CELLS * FRAME_SIZE + (GRID_CELLS - 1) + pad_left + pad_right
    grid = np.zeros((height, width), dtype=int)

    next_index = {color: 0 for color in COLORS}
    for idx, color in enumerate(assignment):
        rr, cc = divmod(idx, GRID_CELLS)
        r0 = pad_top + rr * STRIDE
        c0 = pad_left + cc * STRIDE
        motif = group_masks[color][next_index[color]]
        next_index[color] += 1
        _paint_frame(grid, r0, c0, color, motif)

    expected = _output_from_fill_counts(fill_counts)
    if not np.array_equal(main(grid), expected):
        return None

    try:
        _bg, _frame_h, _frame_w, frames = _parse_frames(grid)
    except Exception:
        return None
    if len(frames) != 9:
        return None

    seen_per_color: Counter[int] = Counter(frame["color"] for frame in frames)
    if dict(seen_per_color) != group_sizes:
        return None

    for color in COLORS:
        canonical = canonical_by_color[color]
        observed = [frame["motif"] for frame in frames if frame["color"] == color]
        if len(observed) != group_sizes[color]:
            return None
        if any(np.array_equal(mask, canonical) for mask in observed):
            return None
        if np.logical_or.reduce(observed).sum() != canonical.sum():
            return None
        if not np.array_equal(np.logical_or.reduce(observed), canonical):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
