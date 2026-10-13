import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _regular_intervals(separators: list[int], size: int) -> list[tuple[int, int]] | None:
    if not separators or separators[0] == 0 or separators[-1] == size - 1:
        return None
    intervals: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if sep <= start:
            return None
        intervals.append((start, sep))
        start = sep + 1
    if start >= size:
        return None
    intervals.append((start, size))
    lengths = {end - begin for begin, end in intervals}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        return None
    return intervals


def _find_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = grid.shape
    for color_value in np.unique(grid):
        color = int(color_value)
        sep_rows = np.where(np.all(grid == color, axis=1))[0].tolist()
        sep_cols = np.where(np.all(grid == color, axis=0))[0].tolist()
        row_intervals = _regular_intervals(sep_rows, height)
        col_intervals = _regular_intervals(sep_cols, width)
        if row_intervals is None or col_intervals is None:
            continue
        if {len(row_intervals), len(col_intervals)} != {3, 4}:
            continue
        return color, row_intervals, col_intervals
    raise AssertionError("failed to locate the separator lattice")


def _majority_block(blocks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    _, block_h, block_w = stack.shape
    consensus = np.empty((block_h, block_w), dtype=int)
    for r in range(block_h):
        for c in range(block_w):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(np.max(counts))
            winners = values[counts == best]
            assert len(winners) == 1
            consensus[r, c] = int(winners[0])
    return consensus


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _sep_color, row_intervals, col_intervals = _find_layout(g)
    block_rows = len(row_intervals)
    block_cols = len(col_intervals)
    assert {block_rows, block_cols} == {3, 4}

    blocks: dict[tuple[int, int], np.ndarray] = {}
    block_color_occurrences: Counter[int] = Counter()
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            block = g[r0:r1, c0:c1]
            blocks[(br, bc)] = block
            for color_value in np.unique(block):
                block_color_occurrences[int(color_value)] += 1

    legend_positions: list[tuple[int, int]] = []
    for pos, block in blocks.items():
        values = np.unique(block)
        if len(values) == 1 and block_color_occurrences[int(values[0])] == 1:
            legend_positions.append(pos)
    assert len(legend_positions) == 2
    legend_set = set(legend_positions)

    consensus = _majority_block([block for pos, block in blocks.items() if pos not in legend_set])

    damaged_positions: list[tuple[int, int]] = []
    diff_masks: dict[tuple[int, int], np.ndarray] = {}
    for pos, block in blocks.items():
        if pos in legend_set:
            continue
        diff = block != consensus
        if np.any(diff):
            damaged_positions.append(pos)
            diff_masks[pos] = diff
    assert len(damaged_positions) == 2

    legend_by_row: dict[int, int] = {}
    for br, bc in legend_positions:
        color = int(blocks[(br, bc)][0, 0])
        assert br not in legend_by_row
        legend_by_row[br] = color

    damaged_by_row: dict[int, tuple[int, int]] = {}
    for pos in damaged_positions:
        br, _bc = pos
        assert br not in damaged_by_row
        damaged_by_row[br] = pos

    assert set(legend_by_row) == set(damaged_by_row)

    out = g.copy()
    for br, bc in legend_positions:
        r0, r1 = row_intervals[br]
        c0, c1 = col_intervals[bc]
        out[r0:r1, c0:c1] = consensus

    for br, pos in damaged_by_row.items():
        bc = pos[1]
        r0, r1 = row_intervals[br]
        c0, c1 = col_intervals[bc]
        repaired = consensus.copy()
        repaired[diff_masks[pos]] = legend_by_row[br]
        out[r0:r1, c0:c1] = repaired

    return out


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _random_parts(total: int, count: int, minimum: int) -> list[int]:
    parts = [minimum] * count
    remaining = total - count * minimum
    while remaining > 0:
        idx = random.randrange(count)
        parts[idx] += 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def _grow_blob(height: int, width: int, blocked: np.ndarray, size: int) -> set[tuple[int, int]] | None:
    free = [(r, c) for r in range(height) for c in range(width) if not blocked[r, c]]
    if len(free) < size:
        return None

    seed = random.choice(free)
    blob = {seed}
    frontier = {cell for cell in _neighbors4(seed[0], seed[1], height, width) if not blocked[cell]}

    while len(blob) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if not blocked[nb] and nb not in blob:
                frontier.add(nb)
    return blob


def _make_motif(block_h: int, block_w: int, base: int, accents: list[int]) -> np.ndarray | None:
    area = block_h * block_w
    min_base = max(3, area // 4)
    min_total = 2 * len(accents)
    max_total = area - min_base
    if min_total > max_total:
        return None

    for _ in range(80):
        accent_total = random.randint(min_total, max_total)
        sizes = _random_parts(accent_total, len(accents), 2)
        motif = np.full((block_h, block_w), base, dtype=int)
        blocked = np.zeros((block_h, block_w), dtype=bool)
        ok = True

        for color, size in zip(accents, sizes):
            blob = _grow_blob(block_h, block_w, blocked, size)
            if blob is None:
                ok = False
                break
            for r, c in blob:
                motif[r, c] = color
                blocked[r, c] = True

        if not ok:
            continue
        if len(np.unique(motif)) != 1 + len(accents):
            continue
        non_base = np.argwhere(motif != base)
        if len({int(r) for r, _c in non_base}) < 2 or len({int(c) for _r, c in non_base}) < 2:
            continue
        return motif
    return None


def _damage_mask(motif: np.ndarray, base: int) -> np.ndarray | None:
    non_base_cells = [tuple(map(int, rc)) for rc in np.argwhere(motif != base)]
    if not non_base_cells:
        return None
    upper = max(1, (len(non_base_cells) * 3) // 5)
    count = random.randint(1, upper)
    chosen = random.sample(non_base_cells, count)
    mask = np.zeros_like(motif, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _block_origin(br: int, bc: int, block_h: int, block_w: int) -> tuple[int, int]:
    return br * (block_h + 1), bc * (block_w + 1)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    block_rows, block_cols = random.choice([(3, 4), (4, 3)])
    block_h = random.randint(4, 7)
    block_w = random.randint(4, 7)

    accent_count = random.choice([1, 2])
    colors = random.sample(range(10), 4 + accent_count)
    sep = colors[0]
    base = colors[1]
    accents = colors[2 : 2 + accent_count]
    legend_colors = colors[2 + accent_count :]

    motif = _make_motif(block_h, block_w, base, accents)
    if motif is None:
        return None

    paired_rows = random.sample(range(block_rows), 2)
    legend_positions: dict[int, tuple[int, int]] = {}
    damaged_positions: dict[int, tuple[int, int]] = {}
    for br in paired_rows:
        legend_col, damaged_col = random.sample(range(block_cols), 2)
        legend_positions[br] = (br, legend_col)
        damaged_positions[br] = (br, damaged_col)

    damage_masks = {}
    for br in paired_rows:
        mask = _damage_mask(motif, base)
        if mask is None or not np.any(mask):
            return None
        damage_masks[br] = mask

    height = block_rows * block_h + (block_rows - 1)
    width = block_cols * block_w + (block_cols - 1)
    inp = np.full((height, width), sep, dtype=int)
    out = np.full((height, width), sep, dtype=int)

    for br in range(block_rows):
        for bc in range(block_cols):
            r0, c0 = _block_origin(br, bc, block_h, block_w)
            if br in legend_positions and legend_positions[br] == (br, bc):
                legend_color = legend_colors[paired_rows.index(br)]
                inp[r0 : r0 + block_h, c0 : c0 + block_w] = legend_color
                out[r0 : r0 + block_h, c0 : c0 + block_w] = motif
            elif br in damaged_positions and damaged_positions[br] == (br, bc):
                mask = damage_masks[br]
                damaged = motif.copy()
                damaged[mask] = base
                repaired = motif.copy()
                repaired[mask] = legend_colors[paired_rows.index(br)]
                inp[r0 : r0 + block_h, c0 : c0 + block_w] = damaged
                out[r0 : r0 + block_h, c0 : c0 + block_w] = repaired
            else:
                inp[r0 : r0 + block_h, c0 : c0 + block_w] = motif
                out[r0 : r0 + block_h, c0 : c0 + block_w] = motif

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _gen_attempt()
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out) and not np.array_equal(inp, out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
