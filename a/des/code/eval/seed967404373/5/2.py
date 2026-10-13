import numpy as np


BG = 0
SEP = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BLOCK_SIZE_RANGE = (5, 9)
BLOCK_ROW_RANGE = (2, 5)
BLOCK_COL_RANGE = (2, 5)
STAMP_FILL_RATIO = (0.28, 0.52)
NOISE_POSITION_PROB = (0.18, 0.34)
GEN_ATTEMPTS = 400


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for value in indices[1:]:
        if value != prev + 1:
            runs.append((start, prev + 1))
            start = value
        prev = value
    runs.append((start, prev + 1))
    return runs


def _non_separator_runs(length: int, separators: list[int]) -> list[tuple[int, int]]:
    sep_set = set(separators)
    payload = [idx for idx in range(length) if idx not in sep_set]
    return _contiguous_runs(payload)


def _detect_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best = None
    for color in map(int, np.unique(g)):
        row_seps = [r for r in range(h) if np.all(g[r, :] == color)]
        col_seps = [c for c in range(w) if np.all(g[:, c] == color)]
        if not row_seps or not col_seps:
            continue

        row_runs = _non_separator_runs(h, row_seps)
        col_runs = _non_separator_runs(w, col_seps)
        if not row_runs or not col_runs:
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_runs}
        col_sizes = {c1 - c0 for c0, c1 in col_runs}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if row_sizes != col_sizes:
            continue
        if len(row_runs) * len(col_runs) < 4:
            continue

        score = (len(row_runs) * len(col_runs), len(row_seps) + len(col_seps))
        candidate = (score, color, row_runs, col_runs)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "failed to detect separator layout"
    _score, color, row_runs, col_runs = best
    return color, row_runs, col_runs


def _corner_views(arr: np.ndarray, size: int = 2) -> list[np.ndarray]:
    return [
        arr[:size, :size],
        arr[:size, -size:],
        arr[-size:, :size],
        arr[-size:, -size:],
    ]


def _rotation_to_reference(block: np.ndarray, reference: np.ndarray) -> int:
    best_k = 0
    best_score = None
    ref_corners = _corner_views(reference)
    for k in range(4):
        aligned = np.rot90(block, k)
        corner_score = sum(
            int(np.sum(a == b)) for a, b in zip(_corner_views(aligned), ref_corners)
        )
        full_score = int(np.sum(aligned == reference))
        score = (corner_score, full_score)
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
    return best_k


def _majority_vote(blocks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    _, h, w = stack.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(counts.max())
            winners = values[counts == best]
            if len(winners) == 1:
                out[r, c] = int(winners[0])
            else:
                ref_color = int(stack[0, r, c])
                out[r, c] = ref_color if ref_color in winners else int(winners[0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a noisy rotated stamp by aligning block copies, voting, and writing back clean rotations.
    """
    g = np.asarray(grid, dtype=int)
    _sep_color, row_runs, col_runs = _detect_layout(g)

    blocks: list[np.ndarray] = []
    slices: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            blocks.append(g[r0:r1, c0:c1])
            slices.append((r0, r1, c0, c1))

    reference = blocks[0]
    rotations = [_rotation_to_reference(block, reference) for block in blocks]
    aligned = [np.rot90(block, k) for block, k in zip(blocks, rotations)]
    clean = _majority_vote(aligned)

    out = g.copy()
    for (r0, r1, c0, c1), k in zip(slices, rotations):
        out[r0:r1, c0:c1] = np.rot90(clean, -k)
    return out


def _rotate_coord(r: int, c: int, size: int, k: int) -> tuple[int, int]:
    if k % 4 == 0:
        return r, c
    if k % 4 == 1:
        return size - 1 - c, r
    if k % 4 == 2:
        return size - 1 - r, size - 1 - c
    return c, size - 1 - r


def _grow_mask(size: int, target_area: int, forbidden: set[tuple[int, int]]) -> set[tuple[int, int]]:
    free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in forbidden]
    seed = free[int(np.random.randint(len(free)))]
    mask = {seed}
    while len(mask) < target_area:
        frontier = set()
        for r, c in mask:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in forbidden and (nr, nc) not in mask:
                    frontier.add((nr, nc))
        if frontier and np.random.rand() < 0.88:
            frontier_list = sorted(frontier)
            mask.add(frontier_list[int(np.random.randint(len(frontier_list)))])
        else:
            remaining = [pt for pt in free if pt not in mask]
            if not remaining:
                break
            mask.add(remaining[int(np.random.randint(len(remaining)))])
    return mask


def _make_stamp(size: int) -> tuple[np.ndarray, set[tuple[int, int]], list[int]]:
    for _ in range(200):
        color_count = 2 + int(np.random.randint(2))
        colors = [int(x) for x in np.random.choice(PALETTE, size=color_count, replace=False)]
        marker = colors[0]
        primary = colors[1]
        secondary = colors[2] if color_count == 3 else primary

        stamp = np.full((size, size), BG, dtype=int)
        protected = {(0, 0), (0, 1), (1, 0), (1, 1)}

        target_area = int(round(size * size * np.random.uniform(*STAMP_FILL_RATIO)))
        target_area = max(target_area, 7)
        mask = _grow_mask(size, target_area, protected)
        for r, c in mask:
            stamp[r, c] = primary

        if color_count == 3:
            accent_count = int(np.random.randint(3, max(4, len(mask) // 3 + 1)))
            mask_list = sorted(mask)
            picks = np.random.choice(len(mask_list), size=min(accent_count, len(mask_list)), replace=False)
            for idx in picks:
                r, c = mask_list[int(idx)]
                stamp[r, c] = secondary

        for _ in range(int(np.random.randint(2, 5))):
            if np.random.rand() < 0.5:
                r = int(np.random.randint(size))
                c0 = int(np.random.randint(size))
                c1 = int(np.random.randint(c0 + 1, size + 1))
                stamp[r, c0:c1] = primary if np.random.rand() < 0.7 else secondary
            else:
                c = int(np.random.randint(size))
                r0 = int(np.random.randint(size))
                r1 = int(np.random.randint(r0 + 1, size + 1))
                stamp[r0:r1, c] = primary if np.random.rand() < 0.7 else secondary

        if np.random.rand() < 0.6:
            h = int(np.random.randint(1, min(3, size) + 1))
            w = int(np.random.randint(1, min(3, size) + 1))
            r0 = int(np.random.randint(0, size - h + 1))
            c0 = int(np.random.randint(0, size - w + 1))
            stamp[r0 : r0 + h, c0 : c0 + w] = secondary

        # Reapply the asymmetric corner marker so it is never overwritten.
        stamp[0, 0] = marker
        stamp[0, 1] = primary
        stamp[1, 0] = primary
        stamp[1, 1] = secondary

        if int((stamp != BG).sum()) < max(8, size + 2):
            continue
        if not np.any(stamp[2:, 2:] != BG):
            continue
        if len(set(map(int, np.unique(stamp))) - {BG}) != color_count:
            continue
        if any(np.array_equal(stamp, np.rot90(stamp, k)) for k in (1, 2, 3)):
            continue

        marker_count = int(np.sum(stamp == marker))
        if marker_count != 1:
            continue
        return stamp, protected, colors

    raise RuntimeError("failed to generate a valid stamp")


def _balanced_rotations(total_blocks: int) -> list[int]:
    rotations = [idx % 4 for idx in range(total_blocks)]
    np.random.shuffle(rotations)
    return [int(x) for x in rotations]


def _compose_grid(
    blocks: list[np.ndarray], block_rows: int, block_cols: int, block_size: int
) -> np.ndarray:
    h = block_rows * block_size + block_rows + 1
    w = block_cols * block_size + block_cols + 1
    grid = np.full((h, w), SEP, dtype=int)
    idx = 0
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = 1 + br * (block_size + 1)
            c0 = 1 + bc * (block_size + 1)
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = blocks[idx]
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    block_size = int(np.random.randint(*BLOCK_SIZE_RANGE))
    block_rows = int(np.random.randint(*BLOCK_ROW_RANGE))
    block_cols = int(np.random.randint(*BLOCK_COL_RANGE))
    total_blocks = block_rows * block_cols

    stamp, protected, colors = _make_stamp(block_size)
    rotations = _balanced_rotations(total_blocks)
    clean_blocks = [np.rot90(stamp, k) for k in rotations]
    noisy_blocks = [block.copy() for block in clean_blocks]

    non_marker_colors = [BG] + [color for color in colors[1:]]
    corrupt_cap = max(1, (total_blocks - 1) // 2)
    corrupt_prob = float(np.random.uniform(*NOISE_POSITION_PROB))
    changed = 0

    for r in range(block_size):
        for c in range(block_size):
            if (r, c) in protected:
                continue
            if np.random.rand() >= corrupt_prob:
                continue

            corrupted_blocks = int(np.random.randint(1, corrupt_cap + 1))
            picked = np.random.choice(total_blocks, size=corrupted_blocks, replace=False)
            true_color = int(stamp[r, c])
            wrong_choices = [color for color in non_marker_colors if color != true_color]
            if not wrong_choices:
                continue

            for block_idx in picked:
                rr, cc = _rotate_coord(r, c, block_size, rotations[int(block_idx)])
                new_color = int(wrong_choices[int(np.random.randint(len(wrong_choices)))])
                if noisy_blocks[int(block_idx)][rr, cc] != new_color:
                    noisy_blocks[int(block_idx)][rr, cc] = new_color
                    changed += 1

    if changed == 0:
        r, c = 2, 2
        true_color = int(stamp[r, c])
        wrong_choices = [color for color in non_marker_colors if color != true_color]
        if wrong_choices:
            rr, cc = _rotate_coord(r, c, block_size, rotations[0])
            noisy_blocks[0][rr, cc] = int(wrong_choices[0])
            changed = 1

    noise_counts = [int(np.sum(noisy != clean)) for noisy, clean in zip(noisy_blocks, clean_blocks)]
    if sum(count > 0 for count in noise_counts) < max(2, total_blocks // 3):
        return None

    input_grid = _compose_grid(noisy_blocks, block_rows, block_cols, block_size)
    expected = _compose_grid(clean_blocks, block_rows, block_cols, block_size)

    if np.array_equal(input_grid, expected):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
