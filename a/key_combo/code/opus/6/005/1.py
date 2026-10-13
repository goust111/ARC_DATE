from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
BLOCK_SIZES = [3, 5]
GEN_ATTEMPTS = 500


def _build_intervals(total: int, separators: list[int], start: int = 0) -> list[tuple[int, int]] | None:
    prev = int(start)
    out: list[tuple[int, int]] = []
    for sep in separators:
        sep = int(sep)
        if sep <= prev or sep >= total:
            return None
        out.append((prev, sep))
        prev = sep + 1
    if prev >= total:
        return None
    out.append((prev, total))
    return out


def _majority_color(block: np.ndarray) -> int | None:
    values = [int(v) for v in block.ravel() if int(v) != 0]
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _analyze_candidate(grid: np.ndarray, marker_row: int) -> dict | None:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    marker_row = int(marker_row)
    if marker_row < 0 or marker_row >= H - 1:
        return None
    if np.any(g[:marker_row] != 0):
        return None

    marker_cols = [int(c) for c in np.flatnonzero(g[marker_row] != 0)]
    if not (1 <= len(marker_cols) <= 3):
        return None
    if len(set(marker_cols)) != len(marker_cols):
        return None

    body = g[marker_row + 1 :, :]
    if body.size == 0:
        return None

    row_separators = [int(r) for r in range(body.shape[0]) if np.all(body[r] == 0)]
    col_separators = [int(c) for c in range(W) if np.all(body[:, c] == 0)]
    row_intervals_local = _build_intervals(body.shape[0], row_separators, 0)
    col_intervals = _build_intervals(W, col_separators, 0)
    if row_intervals_local is None or col_intervals is None:
        return None
    if len(row_intervals_local) < 2 or len(col_intervals) < 2:
        return None

    row_heights = {r1 - r0 for r0, r1 in row_intervals_local}
    col_widths = {c1 - c0 for c0, c1 in col_intervals}
    if len(row_heights) != 1 or len(col_widths) != 1:
        return None

    block_h = next(iter(row_heights))
    block_w = next(iter(col_widths))
    if block_h < 2 or block_w < 3 or block_w % 2 == 0:
        return None

    centers = [int(c0 + block_w // 2) for c0, c1 in col_intervals if c1 - c0 == block_w]
    center_to_col = {center: idx for idx, center in enumerate(centers)}
    if any(center not in center_to_col for center in marker_cols):
        return None

    row_intervals = [(int(marker_row + 1 + r0), int(marker_row + 1 + r1)) for r0, r1 in row_intervals_local]
    dominants = np.zeros((len(row_intervals), len(col_intervals)), dtype=int)
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            block = g[r0:r1, c0:c1]
            if np.any(block == 0):
                return None
            dominant = _majority_color(block)
            if dominant is None:
                return None
            dominants[br, bc] = int(dominant)

    markers = sorted((center_to_col[c], int(g[marker_row, c])) for c in marker_cols)
    return {
        "marker_row": marker_row,
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "dominants": dominants,
        "markers": markers,
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    candidates = []
    for marker_row in range(grid.shape[0] - 1):
        info = _analyze_candidate(grid, marker_row)
        if info is not None:
            candidates.append(info)
    assert len(candidates) == 1, "expected exactly one valid marker row and lattice layout"
    return candidates[0]


def _apply_rules(grid: np.ndarray, info: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    marker_row = int(info["marker_row"])
    out[marker_row, :] = 0

    dominants = np.asarray(info["dominants"], dtype=int)
    row_intervals = info["row_intervals"]
    col_intervals = info["col_intervals"]

    for col_idx, marker_color in info["markers"]:
        stop_row = None
        for row_idx in range(dominants.shape[0]):
            if int(dominants[row_idx, col_idx]) == int(marker_color):
                continue
            stop_row = row_idx
            break
        assert stop_row is not None, "every marker must eventually hit a non-matching block"
        r0, r1 = row_intervals[stop_row]
        c0, c1 = col_intervals[col_idx]
        out[r0:r1, c0:c1] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top marker down its lattice column until it reaches the first block whose denoised dominant color does not match.

    Input structure:
    - Black (0) background with a rectangular lattice of equal-size colored blocks separated by 1-pixel black lines.
    - Each block interior is filled mostly by one foreground color, possibly with a few non-black noise pixels of other colors.
    - One top background row above the lattice contains 1-3 singleton colored markers, each centered over one lattice column.

    Rule:
    - For every block, compute its dominant non-black color.
    - For each marker, scan blocks downward in its aligned column.
    - Matching dominant color means the marker passes through that block; the first non-matching block stops it.
    - Recolor that stopping block's entire interior to the marker color, erase the original top marker pixels, and leave all separator lines and untouched blocks unchanged.

    Key insight:
    - The lattice must be compressed into one dominant color per block before the gravity rule can be evaluated, because noise pixels do not block or guide the markers.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    return _apply_rules(np.asarray(grid, dtype=int), info)


def _make_noisy_block(
    dominant: int,
    block_size: int,
    palette: list[int],
    *,
    min_noise: int = 0,
    forced_noise_color: int | None = None,
) -> np.ndarray:
    area = block_size * block_size
    max_noise = min((area - 1) // 2, int(area * 0.30))
    if max_noise < min_noise:
        max_noise = min_noise
    noise_count = random.randint(min_noise, max_noise) if max_noise > 0 else 0

    block = np.full((block_size, block_size), int(dominant), dtype=int)
    if noise_count == 0:
        return block

    positions = random.sample(range(area), noise_count)
    alt_colors = [int(c) for c in palette if int(c) != int(dominant)]
    noise_colors: list[int] = []
    if forced_noise_color is not None and int(forced_noise_color) != int(dominant):
        noise_colors.append(int(forced_noise_color))
    while len(noise_colors) < noise_count:
        noise_colors.append(int(random.choice(alt_colors)))
    random.shuffle(noise_colors)

    for pos, color in zip(positions, noise_colors):
        block[pos // block_size, pos % block_size] = int(color)

    assert _majority_color(block) == int(dominant)
    return block


def _choose_marker_specs(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    marker_count = random.randint(1, min(3, n_cols))
    cols = sorted(random.sample(range(n_cols), marker_count))
    if marker_count == 1:
        depths = [random.randint(0, n_rows - 1)]
    else:
        depths = [0, random.randint(1, n_rows - 1)]
        while len(depths) < marker_count:
            depths.append(random.randint(0, n_rows - 1))
        random.shuffle(depths)
    return list(zip(cols, depths))


def _column_center(block_size: int, col_idx: int) -> int:
    c0 = col_idx * (block_size + 1)
    return int(c0 + block_size // 2)


def _render_grid(
    dominants: np.ndarray,
    block_size: int,
    palette: list[int],
    marker_specs: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = dominants.shape
    H = 1 + n_rows * block_size + (n_rows - 1)
    W = n_cols * block_size + (n_cols - 1)
    grid = np.zeros((H, W), dtype=int)

    marker_by_col = {int(col): int(color) for col, color, _depth in marker_specs}

    for br in range(n_rows):
        for bc in range(n_cols):
            dominant = int(dominants[br, bc])
            min_noise = 0
            forced_noise_color = None
            if bc in marker_by_col:
                marker_color = int(marker_by_col[bc])
                stop_depth = next(depth for col, _color, depth in marker_specs if col == bc)
                if br == stop_depth:
                    min_noise = 1
                    forced_noise_color = marker_color
                elif br < stop_depth:
                    min_noise = 1 if random.random() < 0.7 else 0
            elif random.random() < 0.55:
                min_noise = 1

            block = _make_noisy_block(
                dominant,
                block_size,
                palette,
                min_noise=min_noise,
                forced_noise_color=forced_noise_color,
            )
            r0 = 1 + br * (block_size + 1)
            c0 = bc * (block_size + 1)
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = block

    for col, color, _depth in marker_specs:
        grid[0, _column_center(block_size, col)] = int(color)

    output = main(grid)
    return grid, output


def _semantic_ok(
    grid: np.ndarray,
    planned_dominants: np.ndarray,
    marker_specs: list[tuple[int, int, int]],
    expected_output: np.ndarray,
) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(info["dominants"], planned_dominants):
        return False

    inferred_markers = {(int(col), int(color)) for col, color in info["markers"]}
    planned_markers = {(int(col), int(color)) for col, color, _depth in marker_specs}
    if inferred_markers != planned_markers:
        return False

    seen_stops: set[tuple[int, int]] = set()
    for col, color, stop_depth in marker_specs:
        col = int(col)
        color = int(color)
        stop_depth = int(stop_depth)
        for row_idx in range(stop_depth):
            if int(planned_dominants[row_idx, col]) != color:
                return False
        if int(planned_dominants[stop_depth, col]) == color:
            return False
        stop_cell = (stop_depth, col)
        if stop_cell in seen_stops:
            return False
        seen_stops.add(stop_cell)

        r0, r1 = info["row_intervals"][stop_depth]
        c0, c1 = info["col_intervals"][col]
        block = grid[r0:r1, c0:c1]
        if not np.any(block == color):
            return False

    if not np.array_equal(main(grid), expected_output):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    block_size = random.choice(BLOCK_SIZES)
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)
    palette = random.sample(PALETTE, random.randint(4, min(6, len(PALETTE))))

    dominants = np.array(
        [[random.choice(palette) for _ in range(n_cols)] for _ in range(n_rows)],
        dtype=int,
    )

    marker_specs: list[tuple[int, int, int]] = []
    used_marker_colors: list[int] = []
    for col, stop_depth in _choose_marker_specs(n_rows, n_cols):
        marker_color = int(random.choice(palette))
        used_marker_colors.append(marker_color)
        for row_idx in range(stop_depth):
            dominants[row_idx, col] = marker_color
        dominants[stop_depth, col] = random.choice([c for c in palette if c != marker_color])
        for row_idx in range(stop_depth + 1, n_rows):
            dominants[row_idx, col] = random.choice(palette)
        marker_specs.append((int(col), marker_color, int(stop_depth)))

    for col in range(n_cols):
        if len(set(int(v) for v in dominants[:, col])) == 1:
            row_idx = random.randint(0, n_rows - 1)
            new_color = random.choice([c for c in palette if c != int(dominants[row_idx, col])])
            dominants[row_idx, col] = int(new_color)

    for color in used_marker_colors:
        if not np.any(dominants == int(color)):
            r = random.randint(0, n_rows - 1)
            c = random.randint(0, n_cols - 1)
            dominants[r, c] = int(color)

    if len(set(int(v) for v in dominants.ravel())) < 3:
        extra_color = random.choice([c for c in palette if c not in set(int(v) for v in dominants.ravel())])
        dominants[random.randint(0, n_rows - 1), random.randint(0, n_cols - 1)] = int(extra_color)

    grid, expected_output = _render_grid(dominants, block_size, palette, marker_specs)
    if max(grid.shape) > 30 or max(expected_output.shape) > 30:
        return None
    if not _semantic_ok(grid, dominants, marker_specs, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-gravity lattice input")
