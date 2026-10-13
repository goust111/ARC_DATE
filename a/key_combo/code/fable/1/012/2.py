from __future__ import annotations

import math
import random

import numpy as np


PALETTE = tuple(range(10))
SCALE_PAIRS = ((1, 2), (1, 3), (2, 3), (1, 4), (3, 4))
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
MAX_RETRIES = 500


def _mode(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int).reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _find_divider(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows = [(r, int(grid[r, 0])) for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    full_cols = [(c, int(grid[0, c])) for c in range(w) if np.all(grid[:, c] == grid[0, c])]

    matches = []
    for r, row_color in full_rows:
        for c, col_color in full_cols:
            if row_color == col_color:
                matches.append((int(r), int(c), int(row_color)))

    if len(matches) != 1:
        raise ValueError("expected exactly one matching full-row/full-col divider")
    return matches[0]


def _canonicalize(quadrant: np.ndarray, row_side: int, col_side: int) -> np.ndarray:
    patch = quadrant
    if col_side == 1:
        patch = np.fliplr(patch)
    if row_side == 1:
        patch = np.flipud(patch)
    return patch


def _downsample_blocks(quadrant: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    qh, qw = quadrant.shape
    if qh % out_h != 0 or qw % out_w != 0:
        raise ValueError("quadrant size must be an integer multiple of the motif size")

    row_scale = qh // out_h
    col_scale = qw // out_w
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            block = quadrant[
                r * row_scale : (r + 1) * row_scale,
                c * col_scale : (c + 1) * col_scale,
            ]
            out[r, c] = _mode(block)
    return out


def _scale_patch(patch: np.ndarray, row_scale: int, col_scale: int) -> np.ndarray:
    return np.repeat(np.repeat(patch, row_scale, axis=0), col_scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a clean motif by unmirroring and consensus-denoising four differently scaled quadrant copies.

    Input structure:
    - A single full row and a single full column of one color form a cross divider.
    - The four quadrants contain the same motif, mirrored by quadrant position.
    - Top vs bottom quadrants use different integer row block scales, and left vs right quadrants use different integer column block scales.
    - Noise overwrites whole enlarged cells, and any motif cell is corrupted in at most one quadrant.

    Rule:
    - Locate the cross divider and split the grid into four quadrants.
    - Infer the base motif size from the gcd of the top/bottom heights and the gcd of the left/right widths.
    - Undo the quadrant-dependent mirror, shrink each quadrant back to the base motif by block voting, and align the four readings.
    - Take a per-cell majority vote across the four aligned readings and output the clean motif in top-left orientation.

    Key insight:
    - The cross position separately encodes the row and column scale factors, while the noise respects enlarged-cell boundaries, so each quadrant can be block-decoded and the four partial copies can repair one another by consensus.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    mid_r, mid_c, _divider_color = _find_divider(g)
    quadrants = {
        (0, 0): g[:mid_r, :mid_c],
        (0, 1): g[:mid_r, mid_c + 1 :],
        (1, 0): g[mid_r + 1 :, :mid_c],
        (1, 1): g[mid_r + 1 :, mid_c + 1 :],
    }

    top_h = quadrants[(0, 0)].shape[0]
    bottom_h = quadrants[(1, 0)].shape[0]
    left_w = quadrants[(0, 0)].shape[1]
    right_w = quadrants[(0, 1)].shape[1]

    motif_h = math.gcd(int(top_h), int(bottom_h))
    motif_w = math.gcd(int(left_w), int(right_w))
    if motif_h <= 0 or motif_w <= 0:
        raise ValueError("invalid motif size")

    readings = []
    for row_side, col_side in QUADRANTS:
        canonical = _canonicalize(quadrants[(row_side, col_side)], row_side, col_side)
        readings.append(_downsample_blocks(canonical, motif_h, motif_w))

    out = np.empty((motif_h, motif_w), dtype=int)
    for r in range(motif_h):
        for c in range(motif_w):
            out[r, c] = _mode(np.array([reading[r, c] for reading in readings], dtype=int))
    return out


def _make_motif(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    area = height * width
    max_colors = min(5, len(colors), area)
    if max_colors < 3:
        return None

    for _ in range(200):
        n_colors = random.randint(3, max_colors)
        motif_colors = random.sample(colors, n_colors)
        motif = np.array(random.choices(motif_colors, k=area), dtype=int).reshape(height, width)

        positions = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(positions)
        for idx, color in enumerate(motif_colors):
            r, c = positions[idx]
            motif[r, c] = color

        if len(set(map(int, motif.reshape(-1)))) < 3:
            continue
        if np.array_equal(motif, np.fliplr(motif)):
            continue
        if np.array_equal(motif, np.flipud(motif)):
            continue
        if np.array_equal(motif, np.flipud(np.fliplr(motif))):
            continue
        if all(len(set(map(int, motif[r, :]))) == 1 for r in range(height)):
            continue
        if all(len(set(map(int, motif[:, c]))) == 1 for c in range(width)):
            continue
        return motif

    return None


def _noise_plan(height: int, width: int) -> dict[tuple[int, int], set[tuple[int, int]]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)

    total_upper = min(len(cells), max(4, (len(cells) + 1) // 2))
    total_noisy = random.randint(4, total_upper)
    chosen = cells[:total_noisy]

    plan = {key: set() for key in QUADRANTS}
    for key, cell in zip(QUADRANTS, chosen[:4]):
        plan[key].add(cell)
    for cell in chosen[4:]:
        plan[random.choice(QUADRANTS)].add(cell)
    return plan


def _apply_noise(
    scaled: np.ndarray,
    motif: np.ndarray,
    row_scale: int,
    col_scale: int,
    noisy_cells: set[tuple[int, int]],
    divider_color: int,
) -> np.ndarray:
    out = scaled.copy()
    usable_colors = [color for color in PALETTE if color != divider_color]

    for r, c in noisy_cells:
        true_color = int(motif[r, c])
        options = [color for color in usable_colors if color != true_color]
        noisy_color = random.choice(options)
        out[
            r * row_scale : (r + 1) * row_scale,
            c * col_scale : (c + 1) * col_scale,
        ] = noisy_color
    return out


def _gen_attempt() -> np.ndarray | None:
    row_pair = list(random.choice(SCALE_PAIRS))
    col_pair = list(random.choice(SCALE_PAIRS))
    if random.random() < 0.5:
        row_pair.reverse()
    if random.random() < 0.5:
        col_pair.reverse()

    top_scale, bottom_scale = row_pair
    left_scale, right_scale = col_pair

    max_motif_h = min(6, 29 // (top_scale + bottom_scale))
    max_motif_w = min(6, 29 // (left_scale + right_scale))
    if max_motif_h < 2 or max_motif_w < 2:
        return None

    height = random.randint(2, max_motif_h)
    width = random.randint(2, max_motif_w)
    if height * width < 6:
        return None

    divider_color = random.choice(PALETTE)
    motif_colors = [color for color in PALETTE if color != divider_color]
    motif = _make_motif(height, width, motif_colors)
    if motif is None:
        return None

    row_scales = {0: top_scale, 1: bottom_scale}
    col_scales = {0: left_scale, 1: right_scale}
    noisy = _noise_plan(height, width)

    quadrants = {}
    for row_side, col_side in QUADRANTS:
        patch = motif
        if col_side == 1:
            patch = np.fliplr(patch)
        if row_side == 1:
            patch = np.flipud(patch)

        mapped_noisy = set()
        for r, c in noisy[(row_side, col_side)]:
            rr = height - 1 - r if row_side == 1 else r
            cc = width - 1 - c if col_side == 1 else c
            mapped_noisy.add((rr, cc))

        scaled = _scale_patch(patch, row_scales[row_side], col_scales[col_side])
        quadrants[(row_side, col_side)] = _apply_noise(
            scaled,
            patch,
            row_scales[row_side],
            col_scales[col_side],
            mapped_noisy,
            divider_color,
        )

    top_h = top_scale * height
    bottom_h = bottom_scale * height
    left_w = left_scale * width
    right_w = right_scale * width

    grid = np.empty((top_h + bottom_h + 1, left_w + right_w + 1), dtype=int)
    grid[:top_h, :left_w] = quadrants[(0, 0)]
    grid[:top_h, left_w] = divider_color
    grid[:top_h, left_w + 1 :] = quadrants[(0, 1)]
    grid[top_h, :] = divider_color
    grid[top_h + 1 :, :left_w] = quadrants[(1, 0)]
    grid[top_h + 1 :, left_w] = divider_color
    grid[top_h + 1 :, left_w + 1 :] = quadrants[(1, 1)]

    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if _find_divider(grid)[:2] != (top_h, left_w):
        return None
    if not np.array_equal(main(grid), motif):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
