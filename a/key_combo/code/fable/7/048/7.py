import math
from typing import Literal

import numpy as np


BLACK = 0
PALETTE = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(line: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(np.asarray(line, dtype=int), return_counts=True)
    idx = int(np.argmax(counts))
    return int(values[idx]), int(counts[idx])


def _axis_purity_and_dominants(grid: np.ndarray, axis: int) -> tuple[float, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    line_count = g.shape[axis]
    line_len = g.shape[1 - axis]
    dominants = np.empty(line_count, dtype=int)
    purity_sum = 0.0

    for i in range(line_count):
        line = g[i, :] if axis == 0 else g[:, i]
        dominant, count = _dominant_color(line)
        dominants[i] = dominant
        purity_sum += count / line_len

    return purity_sum / line_count, dominants


def _signal_mask(grid: np.ndarray, orientation: Literal["row", "col"]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if orientation == "row":
        _, dominants = _axis_purity_and_dominants(g, axis=0)
        return g != dominants[:, None]
    _, dominants = _axis_purity_and_dominants(g, axis=1)
    return g != dominants[None, :]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected at least one signal pixel")
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the stripe-violating pixels and crop them to their tight bounding box.

    Input structure:
    - The whole grid is a striped fabric made of either horizontal or vertical single-color lines.
    - Each stripe line has one dominant cloth color, but a small number of pixels deviate from that line color.
    - Some deviant pixels use colors absent from the cloth, while others disguise themselves as colors from other stripes.

    Rule:
    - Measure how close rows and columns are to being single-color, and choose the clearly purer axis as the stripe direction.
    - On that axis, mark every pixel whose color differs from its own line's dominant color as signal.
    - Output the bounding box of all signal pixels, preserving their original colors and positions inside that box and painting every other output cell black.

    Key insight:
    - The cloth is the noise to discard; the true payload is every per-line color deviation, even when that deviation imitates another stripe color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    row_score, _ = _axis_purity_and_dominants(g, axis=0)
    col_score, _ = _axis_purity_and_dominants(g, axis=1)
    orientation: Literal["row", "col"] = "row" if row_score >= col_score else "col"

    signal = _signal_mask(g, orientation)
    r0, c0, r1, c1 = _bbox(signal)

    cropped_signal = signal[r0 : r1 + 1, c0 : c1 + 1]
    cropped_grid = g[r0 : r1 + 1, c0 : c1 + 1]
    out = np.zeros(cropped_grid.shape, dtype=int)
    out[cropped_signal] = cropped_grid[cropped_signal]
    return out


def _choose_line_colors(num_lines: int, rng: np.random.Generator) -> list[int]:
    num_base_colors = int(rng.integers(4, 8))
    base_colors = rng.permutation(PALETTE)[:num_base_colors].tolist()
    step_choices = [step for step in range(1, num_base_colors) if math.gcd(step, num_base_colors) == 1]
    step = int(rng.choice(step_choices)) if step_choices else 1
    offset = int(rng.integers(num_base_colors))
    return [int(base_colors[(offset + i * step) % num_base_colors]) for i in range(num_lines)]


def _make_striped_grid(
    height: int,
    width: int,
    orientation: Literal["row", "col"],
    line_colors: list[int],
) -> np.ndarray:
    if orientation == "row":
        return np.repeat(np.asarray(line_colors, dtype=int)[:, None], width, axis=1)
    return np.repeat(np.asarray(line_colors, dtype=int)[None, :], height, axis=0)


def _sample_rowwise_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cap = 2 if width <= 7 else 3
    counts = np.zeros(height, dtype=int)

    for r in range(height):
        c = int(rng.integers(width))
        mask[r, c] = True
        counts[r] += 1

    for c in (0, width - 1):
        if not np.any(mask[:, c]):
            candidates = np.where((counts < cap) & (~mask[:, c]))[0]
            r = int(rng.choice(candidates))
            mask[r, c] = True
            counts[r] += 1

    extra = int(rng.integers(height // 2, height + 1))
    for _ in range(extra):
        candidates = np.where(counts < cap)[0]
        if len(candidates) == 0:
            break
        r = int(rng.choice(candidates))
        free_cols = np.where(~mask[r])[0]
        if len(free_cols) == 0:
            counts[r] = cap
            continue
        c = int(rng.choice(free_cols))
        mask[r, c] = True
        counts[r] += 1

    return mask


def _sample_colwise_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cap = 2 if height <= 7 else 3
    counts = np.zeros(width, dtype=int)

    for c in range(width):
        r = int(rng.integers(height))
        mask[r, c] = True
        counts[c] += 1

    for r in (0, height - 1):
        if not np.any(mask[r, :]):
            candidates = np.where((counts < cap) & (~mask[r, :]))[0]
            c = int(rng.choice(candidates))
            mask[r, c] = True
            counts[c] += 1

    extra = int(rng.integers(width // 2, width + 1))
    for _ in range(extra):
        candidates = np.where(counts < cap)[0]
        if len(candidates) == 0:
            break
        c = int(rng.choice(candidates))
        free_rows = np.where(~mask[:, c])[0]
        if len(free_rows) == 0:
            counts[c] = cap
            continue
        r = int(rng.choice(free_rows))
        mask[r, c] = True
        counts[c] += 1

    return mask


def _paint_signals(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    orientation: Literal["row", "col"],
    line_colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    stripe_palette = sorted(set(line_colors))
    exotic_colors = [color for color in PALETTE.tolist() if color not in stripe_palette]
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]

    exotic_count = max(1, len(coords) // 3)
    exotic_indices = set(map(int, rng.choice(len(coords), size=exotic_count, replace=False)))

    for idx, (local_r, local_c) in enumerate(coords):
        global_r = top + local_r
        global_c = left + local_c
        line_index = global_r if orientation == "row" else global_c
        cloth_color = int(line_colors[line_index])

        if idx in exotic_indices and exotic_colors:
            candidates = [color for color in exotic_colors if color != cloth_color]
        else:
            candidates = [color for color in stripe_palette if color != cloth_color]

        color = int(rng.choice(candidates))
        grid[global_r, global_c] = color
        sprite[local_r, local_c] = color

    return sprite


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    orientation: Literal["row", "col"] = "row" if rng.random() < 0.5 else "col"
    height = int(rng.integers(8, 25))
    width = int(rng.integers(8, 25))

    line_count = height if orientation == "row" else width
    line_colors = _choose_line_colors(line_count, rng)
    grid = _make_striped_grid(height, width, orientation, line_colors)

    box_h = int(rng.integers(3, min(height, 9) + 1))
    box_w = int(rng.integers(4, min(width, 10) + 1))
    top = int(rng.integers(0, height - box_h + 1))
    left = int(rng.integers(0, width - box_w + 1))

    if orientation == "row":
        mask = _sample_rowwise_mask(box_h, box_w, rng)
    else:
        mask = _sample_colwise_mask(box_h, box_w, rng)

    sprite = _paint_signals(grid, top, left, mask, orientation, line_colors, rng)
    if np.count_nonzero(sprite) < 4:
        return None

    expected = np.zeros((box_h, box_w), dtype=int)
    expected[mask] = sprite[mask]

    row_score, _ = _axis_purity_and_dominants(grid, axis=0)
    col_score, _ = _axis_purity_and_dominants(grid, axis=1)
    score_gap = abs(row_score - col_score)
    if score_gap < 0.25:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid striped-signal grid")
