import math
import random
from typing import Iterable

import numpy as np


BG = 0
MASK = 6
PALETTE = (1, 2, 3, 4)
HEIGHT_CHOICES = (2, 3)
OBJECT_COUNT_CHOICES = (1, 2)
MASK_COUNT_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _nonzero_rows_in_col(grid: np.ndarray, c: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if int(grid[r, c]) != BG]


def _split_sequence(sequence: list[int]) -> list[list[int]]:
    n = math.isqrt(len(sequence))
    assert n * n == len(sequence), "column sequence length must be a perfect square"
    rows: list[list[int]] = []
    idx = 0
    for layer in range(n):
        width = 2 * layer + 1
        rows.append(sequence[idx : idx + width])
        idx += width
    assert idx == len(sequence)
    return rows


def _mirror_index(width: int, idx: int) -> int:
    return width - 1 - idx


def _build_pyramid_raw(grid: np.ndarray, center_c: int, sequence: list[int], bottom_r: int) -> np.ndarray:
    out = np.full_like(grid, BG)
    rows = _split_sequence(sequence)
    height = len(rows)
    for layer, row_vals in enumerate(rows):
        r = bottom_r - (height - 1) + layer
        half = layer
        c0 = center_c - half
        c1 = center_c + half
        assert 0 <= r < grid.shape[0]
        assert 0 <= c0 <= c1 < grid.shape[1]
        out[r, c0 : c1 + 1] = row_vals
    return out


def _repair_pyramid(pyramid: np.ndarray, center_c: int) -> np.ndarray:
    out = pyramid.copy()
    mask_coords = np.argwhere(out == MASK)
    for r, c in mask_coords:
        rr = int(r)
        cc = int(c)
        mirror_c = 2 * int(center_c) - cc
        assert 0 <= mirror_c < out.shape[1], "mask mirror must stay in bounds"
        mirror_color = int(out[rr, mirror_c])
        assert mirror_color != MASK and mirror_color != BG, "mask mirror must reveal one visible color"
        out[rr, cc] = mirror_color
    return out


def _extract_columns(grid: np.ndarray) -> list[tuple[int, list[int], int]]:
    H, W = grid.shape
    cols: list[tuple[int, list[int], int]] = []
    for c in range(W):
        rows = _nonzero_rows_in_col(grid, c)
        if not rows:
            continue
        sequence = [int(grid[r, c]) for r in rows]
        cols.append((c, sequence, max(rows)))
    assert cols, "expected at least one source column"
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange each masked source column into a centered pyramid and decode pink cells by mirror symmetry.

    Input structure:
    - The background is black.
    - There are one or two source columns, each occupying exactly one grid column.
    - Reading the non-black pixels in a source column from top to bottom yields n^2 values, split into blocks of lengths 1, 3, 5, ... for n in {2, 3}.
    - Some non-central entries inside width-3 or width-5 blocks are pink masks.

    Rule:
    - Clear the original source-column pixels to black.
    - For each source column, rebuild its top-to-bottom sequence as centered rows of widths 1, 3, 5, ... under the same column, with the bottom row aligned to the original bottommost source pixel row.
    - In that rebuilt pyramid, replace every pink cell with the color at its horizontal mirror position across the pyramid's vertical axis.
    - Keep all rebuilt pyramids in the same output grid.

    Key insight:
    - The source column alone only gives a one-dimensional readout; rearranging it into a centered pyramid creates the left-right symmetry that reveals each masked color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BG)
    for center_c, sequence, bottom_r in _extract_columns(g):
        pyramid = _build_pyramid_raw(g, center_c, sequence, bottom_r)
        repaired = _repair_pyramid(pyramid, center_c)
        overlap = (out != BG) & (repaired != BG)
        assert not bool(np.any(overlap)), "pyramids must not overlap"
        out[repaired != BG] = repaired[repaired != BG]
    return out


def _choice_bag(values: Iterable[int], *, weights: Iterable[float] | None = None) -> int:
    items = list(values)
    if weights is None:
        return int(random.choice(items))
    return int(random.choices(items, weights=list(weights), k=1)[0])


def _choose_centers(width: int, heights: list[int]) -> list[int] | None:
    centers: list[int] = []
    for i, height in enumerate(heights):
        radius = height - 1
        lo = radius
        hi = width - 1 - radius
        if lo > hi:
            return None
        candidates = []
        for c in range(lo, hi + 1):
            ok = True
            for prev_c, prev_h in zip(centers, heights[:i]):
                prev_radius = prev_h - 1
                if not (c + radius + 1 < prev_c - prev_radius or prev_c + prev_radius + 1 < c - radius):
                    ok = False
                    break
            if ok:
                candidates.append(c)
        if not candidates:
            return None
        centers.append(int(random.choice(candidates)))
    return centers


def _choose_bottom_rows(grid_h: int, heights: list[int]) -> list[int] | None:
    bottoms = []
    for height in heights:
        minimum = height * height + height - 2
        maximum = grid_h - 2
        if minimum > maximum:
            return None
        bottoms.append(random.randint(minimum, maximum))
    return bottoms


def _pal_row(width: int) -> list[int]:
    if width == 1:
        return [random.choice(PALETTE)]
    if width == 3:
        a, b = random.sample(PALETTE, 2)
        return [a, b, a]
    a, b, c = random.sample(PALETTE, 3)
    return [a, b, c, b, a]


def _row_patterns(height: int) -> list[list[int]]:
    return [_pal_row(2 * layer + 1) for layer in range(height)]


def _eligible_mask_positions(row_vals: list[int]) -> list[int]:
    width = len(row_vals)
    if width == 1:
        return []
    center = width // 2
    positions = []
    for idx in range(width):
        if idx == center:
            continue
        if row_vals[idx] == row_vals[_mirror_index(width, idx)]:
            positions.append(idx)
    return positions


def _place_masks(rows: list[list[int]], mask_count: int) -> tuple[list[list[int]], list[tuple[int, int, int]]]:
    masked = [row[:] for row in rows]
    candidates = []
    for layer, row_vals in enumerate(rows):
        for idx in _eligible_mask_positions(row_vals):
            mirror_idx = _mirror_index(len(row_vals), idx)
            if idx < mirror_idx:
                candidates.append((layer, idx, mirror_idx))

    random.shuffle(candidates)
    chosen: list[tuple[int, int, int]] = []
    used: set[tuple[int, int]] = set()
    for layer, idx, mirror_idx in candidates:
        if len(chosen) >= mask_count:
            break
        if (layer, idx) in used or (layer, mirror_idx) in used:
            continue
        chosen.append((layer, idx, mirror_idx))
        used.add((layer, idx))
        used.add((layer, mirror_idx))

    assert len(chosen) == mask_count, "must place requested number of non-conflicting masks"
    for layer, idx, _mirror_idx in chosen:
        masked[layer][idx] = MASK
    return masked, chosen


def _flatten(rows: list[list[int]]) -> list[int]:
    return [int(v) for row in rows for v in row]


def _scatter_rows(bottom_r: int, height: int, total_pixels: int) -> list[int] | None:
    top_pyramid_r = bottom_r - height + 1
    usable = list(range(top_pyramid_r))
    if len(usable) < total_pixels - 1:
        return None
    picked = sorted(random.sample(usable, total_pixels - 1))
    return picked + [bottom_r]


def _draw_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    H, W = shape
    out = np.full((H, W), BG, dtype=int)
    for spec in specs:
        center = int(spec["center"])
        bottom = int(spec["bottom"])
        rows = spec["rows"]
        for layer, row_vals in enumerate(rows):
            r = bottom - (len(rows) - 1) + layer
            c0 = center - layer
            patch = np.array(row_vals, dtype=int)
            overlap = (out[r, c0 : c0 + len(row_vals)] != BG) & (patch != BG)
            assert not bool(np.any(overlap))
            out[r, c0 : c0 + len(row_vals)] = patch
    return out


def _build_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    H, W = shape
    inp = np.full((H, W), BG, dtype=int)
    for spec in specs:
        center = int(spec["center"])
        seq = _flatten(spec["masked_rows"])
        source_rows = _scatter_rows(int(spec["bottom"]), int(spec["height"]), len(seq))
        if source_rows is None:
            return None
        assert len(source_rows) == len(seq)
        for r, color in zip(source_rows, seq):
            inp[int(r), center] = int(color)
    return inp


def _expected_output_from_input(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _validate_semantics(inp: np.ndarray, expected_out: np.ndarray, specs: list[dict]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected_out):
        return False

    for spec in specs:
        center = int(spec["center"])
        rows = _nonzero_rows_in_col(inp, center)
        if len(rows) != int(spec["height"]) ** 2:
            return False
        if max(rows) != int(spec["bottom"]):
            return False

        seq = [int(inp[r, center]) for r in rows]
        split = _split_sequence(seq)
        if len(split) != int(spec["height"]):
            return False

        rebuilt = _build_pyramid_raw(inp, center, seq, int(spec["bottom"]))
        for layer, idx, mirror_idx in spec["mask_triplets"]:
            r = int(spec["bottom"]) - (int(spec["height"]) - 1) + int(layer)
            c = center - layer + idx
            mirror_c = center - layer + mirror_idx
            if int(rebuilt[r, c]) != MASK:
                return False
            if int(rebuilt[r, mirror_c]) == MASK:
                return False
            target = int(spec["rows"][layer][idx])
            if int(rebuilt[r, mirror_c]) != target:
                return False
            flat_idx = sum(len(row) for row in split[:layer]) + idx
            neighbors = []
            if flat_idx > 0:
                neighbors.append(int(seq[flat_idx - 1]))
            if flat_idx + 1 < len(seq):
                neighbors.append(int(seq[flat_idx + 1]))
            if neighbors and all(val == target for val in neighbors):
                return False

        if int(spec["height"]) >= 2 and len(set(spec["rows"][1])) < 2:
            return False
        if int(spec["height"]) == 3 and len(set(spec["rows"][2])) < 3:
            return False

    if len(specs) == 2:
        centers = [int(spec["center"]) for spec in specs]
        heights = [int(spec["height"]) for spec in specs]
        left = min(centers[0] - heights[0] + 1, centers[1] - heights[1] + 1)
        right = max(centers[0] + heights[0] - 1, centers[1] + heights[1] - 1)
        if right - left + 1 >= inp.shape[1]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(16, 24)
    W = random.randint(11, 20)
    count = _choice_bag(OBJECT_COUNT_CHOICES, weights=(1.0, 1.0))
    heights = [_choice_bag(HEIGHT_CHOICES, weights=(1.0, 1.0)) for _ in range(count)]

    centers = _choose_centers(W, heights)
    if centers is None:
        return None

    bottoms = _choose_bottom_rows(H, heights)
    if bottoms is None:
        return None
    specs: list[dict] = []
    for center, height, bottom in zip(centers, heights, bottoms):
        rows = _row_patterns(height)
        mask_count = min(_choice_bag(MASK_COUNT_CHOICES, weights=(1.0, 1.0)), sum(len(_eligible_mask_positions(row)) > 0 for row in rows))
        if mask_count <= 0:
            return None
        masked_rows, mask_triplets = _place_masks(rows, mask_count)
        specs.append(
            {
                "center": int(center),
                "height": int(height),
                "bottom": int(bottom),
                "rows": rows,
                "masked_rows": masked_rows,
                "mask_triplets": mask_triplets,
            }
        )

    expected_out = _draw_output((H, W), specs)
    inp = _build_input((H, W), specs)
    if inp is None:
        return None
    if not _validate_semantics(inp, expected_out, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt
    raise RuntimeError(f"failed to generate valid input in {MAX_GENERATION_ATTEMPTS} attempts")
