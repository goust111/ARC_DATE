from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GRID = 5
CELL_SIZE = 3
STEP = CELL_SIZE + 1
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a non-empty color set"
    ranked = counts.most_common(2)
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _infer_line_indices(grid: np.ndarray, axis: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    major = g.shape[axis]
    minor = g.shape[1 - axis]
    out: list[int] = []
    for idx in range(major):
        vec = g[idx, :] if axis == 0 else g[:, idx]
        if np.all((vec == GRID) | (vec == BG)) and int(np.count_nonzero(vec == GRID)) >= minor - 4:
            out.append(int(idx))

    assert len(out) >= 4, "expected at least four grid lines"
    gaps = [b - a for a, b in zip(out, out[1:])]
    assert gaps and len(set(gaps)) == 1, "grid lines must be evenly spaced"
    assert gaps[0] >= 2, "grid cells must have positive interior width"
    return out


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_rows = _infer_line_indices(g, axis=0)
    line_cols = _infer_line_indices(g, axis=1)
    row_step = line_rows[1] - line_rows[0]
    col_step = line_cols[1] - line_cols[0]
    assert row_step == col_step, "row and column grid spacing must match"
    assert (row_step - 1) % 2 == 1, "cell interior must have an odd side length"

    missing = [(r, c) for r in line_rows for c in line_cols if int(g[r, c]) == BG]
    assert len(missing) == 4, "expected exactly four missing intersections"
    miss_rows = sorted({r for r, _ in missing})
    miss_cols = sorted({c for _, c in missing})
    assert len(miss_rows) == 2 and len(miss_cols) == 2, "missing intersections must form rectangle corners"
    expected_corners = {
        (miss_rows[0], miss_cols[0]),
        (miss_rows[0], miss_cols[1]),
        (miss_rows[1], miss_cols[0]),
        (miss_rows[1], miss_cols[1]),
    }
    assert set(missing) == expected_corners, "missing intersections must be the rectangle corners"

    r0, r1 = miss_rows
    c0, c1 = miss_cols
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    crop_line_rows = [r - r0 for r in line_rows if r0 <= r <= r1]
    crop_line_cols = [c - c0 for c in line_cols if c0 <= c <= c1]
    assert len(crop_line_rows) >= 3, "crop must contain at least two cell rows"
    assert len(crop_line_cols) >= 4, "crop must contain at least three stripes"

    interior_row_mask = np.ones(crop.shape[0], dtype=bool)
    interior_row_mask[crop_line_rows] = False
    marker_r = (crop_line_rows[0] + crop_line_rows[1]) // 2

    stripe_slices: list[slice] = []
    dominant_colors: list[int] = []
    marker_order: list[int] = []
    for left, right in zip(crop_line_cols, crop_line_cols[1:]):
        sl = slice(left + 1, right)
        stripe_slices.append(sl)

        stripe_region = crop[:, sl][interior_row_mask]
        stripe_colors = stripe_region[(stripe_region != BG) & (stripe_region != GRID)]
        assert stripe_colors.size > 0, "each stripe must have colored interior pixels"
        dominant_colors.append(_unique_mode(stripe_colors))

        marker_c = (left + right) // 2
        marker = int(crop[marker_r, marker_c])
        assert marker not in (BG, GRID), "marker pixels must be colored"
        marker_order.append(marker)

    assert len(set(dominant_colors)) == len(dominant_colors), "stripe dominant colors must be unique in the crop"
    assert len(set(marker_order)) == len(marker_order), "marker colors must be unique"
    assert set(marker_order) == set(dominant_colors), "marker colors must permute the stripe colors"

    return {
        "crop": crop,
        "bbox": (r0, c0, r1, c1),
        "line_rows": line_rows,
        "line_cols": line_cols,
        "crop_line_rows": crop_line_rows,
        "crop_line_cols": crop_line_cols,
        "stripe_slices": stripe_slices,
        "dominant_colors": dominant_colors,
        "marker_order": marker_order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the rectangle defined by the four missing grid intersections, then reorder its vertical stripes by the top marker colors.

    Input structure:
    - Gray pixels form a regular 1-pixel lattice grid; exactly four lattice intersections are replaced by black and form one axis-aligned rectangle.
    - Cell interiors are filled by vertical one-cell-wide color-major stripes; within the crop, each stripe has a unique dominant non-gray color.
    - In the crop's top cell row, each stripe contains one centered marker pixel whose color names one of the stripe dominant colors.

    Rule:
    - Detect the four missing gray intersections and crop the unique rectangle bounded by them, keeping the gray frame lines.
    - Inside the crop, identify each vertical stripe by its cell-column and determine its dominant color from all non-black, non-gray interior pixels.
    - Read the centered marker colors across the top crop row from left to right; this color sequence is the target stripe order.
    - Move whole stripe contents horizontally so the stripe whose dominant color matches the first marker goes leftmost, the next marker's stripe goes second, and so on.

    Key insight:
    - The missing intersections define both the output boundary and which stripe set participates, while the top-row marker colors specify a color-to-stripe permutation rather than an in-place recoloring.
    """
    details = _solve_details(grid)
    crop = details["crop"]
    stripe_slices = details["stripe_slices"]
    dominant_colors = details["dominant_colors"]
    marker_order = details["marker_order"]

    by_color = {
        color: crop[:, stripe_slice].copy()
        for color, stripe_slice in zip(dominant_colors, stripe_slices)
    }
    out = crop.copy()
    for dst_slice, color in zip(stripe_slices, marker_order):
        out[:, dst_slice] = by_color[color]
    return out


def _line_positions(count: int) -> list[int]:
    return [idx * STEP for idx in range(count + 1)]


def _cell_slice(lines: list[int], idx: int) -> tuple[slice, slice]:
    r0 = lines[idx] + 1
    r1 = lines[idx + 1]
    return slice(r0, r1), slice(r0, r1)


def _cell_bounds(lines: list[int], idx: int) -> tuple[int, int]:
    return lines[idx] + 1, lines[idx + 1] - 1


def _random_derangement(items: list[int]) -> list[int]:
    if len(items) < 2:
        raise ValueError("need at least two items for a derangement")
    while True:
        perm = items.copy()
        random.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return perm


def _build_expected(inp: np.ndarray, spec: dict) -> np.ndarray:
    top_line = spec["top_line"]
    bottom_line = spec["bottom_line"]
    left_line = spec["left_line"]
    right_line = spec["right_line"]
    crop = inp[top_line : bottom_line + 1, left_line : right_line + 1].copy()

    crop_cols = spec["crop_col_indices"]
    local_lines = [spec["line_cols"][idx] - left_line for idx in crop_cols]
    slices = [slice(local_lines[i] + 1, local_lines[i + 1]) for i in range(len(local_lines) - 1)]
    colors = spec["participant_colors"]
    order = spec["target_order"]

    by_color = {color: crop[:, sl].copy() for color, sl in zip(colors, slices)}
    out = crop.copy()
    for sl, color in zip(slices, order):
        out[:, sl] = by_color[color]
    return out


def _candidate_pixels(row_idx: int, col_idx: int, line_rows: list[int], line_cols: list[int]) -> list[tuple[int, int]]:
    r0, r1 = _cell_bounds(line_rows, row_idx)
    c0, c1 = _cell_bounds(line_cols, col_idx)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _paint_input(spec: dict) -> np.ndarray:
    n_rows = spec["n_rows"]
    n_cols = spec["n_cols"]
    line_rows = spec["line_rows"]
    line_cols = spec["line_cols"]
    stripe_colors = spec["all_stripe_colors"]
    top_row_idx = spec["top_row_idx"]
    left_col_idx = spec["left_col_idx"]
    crop_height = spec["crop_height"]
    crop_width = spec["crop_width"]
    target_order = spec["target_order"]

    grid = np.full((n_rows * STEP + 1, n_cols * STEP + 1), BG, dtype=int)
    grid[line_rows, :] = GRID
    grid[:, line_cols] = GRID

    for col_idx, color in enumerate(stripe_colors):
        c0, c1 = _cell_bounds(line_cols, col_idx)
        for row_idx in range(n_rows):
            r0, r1 = _cell_bounds(line_rows, row_idx)
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    crop_row_indices = list(range(top_row_idx, top_row_idx + crop_height + 1))
    crop_col_indices = list(range(left_col_idx, left_col_idx + crop_width + 1))
    spec["crop_row_indices"] = crop_row_indices
    spec["crop_col_indices"] = crop_col_indices
    spec["top_line"] = line_rows[crop_row_indices[0]]
    spec["bottom_line"] = line_rows[crop_row_indices[-1]]
    spec["left_line"] = line_cols[crop_col_indices[0]]
    spec["right_line"] = line_cols[crop_col_indices[-1]]

    marker_row = top_row_idx
    marker_r = (line_rows[marker_row] + line_rows[marker_row + 1]) // 2
    for offset, color in enumerate(target_order):
        col_idx = left_col_idx + offset
        marker_c = (line_cols[col_idx] + line_cols[col_idx + 1]) // 2
        grid[marker_r, marker_c] = int(color)

    for offset, base_color in enumerate(spec["participant_colors"]):
        col_idx = left_col_idx + offset
        row_choices = list(range(top_row_idx + 1, top_row_idx + crop_height))
        if not row_choices:
            row_choices = [top_row_idx]
        dot_row = random.choice(row_choices)
        candidates = [
            pos
            for pos in _candidate_pixels(dot_row, col_idx, line_rows, line_cols)
            if pos != ((line_rows[top_row_idx] + line_rows[top_row_idx + 1]) // 2, (line_cols[col_idx] + line_cols[col_idx + 1]) // 2)
        ]
        sig_r, sig_c = random.choice(candidates)
        alt_colors = [c for c in PALETTE if c != base_color]
        grid[sig_r, sig_c] = random.choice(alt_colors)

        extra_noise = random.randint(1, 3)
        all_cells = [(rr, col_idx) for rr in range(n_rows)]
        random.shuffle(all_cells)
        placed = 0
        for rr, cc in all_cells:
            candidates = _candidate_pixels(rr, cc, line_rows, line_cols)
            random.shuffle(candidates)
            for cand_r, cand_c in candidates:
                if rr == top_row_idx and (cand_r, cand_c) == (marker_r, (line_cols[col_idx] + line_cols[col_idx + 1]) // 2):
                    continue
                new_color = random.choice(alt_colors)
                if int(grid[cand_r, cand_c]) == new_color:
                    continue
                grid[cand_r, cand_c] = new_color
                placed += 1
                break
            if placed >= extra_noise:
                break

    corners = [
        (line_rows[top_row_idx], line_cols[left_col_idx]),
        (line_rows[top_row_idx], line_cols[left_col_idx + crop_width]),
        (line_rows[top_row_idx + crop_height], line_cols[left_col_idx]),
        (line_rows[top_row_idx + crop_height], line_cols[left_col_idx + crop_width]),
    ]
    for r, c in corners:
        grid[r, c] = BG

    return grid


def _validate_semantics(inp: np.ndarray, spec: dict, expected: np.ndarray) -> None:
    details = _solve_details(inp)
    assert tuple(details["bbox"]) == (
        spec["top_line"],
        spec["left_line"],
        spec["bottom_line"],
        spec["right_line"],
    ), "solver found the wrong crop rectangle"
    assert details["dominant_colors"] == spec["participant_colors"], "dominant stripe colors changed"
    assert details["marker_order"] == spec["target_order"], "marker order changed"
    assert all(a != b for a, b in zip(details["dominant_colors"], details["marker_order"])), "markers must form a derangement"
    assert details["crop"].shape == expected.shape, "crop and expected output must agree in shape"
    assert np.array_equal(main(inp), expected), "solver output must match the spec-built target"


def _generate_one() -> np.ndarray | None:
    n_rows = random.randint(4, 6)
    n_cols = random.randint(4, 6)
    line_rows = _line_positions(n_rows)
    line_cols = _line_positions(n_cols)

    crop_height = random.randint(2, n_rows)
    crop_width = random.randint(3, n_cols)
    top_row_idx = random.randint(0, n_rows - crop_height)
    left_col_idx = random.randint(0, n_cols - crop_width)
    if crop_height == n_rows and crop_width == n_cols:
        return None

    participant_colors = random.sample(PALETTE, crop_width)
    target_order = _random_derangement(participant_colors)

    all_stripe_colors: list[int] = []
    for col_idx in range(n_cols):
        if left_col_idx <= col_idx < left_col_idx + crop_width:
            all_stripe_colors.append(participant_colors[col_idx - left_col_idx])
        else:
            all_stripe_colors.append(random.choice(PALETTE))

    spec = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "crop_height": crop_height,
        "crop_width": crop_width,
        "top_row_idx": top_row_idx,
        "left_col_idx": left_col_idx,
        "participant_colors": participant_colors,
        "target_order": target_order,
        "all_stripe_colors": all_stripe_colors,
    }

    inp = _paint_input(spec)
    expected = _build_expected(inp, spec)

    if np.array_equal(expected, inp[spec["top_line"] : spec["bottom_line"] + 1, spec["left_line"] : spec["right_line"] + 1]):
        return None
    try:
        _validate_semantics(inp, spec, expected)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
