from __future__ import annotations

import math
import random

import numpy as np


BG = 0
RED = 2
SEP = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(np.asarray(mask, dtype=bool).tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _stripe_spans(height: int, sep_rows: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [r + 1 for r in sep_rows]
    ends = [r - 1 for r in sep_rows] + [height - 1]
    spans = [(r0, r1) for r0, r1 in zip(starts, ends) if r0 <= r1]
    assert spans, "expected at least one stripe"
    stripe_heights = {r1 - r0 + 1 for r0, r1 in spans}
    assert len(stripe_heights) == 1, "horizontal stripes must have equal height"
    return spans


def _separator_rows(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    rows: list[int] = []
    sep_color: int | None = None
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color == bg or not np.all(row == color):
            continue
        rows.append(r)
        if sep_color is None:
            sep_color = color
        else:
            assert color == sep_color, "all separator rows must share one color"
    assert rows, "expected solid separator rows"
    assert sep_color is not None
    return rows, sep_color


def _extract_patches(stripe: np.ndarray, bg: int) -> list[np.ndarray]:
    active_rows = np.any(stripe != bg, axis=1)
    row_runs = _runs(active_rows)
    assert len(row_runs) == 1, "each stripe should contain one aligned template/copy row"
    r0, r1 = row_runs[0]
    core = stripe[r0 : r1 + 1]

    active_cols = np.any(core != bg, axis=0)
    col_runs = _runs(active_cols)
    assert len(col_runs) >= 2, "each stripe must contain a template and at least one copy"

    widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(widths) == 1, "template and copies must share one width"

    patches: list[np.ndarray] = []
    for c0, c1 in col_runs:
        patch = core[:, c0 : c1 + 1].copy()
        assert np.all(np.any(patch != bg, axis=0)), "each patch bbox must keep every column active"
        assert np.all(np.any(patch != bg, axis=1)), "each patch bbox must keep every row active"
        patches.append(patch)
    return patches


def _paste_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _random_mask(height: int, width: int, min_cells: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    max_holes = max(0, height * width - min_cells)
    hole_target = rng.randint(0, max_holes)
    cells = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(cells)

    removed = 0
    for r, c in cells:
        if removed >= hole_target:
            break
        mask[r, c] = False
        if not mask[r].any() or not mask[:, c].any():
            mask[r, c] = True
            continue
        removed += 1
    return mask


def _random_template(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    template = np.full(mask.shape, BG, dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    color_count = rng.randint(2, min(4, len(coords)))
    palette = rng.sample(list(TEMPLATE_COLORS), color_count)
    rng.shuffle(coords)

    for color, (r, c) in zip(palette, coords):
        template[r, c] = color
    for r, c in coords[color_count:]:
        template[r, c] = rng.choice(palette)
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather the red defect locations from each stripe's corrupted template copies into one template-sized canvas.

    Input structure:
    - The background is black, and solid gray rows split the grid into equal-height horizontal stripes.
    - Inside each stripe, the leftmost patch is a multicolor template and the patches to its right are same-sized copies.
    - Every copy matches the template except for exactly one active pixel that has been replaced by red.

    Rule:
    - Detect the separator rows, then process the stripes from top to bottom.
    - In each stripe, treat the leftmost patch as the template and compare every right-side patch against it.
    - For the unique red mismatch in each copy, write the template's original color at that relative coordinate on a blank output grid with the template's shape.
    - Later stripes overwrite earlier stripes when multiple copies point to the same output cell.

    Key insight:
    - The red pixel is only a visible mismatch marker; the real answer is the collection of the corresponding template colors, merged across all stripes on shared template coordinates.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    sep_rows, _sep_color = _separator_rows(g, bg)
    spans = _stripe_spans(g.shape[0], sep_rows)

    out: np.ndarray | None = None
    for r0, r1 in spans:
        stripe = g[r0 : r1 + 1]
        patches = _extract_patches(stripe, bg)
        template = patches[0]
        assert not np.any(template == RED), "template cells must never be red"

        if out is None:
            out = np.full(template.shape, bg, dtype=int)
        else:
            assert template.shape == out.shape, "all stripes must share one template size"

        for copy in patches[1:]:
            assert copy.shape == template.shape, "copy size must match template size"
            diffs = np.argwhere(copy != template)
            assert len(diffs) == 1, "each copy must have exactly one defect"
            rr, cc = map(int, diffs[0])
            assert int(copy[rr, cc]) == RED, "the defect marker must be red"
            assert int(template[rr, cc]) not in (BG, RED), "the replaced template pixel must be a non-red foreground color"
            out[rr, cc] = template[rr, cc]

    assert out is not None
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stripe_count = rng.choice([3, 4])
    copies_per_stripe = [rng.choice([1, 2]) for _ in range(stripe_count)]
    total_copies = sum(copies_per_stripe)

    template_h = rng.randint(3, 5)
    template_w = rng.randint(3, 6)
    stripe_h = template_h + rng.choice([2, 3])
    total_h = stripe_count * stripe_h + (stripe_count - 1)
    if total_h > 30:
        return None

    min_cells = max(total_copies, template_h + template_w - 1, math.ceil(template_h * template_w * 0.6))
    mask = _random_mask(template_h, template_w, min_cells, rng)
    active_positions = [(int(r), int(c)) for r, c in np.argwhere(mask)]

    overlap_branch = rng.random() < 0.5
    unique_needed = total_copies if not overlap_branch else total_copies - 1
    if len(active_positions) < unique_needed:
        return None

    unique_positions = rng.sample(active_positions, unique_needed)
    overlap_target = rng.randrange(1, stripe_count) if overlap_branch else None

    positions_by_stripe: list[list[tuple[int, int]]] = []
    cursor = 0
    if overlap_branch:
        shared = unique_positions[0]
        leftovers = unique_positions[1:]
        for i, count in enumerate(copies_per_stripe):
            picks: list[tuple[int, int]] = []
            if i == 0 or i == overlap_target:
                picks.append(shared)
                count -= 1
            picks.extend(leftovers[cursor : cursor + count])
            cursor += count
            rng.shuffle(picks)
            positions_by_stripe.append(picks)
    else:
        for count in copies_per_stripe:
            picks = unique_positions[cursor : cursor + count]
            cursor += count
            rng.shuffle(picks)
            positions_by_stripe.append(list(picks))

    layouts: list[tuple[int, list[int]]] = []
    max_right = 0
    for count in copies_per_stripe:
        start_col = rng.randint(1, 2)
        gaps = [rng.randint(1, 2) for _ in range(count)]
        x_positions = [start_col]
        cursor_col = start_col + template_w
        for gap in gaps:
            cursor_col += gap
            x_positions.append(cursor_col)
            cursor_col += template_w
        layouts.append((rng.randint(0, stripe_h - template_h), x_positions))
        max_right = max(max_right, cursor_col)

    total_w = max_right + rng.randint(1, 2)
    if total_w > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    for band in range(stripe_count - 1):
        sep_row = (band + 1) * stripe_h + band
        grid[sep_row, :] = SEP

    expected = np.full((template_h, template_w), BG, dtype=int)
    prior_colors: dict[tuple[int, int], int] = {}

    for stripe_idx in range(stripe_count):
        template = _random_template(mask, rng)
        diff_positions = positions_by_stripe[stripe_idx]
        for pos in diff_positions:
            if pos in prior_colors and int(template[pos]) == prior_colors[pos]:
                alternatives = [c for c in TEMPLATE_COLORS if c != prior_colors[pos]]
                template[pos] = rng.choice(alternatives)
        band_top = stripe_idx * (stripe_h + 1)
        row_offset, x_positions = layouts[stripe_idx]
        patch_top = band_top + row_offset

        _paste_patch(grid, patch_top, x_positions[0], template)
        for copy_idx, pos in enumerate(diff_positions, start=1):
            copy = template.copy()
            copy[pos] = RED
            _paste_patch(grid, patch_top, x_positions[copy_idx], copy)
            expected[pos] = template[pos]
            prior_colors[pos] = int(template[pos])

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-template input")
