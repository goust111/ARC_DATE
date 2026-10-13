from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
SPRITE_PALETTE = (1, 2, 3, 4, 7, 8)
MARKER_COLOR = 4
MAX_GENERATION_ATTEMPTS = 500


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _window_from_runs(row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            windows.append((r0, r1, c0, c1))
    return windows


def _find_sprite_windows(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    row_runs = _active_runs(np.any(grid != BG, axis=1))
    candidates: list[tuple[list[tuple[int, int, int, int]], int]] = []
    for idx in range(len(row_runs) - 2):
        trio = row_runs[idx : idx + 3]
        if any(r1 - r0 + 1 != 4 for r0, r1 in trio):
            continue
        if trio[1][0] - trio[0][1] - 1 not in (1, 2):
            continue
        if trio[2][0] - trio[1][1] - 1 not in (1, 2):
            continue

        r0 = trio[0][0]
        r1 = trio[2][1]
        band = grid[r0 : r1 + 1]
        col_runs = _active_runs(np.any(band != BG, axis=0))
        if len(col_runs) != 2:
            continue
        if any(c1 - c0 + 1 != 4 for c0, c1 in col_runs):
            continue
        if col_runs[1][0] - col_runs[0][1] - 1 not in (1, 2):
            continue

        windows = _window_from_runs(trio, col_runs)
        colors: list[int] = []
        valid = True
        for wr0, wr1, wc0, wc1 in windows:
            patch = grid[wr0 : wr1 + 1, wc0 : wc1 + 1]
            nonzero = sorted(int(v) for v in np.unique(patch) if int(v) != BG)
            if len(nonzero) != 1:
                valid = False
                break
            if np.count_nonzero(patch) == 0:
                valid = False
                break
            colors.append(nonzero[0])
        if valid and len(set(colors)) == 6:
            candidates.append((windows, r1))

    assert len(candidates) == 1, "expected exactly one 3x2 sprite array"
    return candidates[0]


def _extract_sprite_masks_and_colors(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray], list[int], int]:
    windows, sprite_bottom = _find_sprite_windows(grid)
    masks: list[np.ndarray] = []
    colors: list[int] = []
    for r0, r1, c0, c1 in windows:
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(next(v for v in np.unique(patch) if int(v) != BG))
        masks.append(patch != BG)
        colors.append(color)
    return windows, masks, colors, sprite_bottom


def _find_union_triple(masks: list[np.ndarray]) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        for i, j in combinations([k for k in range(len(masks)) if k != dst], 2):
            mask_i = masks[i]
            mask_j = masks[j]
            if not np.array_equal(mask_i | mask_j, masks[dst]):
                continue
            if not np.any(mask_i & ~mask_j):
                continue
            if not np.any(mask_j & ~mask_i):
                continue
            candidates.append((i, j, dst))
    assert len(candidates) == 1, "expected a unique union triple"
    return candidates[0]


def _lower_region_bbox(grid: np.ndarray, sprite_bottom: int) -> tuple[int, int, int, int]:
    tail = grid[sprite_bottom + 1 :]
    coords = np.argwhere(tail != BG)
    assert coords.size > 0, "expected a lower colored region"
    coords[:, 0] += sprite_bottom + 1
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _selected_segment_mask(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    row_pairs = [r for r in range(1, h - 1) if region[r, 0] == MARKER_COLOR and region[r, w - 1] == MARKER_COLOR]
    col_pairs = [c for c in range(1, w - 1) if region[0, c] == MARKER_COLOR and region[h - 1, c] == MARKER_COLOR]
    assert 2 <= len(row_pairs) + len(col_pairs) <= 3, "expected 2-3 marker pairs"

    expected_markers = {(r, 0) for r in row_pairs} | {(r, w - 1) for r in row_pairs}
    expected_markers |= {(0, c) for c in col_pairs} | {(h - 1, c) for c in col_pairs}
    actual_markers = {
        (int(r), int(c))
        for r, c in np.argwhere(region == MARKER_COLOR)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    assert actual_markers == expected_markers, "expected all boundary yellow pixels to be valid marker endpoints"

    selected = np.zeros(region.shape, dtype=bool)
    for r in row_pairs:
        selected[r, 1 : w - 1] = True
    for c in col_pairs:
        selected[1 : h - 1, c] = True
    assert np.any(selected), "expected selected interior cells"
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the operand colors from a sprite-mask union triple only along marker-selected interior row and column segments.

    Input structure:
    - The upper part contains six separated 4x4 monochrome sprite windows arranged as a 3x2 array on a black background.
    - The six sprite colors are distinct, and exactly one pair of sprite masks has a pixelwise union equal to a third sprite mask.
    - The lower part contains one non-black rectangular color region whose yellow edge markers appear as 2-3 opposite-edge row or column pairs.

    Rule:
    - Extract the six 4x4 sprite masks and find the unique triple where mask(A) OR mask(B) equals mask(C), with A and B each contributing at least one exclusive pixel.
    - Take the two operand sprite colors as the dynamic swap pair.
    - In the lower rectangular region, every left-right yellow pair selects that row's strict interior and every top-bottom yellow pair selects that column's strict interior.
    - On the union of all selected interior cells, swap the two operand colors and leave every other color and every other cell unchanged.

    Key insight:
    - The top sprite array determines which two colors to swap, while the lower yellow markers determine exactly where that swap is allowed to happen.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _windows, masks, colors, sprite_bottom = _extract_sprite_masks_and_colors(g)
    src_a, src_b, _dst = _find_union_triple(masks)
    color_a = colors[src_a]
    color_b = colors[src_b]

    r0, c0, r1, c1 = _lower_region_bbox(g, sprite_bottom)
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    selected = _selected_segment_mask(region)

    out = g.copy()
    region_out = out[r0 : r1 + 1, c0 : c1 + 1]
    swap_a = selected & (region == color_a)
    swap_b = selected & (region == color_b)
    region_out[swap_a] = color_b
    region_out[swap_b] = color_a
    return out


def _mask_from_positions(positions: np.ndarray) -> np.ndarray:
    mask = np.zeros((4, 4), dtype=bool)
    mask.flat[np.asarray(positions, dtype=int)] = True
    return mask


def _random_operand_masks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    union_count = int(rng.integers(7, 13))
    union_positions = rng.choice(16, size=union_count, replace=False)
    overlap_count = int(rng.integers(1, union_count - 1))
    overlap_positions = rng.choice(union_positions, size=overlap_count, replace=False)
    leftovers = np.array([p for p in union_positions if p not in set(map(int, overlap_positions))], dtype=int)
    if len(leftovers) < 2:
        raise ValueError("not enough exclusive union pixels")
    cut = int(rng.integers(1, len(leftovers)))
    rng.shuffle(leftovers)
    only_a = leftovers[:cut]
    only_b = leftovers[cut:]
    if len(only_a) == 0 or len(only_b) == 0:
        raise ValueError("operands must both contribute exclusive pixels")
    mask_a = _mask_from_positions(np.concatenate([overlap_positions, only_a]))
    mask_b = _mask_from_positions(np.concatenate([overlap_positions, only_b]))
    mask_c = mask_a | mask_b
    return mask_a, mask_b, mask_c


def _random_mask(rng: np.random.Generator) -> np.ndarray:
    count = int(rng.integers(4, 13))
    return _mask_from_positions(rng.choice(16, size=count, replace=False))


def _unique_union_masks(rng: np.random.Generator) -> list[np.ndarray] | None:
    try:
        a, b, c = _random_operand_masks(rng)
    except ValueError:
        return None
    masks = [a, b, c] + [_random_mask(rng) for _ in range(3)]
    if len({_mask_key(mask) for mask in masks}) != 6:
        return None
    if _find_union_candidates(masks) != [(0, 1, 2)]:
        return None
    return masks


def _find_union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        others = [k for k in range(len(masks)) if k != dst]
        for i, j in combinations(others, 2):
            if not np.array_equal(masks[i] | masks[j], masks[dst]):
                continue
            if not np.any(masks[i] & ~masks[j]):
                continue
            if not np.any(masks[j] & ~masks[i]):
                continue
            candidates.append((i, j, dst))
    return candidates


def _spaced_choices(limit: int, count: int) -> list[tuple[int, ...]]:
    positions = range(1, limit - 1)
    valid = []
    for combo in combinations(positions, count):
        if all(b - a >= 2 for a, b in zip(combo, combo[1:])):
            valid.append(combo)
    return valid


def _sample_axis_pairs(limit: int, counts: tuple[int, ...], rng: np.random.Generator) -> list[int]:
    options: list[tuple[int, ...]] = []
    for count in counts:
        options.extend(_spaced_choices(limit, count))
    assert options, "expected feasible marker choices"
    return list(options[int(rng.integers(len(options)))])


def _sample_mixed_pairs(h: int, w: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    total = 2 if rng.random() < 0.5 else 3
    row_count = 1 if total == 2 else int(rng.integers(1, 3))
    col_count = total - row_count
    row_options = _spaced_choices(h, row_count)
    col_options = _spaced_choices(w, col_count)
    assert row_options and col_options, "expected feasible mixed marker choices"
    row_pairs = list(row_options[int(rng.integers(len(row_options)))])
    col_pairs = list(col_options[int(rng.integers(len(col_options)))])
    return row_pairs, col_pairs


def _build_region(
    h: int,
    w: int,
    color_a: int,
    color_b: int,
    row_pairs: list[int],
    col_pairs: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    extra_count = int(rng.integers(2, 5))
    extras = [c for c in SPRITE_PALETTE if c not in {color_a, color_b}]
    rng.shuffle(extras)
    fill_palette = [color_a, color_b, *extras[:extra_count]]
    if color_a == MARKER_COLOR or color_b == MARKER_COLOR:
        if MARKER_COLOR not in fill_palette:
            fill_palette.append(MARKER_COLOR)
    fill_palette = sorted(set(fill_palette))

    region = np.zeros((h, w), dtype=int)
    edge_palette = [c for c in fill_palette if c != MARKER_COLOR]
    assert edge_palette, "need a non-marker boundary color"

    region[1 : h - 1, 1 : w - 1] = rng.choice(fill_palette, size=(h - 2, w - 2))
    region[0, :] = rng.choice(edge_palette, size=w)
    region[h - 1, :] = rng.choice(edge_palette, size=w)
    region[:, 0] = rng.choice(edge_palette, size=h)
    region[:, w - 1] = rng.choice(edge_palette, size=h)

    for r in row_pairs:
        region[r, 0] = MARKER_COLOR
        region[r, w - 1] = MARKER_COLOR
    for c in col_pairs:
        region[0, c] = MARKER_COLOR
        region[h - 1, c] = MARKER_COLOR

    selected = np.zeros((h, w), dtype=bool)
    for r in row_pairs:
        selected[r, 1 : w - 1] = True
    for c in col_pairs:
        selected[1 : h - 1, c] = True

    selected_coords = [tuple(map(int, rc)) for rc in np.argwhere(selected)]
    outside_coords = [
        tuple(map(int, rc))
        for rc in np.argwhere(np.ones((h, w), dtype=bool) & ~selected)
        if 0 < rc[0] < h - 1 and 0 < rc[1] < w - 1
    ]
    assert len(selected_coords) >= 5 and len(outside_coords) >= 2

    rng.shuffle(selected_coords)
    rng.shuffle(outside_coords)
    region[selected_coords[0]] = color_a
    region[selected_coords[1]] = color_b
    region[selected_coords[2]] = color_a
    region[selected_coords[3]] = color_b
    region[selected_coords[4]] = extras[0] if extras else fill_palette[0]
    region[outside_coords[0]] = color_a
    region[outside_coords[1]] = color_b
    return region, selected


def _build_expected_output(
    grid: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    selected: np.ndarray,
    color_a: int,
    color_b: int,
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = region_bbox
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    region_out = out[r0 : r1 + 1, c0 : c1 + 1]
    mask_a = selected & (region == color_a)
    mask_b = selected & (region == color_b)
    region_out[mask_a] = color_b
    region_out[mask_b] = color_a
    return out


def _construct_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    masks = _unique_union_masks(rng)
    if masks is None:
        raise ValueError("failed to sample unique union masks")

    role_colors = list(rng.permutation(np.array(SPRITE_PALETTE, dtype=int)))
    color_a = int(role_colors[0])
    color_b = int(role_colors[1])

    top_margin = 1
    left_margin = int(rng.integers(1, 4))
    row_gaps = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    col_gap = int(rng.integers(1, 3))
    upper_h = 12 + sum(row_gaps)
    upper_w = 8 + col_gap

    region_h = int(rng.integers(7, 10))
    region_w = int(rng.integers(7, 11))
    mode = int(rng.integers(3))
    row_pairs, col_pairs = [], []
    if mode == 0:
        row_pairs = _sample_axis_pairs(region_h, (2, 3), rng)
    elif mode == 1:
        col_pairs = _sample_axis_pairs(region_w, (2, 3), rng)
    else:
        row_pairs, col_pairs = _sample_mixed_pairs(region_h, region_w, rng)
    if len(row_pairs) + len(col_pairs) not in (2, 3):
        raise ValueError("need 2-3 marker pairs")

    section_gap = int(rng.integers(1, 3))
    lower_r0 = top_margin + upper_h + section_gap
    lower_c0 = int(rng.integers(1, 9))
    total_h = lower_r0 + region_h + int(rng.integers(1, 3))
    total_w = max(left_margin + upper_w, lower_c0 + region_w) + int(rng.integers(1, 3))
    if total_h > 30 or total_w > 30:
        raise ValueError("grid too large")

    grid = np.zeros((total_h, total_w), dtype=int)

    row_starts = [
        top_margin,
        top_margin + 4 + row_gaps[0],
        top_margin + 8 + row_gaps[0] + row_gaps[1],
    ]
    col_starts = [left_margin, left_margin + 4 + col_gap]
    windows = [(r, c) for r in row_starts for c in col_starts]
    placement_order = list(rng.permutation(6))
    for window_idx, role_idx in enumerate(placement_order):
        r0, c0 = windows[window_idx]
        patch = grid[r0 : r0 + 4, c0 : c0 + 4]
        patch[masks[role_idx]] = int(role_colors[role_idx])

    region, selected = _build_region(region_h, region_w, color_a, color_b, row_pairs, col_pairs, rng)
    grid[lower_r0 : lower_r0 + region_h, lower_c0 : lower_c0 + region_w] = region
    region_bbox = (lower_r0, lower_c0, lower_r0 + region_h - 1, lower_c0 + region_w - 1)
    expected = _build_expected_output(grid, region_bbox, selected, color_a, color_b)
    metadata = {
        "swap_pair": (color_a, color_b),
        "region_bbox": region_bbox,
        "selected": selected,
        "row_pairs": row_pairs,
        "col_pairs": col_pairs,
        "expected": expected,
    }
    return grid, expected, metadata


def _validate_instance(grid: np.ndarray, expected: np.ndarray, metadata: dict) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False
    if int(np.count_nonzero(grid != expected)) < 4:
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    r0, c0, r1, c1 = metadata["region_bbox"]
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    selected = metadata["selected"]
    color_a, color_b = metadata["swap_pair"]
    if len(set(int(v) for v in np.unique(region))) < 4:
        return False
    if not np.any(selected & (region == color_a)):
        return False
    if not np.any(selected & (region == color_b)):
        return False
    interior_unselected = np.zeros(region.shape, dtype=bool)
    interior_unselected[1:-1, 1:-1] = True
    interior_unselected &= ~selected
    if not np.any(interior_unselected & (region == color_a)):
        return False
    if not np.any(interior_unselected & (region == color_b)):
        return False
    if not np.any(selected & ~np.isin(region, [color_a, color_b])):
        return False

    _, masks, _colors, _sprite_bottom = _extract_sprite_masks_and_colors(grid)
    if _find_union_candidates(masks) != [_find_union_triple(masks)]:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        try:
            grid, expected, metadata = _construct_instance(rng)
        except Exception:
            continue
        if _validate_instance(grid, expected, metadata):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
