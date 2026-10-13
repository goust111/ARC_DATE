from __future__ import annotations

import random

import numpy as np


GREY = 5
RED = 2
AREA_COLORS = (1, 2, 3, 4, 6, 7)
GEN_HEIGHT_RANGE = (8, 10)
GEN_WIDTH_RANGE = (8, 10)
MAX_GENERATION_ATTEMPTS = 500


def _append_mirror(grid: np.ndarray) -> np.ndarray:
    return np.concatenate([grid, np.fliplr(grid)], axis=1)


def _candidate_pair_offsets(width: int) -> list[int]:
    return list(range(width // 2))


def _emergent_separator_specs(grid: np.ndarray) -> list[dict[str, int | str]]:
    """Describe one emergent separator per complementary source-column pair."""
    _h, w = grid.shape
    specs: list[dict[str, int | str]] = []
    for j in _candidate_pair_offsets(w):
        left = grid[:, j] == GREY
        right = grid[:, w - 1 - j] == GREY
        if left.all() or right.all():
            continue
        if np.all(left | right):
            left_count = int(left.sum())
            right_count = int(right.sum())
            if left_count <= right_count:
                active_col = j
                active_side = "L"
            else:
                active_col = w - 1 - j
                active_side = "R"
            specs.append(
                {
                    "pair": j,
                    "active_col": active_col,
                    "position": 2 * w - 1 - active_col,
                    "active_side": active_side,
                }
            )
    return specs


def _materialize_separator_board(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    g = np.asarray(grid, dtype=int)
    board = _append_mirror(g)
    specs = _emergent_separator_specs(g)
    for spec in specs:
        board[:, int(spec["position"])] = GREY
    return board, specs


def _bands_from_separators(separator_flags: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(separator_flags.tolist()):
        if is_separator:
            if start is not None:
                bands.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        bands.append((start, len(separator_flags)))
    return bands


def _partition_regions(board: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray, np.ndarray]:
    sep_rows = np.all(board == GREY, axis=1)
    sep_cols = np.all(board == GREY, axis=0)
    row_bands = _bands_from_separators(sep_rows)
    col_bands = _bands_from_separators(sep_cols)

    regions: list[dict[str, int]] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            area = (r1 - r0) * (c1 - c0)
            if area > 0:
                regions.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "area": area})
    return regions, sep_rows, sep_cols


def _render_output(board: np.ndarray) -> np.ndarray:
    regions, sep_rows, sep_cols = _partition_regions(board)
    assert 1 <= len(regions) <= len(AREA_COLORS), "region count exceeds available recolor roles"

    ranked = sorted(regions, key=lambda region: (-region["area"], region["r0"], region["c0"]))

    out = board.copy()
    for rank, region in enumerate(ranked):
        color = AREA_COLORS[rank]
        out[region["r0"] : region["r1"], region["c0"] : region["c1"]] = color

    out[sep_rows, :] = GREY
    out[:, sep_cols] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the board, let complementary grey column pairs create separator columns, then recolor regions by area rank.

    Input structure:
    - The input is a black-background grid with grey structure pixels and a few red marker pixels.
    - Some source columns are paired with their left-right mirror partners so that, row by row, at least one of the two partner cells is grey.
    - The input may also contain an already complete grey separator row; red pixels are only distractor markers.

    Rule:
    - Concatenate the input with its left-right mirror, keeping the original on the left.
    - For every source offset j, compare column j with column W-1-j; if their grey cells jointly cover every row while neither column is already complete, paint the mirrored copy column W+j solid grey to materialize an emergent separator.
    - Treat every complete grey row and every complete grey column in that mirrored board as separator lines, split the board into rectangular regions, rank regions by area from large to small, and recolor them with 1,2,3,4,6,7 in that order.
    - Separator lines stay grey.

    Key insight:
    - The separator layout is not readable from the input alone: you must mirror first, then test complementary source-column pairs to discover which separator columns emerge, and only then can you compare region areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    board, _specs = _materialize_separator_board(g)
    return _render_output(board)


def _separator_positions(width: int, plans: list[tuple[int, str]]) -> list[int]:
    positions: list[int] = []
    for pair, active_side in plans:
        if active_side == "L":
            active_col = pair
        else:
            active_col = width - 1 - pair
        positions.append(2 * width - 1 - active_col)
    return sorted(positions)


def _column_band_widths(width: int, plans: list[tuple[int, str]]) -> list[int]:
    separators = _separator_positions(width, plans)
    band_widths: list[int] = []
    prev = 0
    for sep in separators:
        band_widths.append(sep - prev)
        prev = sep + 1
    band_widths.append(2 * width - prev)
    return band_widths


def _sample_layout() -> tuple[int, int, list[tuple[int, str]], int | None]:
    layouts = (
        (1, True),
        (2, False),
        (2, True),
        (3, False),
    )
    emergent_count, has_row_separator = random.choice(layouts)

    width_min = max(GEN_WIDTH_RANGE[0], 10 if emergent_count == 3 else GEN_WIDTH_RANGE[0])
    width = random.randint(width_min, GEN_WIDTH_RANGE[1])
    height = random.randint(*GEN_HEIGHT_RANGE)

    candidate_offsets = _candidate_pair_offsets(width)
    row_separator = None

    for _ in range(200):
        pairs = sorted(random.sample(candidate_offsets, emergent_count))
        side_choices = [random.choice(("L", "R")) for _ in pairs]
        plans = list(zip(pairs, side_choices))
        band_widths = _column_band_widths(width, plans)
        if any(b <= 0 for b in band_widths):
            continue
        if len(set(band_widths)) != len(band_widths):
            continue

        if not has_row_separator:
            return height, width, plans, None

        for _ in range(100):
            row_separator = random.randint(1, height - 2)
            top = row_separator
            bottom = height - row_separator - 1
            if top == bottom:
                continue
            areas = [top * band for band in band_widths] + [bottom * band for band in band_widths]
            if len(set(areas)) == len(areas):
                return height, width, plans, row_separator

    raise RuntimeError("failed to sample a non-degenerate partition layout")


def _paint_emergent_pair(grid: np.ndarray, offset: int, active_side: str, row_separator: int | None) -> None:
    h, w = grid.shape
    left_col = offset
    right_col = w - 1 - offset
    active_rows = [r for r in range(h) if r != row_separator]
    minority_size = random.randint(1, max(1, (len(active_rows) - 1) // 2))
    minority_rows = set(random.sample(active_rows, minority_size))
    majority_rows = [r for r in active_rows if r not in minority_rows]

    if active_side == "L":
        left_rows = minority_rows
        right_rows = majority_rows
    else:
        left_rows = majority_rows
        right_rows = minority_rows

    for r in left_rows:
        grid[r, left_col] = GREY
    for r in right_rows:
        grid[r, right_col] = GREY


def _paint_counterexample_pair(grid: np.ndarray, offset: int, row_separator: int | None) -> None:
    h, w = grid.shape
    left_col = offset
    right_col = w - 1 - offset
    active_rows = [r for r in range(h) if r != row_separator]
    miss_count = random.choice((1, 2)) if len(active_rows) >= 4 else 1
    uncovered = set(random.sample(active_rows, miss_count))
    almost_full_is_left = random.random() < 0.5
    almost_col = left_col if almost_full_is_left else right_col
    other_col = right_col if almost_full_is_left else left_col

    for r in active_rows:
        if r not in uncovered:
            grid[r, almost_col] = GREY

    covered_elsewhere = set(random.sample(active_rows, random.randint(1, max(1, len(active_rows) // 3))))
    covered_elsewhere -= uncovered
    for r in covered_elsewhere:
        grid[r, other_col] = GREY


def _place_red_markers(
    grid: np.ndarray,
    row_separator: int | None,
    blocked_cols: set[int],
    desired_count: int,
) -> None:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == 0 and r != row_separator and c not in blocked_cols
    ]
    random.shuffle(candidates)
    for r, c in candidates[:desired_count]:
        grid[r, c] = RED


def _place_grey_speckles(
    grid: np.ndarray,
    row_separator: int | None,
    blocked_cols: set[int],
    max_count: int,
) -> None:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == 0 and r != row_separator and c not in blocked_cols
    ]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(0, max_count)]:
        grid[r, c] = GREY


def _has_required_counterexample(grid: np.ndarray, row_separator: int | None, emergent_offsets: set[int]) -> bool:
    h, w = grid.shape
    active_rows = [r for r in range(h) if r != row_separator]
    for j in _candidate_pair_offsets(w):
        if j in emergent_offsets:
            continue
        left = grid[:, j] == GREY
        right = grid[:, w - 1 - j] == GREY
        union = left | right
        if union.all():
            continue
        for col_mask in (left, right):
            count = int(col_mask[active_rows].sum())
            if count in {len(active_rows) - 1, len(active_rows) - 2}:
                return True
    return False


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 8 <= h <= 10 and 8 <= w <= 10

    board, specs = _materialize_separator_board(g)
    assert 1 <= len(specs) <= 3
    positions = [int(spec["position"]) for spec in specs]
    assert len(positions) == len(set(positions))

    literal_full_cols = np.where(np.all(_append_mirror(g) == GREY, axis=0))[0]
    assert literal_full_cols.size == 0, "generator should rely on emergent columns, not pre-existing full columns"

    full_rows_in_input = np.where(np.all(g == GREY, axis=1))[0]
    assert full_rows_in_input.size <= 1

    for spec in specs:
        j = int(spec["pair"])
        left = g[:, j] == GREY
        right = g[:, w - 1 - j] == GREY
        assert np.all(left | right)
        assert not left.all()
        assert not right.all()
        left_count = int(left.sum())
        right_count = int(right.sum())
        if spec["active_side"] == "L":
            assert left_count <= right_count
        else:
            assert right_count < left_count

    regions, sep_rows, sep_cols = _partition_regions(board)
    assert 3 <= len(regions) <= 6
    areas = [region["area"] for region in regions]
    assert len(set(areas)) == len(areas)
    assert sep_cols.sum() >= len(specs)
    assert _has_required_counterexample(
        g,
        int(full_rows_in_input[0]) if full_rows_in_input.size else None,
        {int(spec["pair"]) for spec in specs},
    )

    out = main(g)
    assert out.ndim == 2 and max(out.shape) <= 30
    assert np.all(out[sep_rows, :] == GREY)
    assert np.all(out[:, sep_cols] == GREY)

    ranked = sorted(regions, key=lambda region: (-region["area"], region["r0"], region["c0"]))
    for rank, region in enumerate(ranked):
        color = AREA_COLORS[rank]
        patch = out[region["r0"] : region["r1"], region["c0"] : region["c1"]]
        assert np.all(patch == color)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height, width, plans, row_separator = _sample_layout()
        grid = np.zeros((height, width), dtype=int)

        if row_separator is not None:
            grid[row_separator, :] = GREY

        used_offsets = {pair for pair, _active_side in plans}
        for offset, active_side in plans:
            _paint_emergent_pair(grid, offset, active_side, row_separator)

        available_counterexamples = [j for j in _candidate_pair_offsets(width) if j not in used_offsets]
        if not available_counterexamples:
            continue
        counterexample_offset = random.choice(available_counterexamples)
        _paint_counterexample_pair(grid, counterexample_offset, row_separator)

        blocked_cols = {j for j in used_offsets | {counterexample_offset}}
        blocked_cols |= {width - 1 - j for j in used_offsets | {counterexample_offset}}

        desired_reds = 1 + (row_separator is not None)
        _place_red_markers(grid, row_separator, blocked_cols, desired_reds)
        _place_grey_speckles(grid, row_separator, blocked_cols, max_count=2)

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid mirror-separator puzzle input")
