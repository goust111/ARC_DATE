from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

TOP_MARKER = GREEN
TOP_TARGET = YELLOW
BOTTOM_MARKER = BLUE
BOTTOM_TARGET = RED

TOP_DISTANCE_OPTIONS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _sign(delta: int) -> int:
    return int((delta > 0) - (delta < 0))


def _separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert rows.size == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _parse_input(grid: np.ndarray) -> dict[str, int | tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _separator_row(g)
    top_marker = _single_coord(g, TOP_MARKER)
    top_target = _single_coord(g, TOP_TARGET)
    bottom_marker = _single_coord(g, BOTTOM_MARKER)
    bottom_target = _single_coord(g, BOTTOM_TARGET)

    tmr, tmc = top_marker
    tyr, tyc = top_target
    bmr, bmc = bottom_marker
    btr, btc = bottom_target

    assert tmr == tyr, "green/yellow pair must be horizontally aligned"
    assert bmc == btc, "blue/red pair must be vertically aligned"
    assert tmr < sep and tyr < sep, "top pair must lie above separator"
    assert bmr > sep and btr > sep, "bottom pair must lie below separator"

    top_distance = abs(tyc - tmc)
    bottom_distance = abs(btr - bmr)
    assert top_distance >= 2 and bottom_distance >= 2

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, RED, GREEN, YELLOW, GREY}
    expected_counts = {
        TOP_MARKER: 1,
        TOP_TARGET: 1,
        BOTTOM_MARKER: 1,
        BOTTOM_TARGET: 1,
    }
    for color, count in expected_counts.items():
        assert int(np.sum(g == int(color))) == count
    assert int(np.sum(g == GREY)) == g.shape[1]

    return {
        "separator": sep,
        "top_marker": top_marker,
        "top_target": top_target,
        "bottom_marker": bottom_marker,
        "bottom_target": bottom_target,
        "top_distance": top_distance,
        "bottom_distance": bottom_distance,
    }


def _paint_horizontal_half(
    out: np.ndarray,
    row_start: int,
    row_end: int,
    anchor_row: int,
    period: int,
    marker_color: int,
    target_color: int,
) -> None:
    for r in range(int(row_start), int(row_end) + 1):
        offset = r - int(anchor_row)
        if offset % int(period) != 0:
            continue
        color = int(marker_color) if (abs(offset) // int(period)) % 2 == 0 else int(target_color)
        mask = out[r] == BLACK
        out[r, mask] = color


def _paint_vertical_half(
    out: np.ndarray,
    row_start: int,
    row_end: int,
    anchor_col: int,
    period: int,
    marker_color: int,
    target_color: int,
) -> None:
    for c in range(out.shape[1]):
        offset = c - int(anchor_col)
        if offset % int(period) != 0:
            continue
        color = int(marker_color) if (abs(offset) // int(period)) % 2 == 0 else int(target_color)
        mask = out[row_start : row_end + 1, c] == BLACK
        out[row_start : row_end + 1, c][mask] = color


def _slide_path(marker: tuple[int, int], target: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    mr, mc = marker
    tr, tc = target
    dr = _sign(tr - mr)
    dc = _sign(tc - mc)
    assert (dr, dc) in {(0, 1), (0, -1), (1, 0), (-1, 0)}

    stop_r = tr - dr
    stop_c = tc - dc
    path: list[tuple[int, int]] = []
    r, c = mr, mc
    while True:
        path.append((int(r), int(c)))
        if (r, c) == (stop_r, stop_c):
            break
        r += dr
        c += dc
    return path, len(path)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marker toward its target, then use the trajectory row or column as the seed for periodic stripes.

    Input structure:
    - A black background contains exactly two aligned marker-target pairs separated by one full grey horizontal line.
    - Above the grey line, one green marker and one yellow target lie on the same row.
    - Below the grey line, one blue marker and one red target lie on the same column.
    - Each pair is separated by several empty cells, and the two pair distances are different.

    Rule:
    - For each pair, slide the marker along its shared row or column until it stops one cell before the target; color every visited cell of that trajectory with the marker color while leaving the target fixed.
    - Let D be the number of cells on that trajectory, including the original marker cell and the stopping cell.
    - In the pair's own half of the grid, repeat full stripes every D cells along the perpendicular axis: the stripe through the trajectory row or column uses the marker color, the next stripe D cells away uses the target color, then they alternate.
    - Stripes recolor only black background cells; the grey separator and both target pixels stay unchanged, and trajectory cells keep the marker color if a stripe overlaps them.

    Key insight:
    - The aligned marker-target pair supplies a slide direction, but the stripe period comes from the marker's actual stopping trajectory length rather than from an arbitrary fixed tiling rule.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse_input(g)
    sep = int(info["separator"])

    top_marker = info["top_marker"]
    top_target = info["top_target"]
    bottom_marker = info["bottom_marker"]
    bottom_target = info["bottom_target"]

    out = g.copy()

    top_path, top_period = _slide_path(top_marker, top_target)
    for r, c in top_path:
        out[r, c] = TOP_MARKER

    bottom_path, bottom_period = _slide_path(bottom_marker, bottom_target)
    for r, c in bottom_path:
        out[r, c] = BOTTOM_MARKER

    _paint_horizontal_half(
        out,
        row_start=0,
        row_end=sep - 1,
        anchor_row=int(top_marker[0]),
        period=top_period,
        marker_color=TOP_MARKER,
        target_color=TOP_TARGET,
    )
    _paint_vertical_half(
        out,
        row_start=sep + 1,
        row_end=out.shape[0] - 1,
        anchor_col=int(bottom_marker[1]),
        period=bottom_period,
        marker_color=BOTTOM_MARKER,
        target_color=BOTTOM_TARGET,
    )

    tr, tc = top_target
    br, bc = bottom_target
    out[tr, tc] = TOP_TARGET
    out[br, bc] = BOTTOM_TARGET
    out[sep, :] = GREY
    return out


def _stripe_positions(length: int, anchor: int, period: int) -> list[int]:
    return [idx for idx in range(int(length)) if (idx - int(anchor)) % int(period) == 0]


def _candidate_anchors(length: int, period: int, min_count: int = 3) -> list[int]:
    return [
        anchor
        for anchor in range(int(length))
        if len(_stripe_positions(length, anchor, period)) >= int(min_count)
    ]


def _choose_top_columns(width: int, distance: int) -> tuple[int, int]:
    marker_left = random.choice([True, False])
    if marker_left:
        marker_c = random.randint(0, int(width) - int(distance) - 1)
        target_c = marker_c + int(distance)
    else:
        marker_c = random.randint(int(distance), int(width) - 1)
        target_c = marker_c - int(distance)
    return int(marker_c), int(target_c)


def _choose_bottom_rows(height: int, distance: int) -> tuple[int, int]:
    marker_above = random.choice([True, False])
    if marker_above:
        marker_r = random.randint(0, int(height) - int(distance) - 1)
        target_r = marker_r + int(distance)
    else:
        marker_r = random.randint(int(distance), int(height) - 1)
        target_r = marker_r - int(distance)
    return int(marker_r), int(target_r)


def _simulate_with_counterfactual_periods(grid: np.ndarray, top_period: int, bottom_period: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _parse_input(g)
    sep = int(info["separator"])
    out = g.copy()

    top_path, _ = _slide_path(info["top_marker"], info["top_target"])
    bottom_path, _ = _slide_path(info["bottom_marker"], info["bottom_target"])
    for r, c in top_path:
        out[r, c] = TOP_MARKER
    for r, c in bottom_path:
        out[r, c] = BOTTOM_MARKER

    _paint_horizontal_half(out, 0, sep - 1, int(info["top_marker"][0]), int(top_period), TOP_MARKER, TOP_TARGET)
    _paint_vertical_half(
        out,
        sep + 1,
        out.shape[0] - 1,
        int(info["bottom_marker"][1]),
        int(bottom_period),
        BOTTOM_MARKER,
        BOTTOM_TARGET,
    )
    tr, tc = info["top_target"]
    br, bc = info["bottom_target"]
    out[tr, tc] = TOP_TARGET
    out[br, bc] = BOTTOM_TARGET
    out[sep, :] = GREY
    return out


def _validate_output_properties(inp: np.ndarray, out: np.ndarray) -> None:
    info = _parse_input(inp)
    sep = int(info["separator"])

    top_marker = info["top_marker"]
    top_target = info["top_target"]
    bottom_marker = info["bottom_marker"]
    bottom_target = info["bottom_target"]
    top_period = int(info["top_distance"])
    bottom_period = int(info["bottom_distance"])

    top_path, calc_top_period = _slide_path(top_marker, top_target)
    bottom_path, calc_bottom_period = _slide_path(bottom_marker, bottom_target)
    assert calc_top_period == top_period
    assert calc_bottom_period == bottom_period

    assert np.all(out[sep, :] == GREY)
    assert out[top_target] == TOP_TARGET
    assert out[bottom_target] == BOTTOM_TARGET

    for r, c in top_path:
        assert out[r, c] == TOP_MARKER
    for r, c in bottom_path:
        assert out[r, c] == BOTTOM_MARKER

    top_stripes = _stripe_positions(sep, int(top_marker[0]), top_period)
    bottom_stripes = _stripe_positions(inp.shape[1], int(bottom_marker[1]), bottom_period)
    assert len(top_stripes) >= 3
    assert len(bottom_stripes) >= 3
    assert any(abs(r - int(top_marker[0])) == top_period for r in top_stripes)
    assert any(abs(c - int(bottom_marker[1])) == bottom_period for c in bottom_stripes)

    expected = main(inp)
    assert np.array_equal(out, expected)

    wrong_top = _simulate_with_counterfactual_periods(inp, top_period - 1, bottom_period)
    wrong_bottom = _simulate_with_counterfactual_periods(inp, top_period, bottom_period - 1)
    assert not np.array_equal(out, wrong_top)
    assert not np.array_equal(out, wrong_bottom)

    assert np.any(out[:sep] == BLACK), "top half should retain some background"
    assert np.any(out[sep + 1 :] == BLACK), "bottom half should retain some background"


def _gen_attempt() -> np.ndarray | None:
    top_distance, bottom_distance = random.sample(TOP_DISTANCE_OPTIONS, 2)

    min_width = max(2 * bottom_distance + 1, top_distance + 3, 11)
    width = random.randint(min_width, min(18, 30))

    top_half_h = random.randint(max(2 * top_distance + 1, 7), 12)
    bottom_half_h = random.randint(max(bottom_distance + 3, 7), 12)
    if top_half_h + 1 + bottom_half_h > 30:
        return None

    top_anchor_choices = _candidate_anchors(top_half_h, top_distance, min_count=3)
    bottom_anchor_choices = _candidate_anchors(width, bottom_distance, min_count=3)
    if not top_anchor_choices or not bottom_anchor_choices:
        return None

    top_row = random.choice(top_anchor_choices)
    bottom_col = random.choice(bottom_anchor_choices)

    top_marker_c, top_target_c = _choose_top_columns(width, top_distance)
    bottom_marker_local_r, bottom_target_local_r = _choose_bottom_rows(bottom_half_h, bottom_distance)

    grid = np.zeros((top_half_h + 1 + bottom_half_h, width), dtype=int)
    sep = top_half_h
    grid[sep, :] = GREY

    grid[top_row, top_marker_c] = TOP_MARKER
    grid[top_row, top_target_c] = TOP_TARGET

    bottom_marker_r = sep + 1 + bottom_marker_local_r
    bottom_target_r = sep + 1 + bottom_target_local_r
    grid[bottom_marker_r, bottom_col] = BOTTOM_MARKER
    grid[bottom_target_r, bottom_col] = BOTTOM_TARGET

    try:
        info = _parse_input(grid)
        assert int(info["top_distance"]) != int(info["bottom_distance"])
        out = main(grid)
        _validate_output_properties(grid, out)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
