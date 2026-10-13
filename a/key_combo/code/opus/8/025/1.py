import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

PALETTE = np.array([BLUE, RED, GREEN, YELLOW, PINK, ORANGE], dtype=int)
RNG = np.random.default_rng()

GEN_RETRIES = 400
CASE_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_row_run(row: np.ndarray) -> tuple[int, int, int]:
    cols = np.flatnonzero((row != BLACK) & (row != GREY))
    assert cols.size > 0, "each active row must contain one colored run"
    start = int(cols[0])
    end = int(cols[-1])
    assert cols.size == end - start + 1, "the colored cells in a row must be contiguous"
    color = int(row[start])
    assert color not in (BLACK, GREY)
    assert np.all(row[start : end + 1] == color), "a run must be monochrome"
    assert np.all(row[:start] == BLACK), "left padding must be black"
    assert np.all(row[end + 1 :] == BLACK), "right padding must be black"
    return start, end, color


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_box = _bbox(g == GREY)
    assert frame_box is not None, "expected one grey container"
    r0, c0, r1, c1 = frame_box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "container must have a non-empty interior"

    expected_frame = np.zeros_like(g, dtype=bool)
    expected_frame[r0, c0 : c1 + 1] = True
    expected_frame[r1, c0 : c1 + 1] = True
    expected_frame[r0 : r1 + 1, c0] = True
    expected_frame[r0 : r1 + 1, c1] = True
    assert np.array_equal(g == GREY, expected_frame), "grey pixels must form a single rectangular frame"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "container interior must start empty"

    gap_col = c0 - 1
    assert gap_col >= 1, "need at least one column between the left area and the container"
    assert np.all(g[:, gap_col] == BLACK), "the separator column must be black"

    active_rows = list(range(r0 + 1, r1))
    assert active_rows, "container interior must have height"

    left_rows = g[:, :gap_col]
    inactive = [r for r in range(g.shape[0]) if r not in active_rows]
    if inactive:
        assert np.all(left_rows[inactive] == BLACK), "inactive rows in the left area must stay black"

    rows = []
    for r in active_rows:
        start, end, color = _parse_row_run(g[r, :gap_col])
        rows.append(
            {
                "row": int(r),
                "start": int(start),
                "end": int(end),
                "width": int(end - start + 1),
                "color": int(color),
            }
        )

    return {
        "grid": g,
        "frame_box": frame_box,
        "gap_col": int(gap_col),
        "rows": rows,
        "inner_left": int(c0 + 1),
        "inner_width": int(c1 - c0 - 1),
    }


def _compose_output(info: dict, aligned_starts: list[int], stripe_widths: list[int]) -> np.ndarray:
    g = info["grid"]
    r0, c0, r1, c1 = info["frame_box"]
    out = g.copy()
    out[r0 + 1 : r1, : info["gap_col"]] = BLACK
    out[r0 + 1 : r1, c0 + 1 : c1] = BLACK

    for row_info, new_start, stripe_width in zip(info["rows"], aligned_starts, stripe_widths):
        r = row_info["row"]
        width = row_info["width"]
        color = row_info["color"]
        new_start = int(new_start)
        stripe_width = int(stripe_width)
        assert 0 <= new_start <= info["gap_col"] - width
        assert 0 <= stripe_width <= info["inner_width"]
        out[r, new_start : new_start + width] = color
        if stripe_width:
            out[r, info["inner_left"] : info["inner_left"] + stripe_width] = color

    return out


def _expected_output(info: dict) -> np.ndarray:
    anchor_end = min(row["end"] for row in info["rows"])
    aligned_starts = [max(0, anchor_end - row["width"] + 1) for row in info["rows"]]
    stripe_widths = [row["end"] - anchor_end for row in info["rows"]]
    return _compose_output(info, aligned_starts, stripe_widths)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each left-side color run so all original right edges align, then encode each row's shift amount as a stripe inside the grey container.

    Input structure:
    - A black background contains one grey one-cell-thick rectangular frame on the right with an all-black interior.
    - One black separator column sits immediately to the left of that frame.
    - In each row aligned with the frame interior, the left area contains exactly one contiguous non-black run and black elsewhere.

    Rule:
    - Read every row's run width, color, and original right edge in the left area.
    - Find the minimum original right edge and shift every run left so its right edge lands there; if that would push the run past column 0, start it at column 0 instead.
    - Erase the vacated cells back to black.
    - In the container interior, fill each row from the left inner wall using the same row color for a stripe whose width equals original_right_edge minus the minimum right edge.
    - Keep the grey frame and all untouched black cells unchanged.

    Key insight:
    - The alignment step produces one numeric by-product per row, and that same shift distance is reused as the stripe width, so the output depends on both moving the run and counting how far it moved.
    """
    info = _parse_grid(grid)
    return _expected_output(info)


def _make_input(
    n_rows: int,
    left_width: int,
    lengths: np.ndarray,
    ends: np.ndarray,
    colors: np.ndarray,
    inner_width: int,
    right_pad: int,
) -> np.ndarray:
    height = n_rows + 2
    frame_c0 = left_width + 1
    frame_c1 = frame_c0 + inner_width + 1
    width = frame_c1 + 1 + right_pad
    grid = np.zeros((height, width), dtype=int)
    grid[:, frame_c0] = GREY
    grid[:, frame_c1] = GREY
    grid[0, frame_c0 : frame_c1 + 1] = GREY
    grid[-1, frame_c0 : frame_c1 + 1] = GREY

    starts = ends - lengths + 1
    for idx in range(n_rows):
        r = idx + 1
        grid[r, starts[idx] : ends[idx] + 1] = int(colors[idx])
    return grid


def _row_vectors(info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.array([row["start"] for row in info["rows"]], dtype=int)
    ends = np.array([row["end"] for row in info["rows"]], dtype=int)
    widths = np.array([row["width"] for row in info["rows"]], dtype=int)
    return starts, ends, widths


def _anti_shortcut_checks(info: dict, expected: np.ndarray) -> bool:
    starts, ends, widths = _row_vectors(info)
    anchor_end = int(ends.min())
    shifts = ends - anchor_end
    correct_starts = [max(0, anchor_end - int(width) + 1) for width in widths]

    start_anchor = int(starts.min())
    left_aligned = _compose_output(
        info,
        [start_anchor] * len(widths),
        list(starts - start_anchor),
    )
    if np.array_equal(left_aligned, expected):
        return False

    length_stripes = _compose_output(info, correct_starts, list(widths))
    if np.array_equal(length_stripes, expected):
        return False

    if np.count_nonzero(shifts) < 2 or int(shifts.max()) < 2:
        return False
    if np.unique(widths).size < 3:
        return False
    if np.all(shifts == widths):
        return False
    if np.all(shifts == (starts - start_anchor)):
        return False
    return True


def _valid_generated_case(grid: np.ndarray) -> bool:
    try:
        info = _parse_grid(grid)
        starts, ends, widths = _row_vectors(info)
        colors = [row["color"] for row in info["rows"]]
        shifts = ends - int(ends.min())

        if len(colors) != len(set(colors)):
            return False
        if np.unique(ends).size != len(ends):
            return False
        if np.any(widths < 3) or np.any(widths > 7):
            return False
        if int(ends.min()) < int(widths.max()) - 1:
            return False
        if int(shifts.max()) > info["inner_width"]:
            return False
        if not np.array_equal(main(grid), _expected_output(info)):
            return False
        if not _anti_shortcut_checks(info, _expected_output(info)):
            return False
        if np.unique(starts).size == 1:
            return False
        return True
    except AssertionError:
        return False


def _sample_case() -> np.ndarray | None:
    for _ in range(CASE_RETRIES):
        n_rows = int(RNG.integers(4, 7))
        colors = RNG.permutation(PALETTE)[:n_rows]

        lengths = RNG.integers(3, 8, size=n_rows)
        if np.unique(lengths).size < 3:
            continue

        min_left_width = max(9, int(lengths.max()) + n_rows - 1)
        if min_left_width > 14:
            continue
        left_width = int(RNG.integers(min_left_width, 15))

        end_choices = np.arange(int(lengths.max()) - 1, left_width, dtype=int)
        if end_choices.size < n_rows:
            continue
        ends = RNG.choice(end_choices, size=n_rows, replace=False)
        shifts = ends - int(ends.min())
        starts = ends - lengths + 1

        if np.any(starts < 0):
            continue
        if np.unique(ends).size != n_rows:
            continue
        if np.count_nonzero(shifts) < 2 or int(shifts.max()) < 2:
            continue
        if np.all(shifts == lengths):
            continue
        if np.all(shifts == (starts - int(starts.min()))):
            continue

        inner_width = int(shifts.max()) + int(RNG.integers(0, 4))
        right_pad = int(RNG.integers(0, 3))
        grid = _make_input(n_rows, left_width, lengths, ends, colors, inner_width, right_pad)
        if max(grid.shape) > 30:
            continue
        if _valid_generated_case(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
