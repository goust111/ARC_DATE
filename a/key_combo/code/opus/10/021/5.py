from __future__ import annotations

import hashlib
import random

import numpy as np


BACKGROUND = 0
CONTROL_COLORS = (1, 2, 3, 4)
RAY_COLOR = 5
PIVOT_COLOR = 7

GRID_ROWS = (16, 22)
GRID_COLS = (16, 22)
FRAME_HEIGHT = (8, 15)
FRAME_WIDTH = (8, 15)
FRAME_MARGIN = 1
PIVOT_CLEARANCE = 2
GEN_RETRIES = 400
MIN_TOTAL_RAY_PIXELS = 6

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode(), usedforsecurity=False).hexdigest()


def _unique_extreme(
    controls: list[tuple[int, int, int]],
    axis: int,
    reducer,
    name: str,
) -> tuple[int, int, int]:
    value = reducer(ctrl[axis] for ctrl in controls)
    matches = [ctrl for ctrl in controls if ctrl[axis] == value]
    assert len(matches) == 1, f"expected a unique {name} control"
    return matches[0]


def _extract_structure(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    allowed = set(CONTROL_COLORS) | {BACKGROUND, PIVOT_COLOR}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"

    pivot_coords = np.argwhere(g == PIVOT_COLOR)
    assert len(pivot_coords) == 1, "expected exactly one orange pivot"
    pivot_r, pivot_c = map(int, pivot_coords[0])

    controls: list[tuple[int, int, int]] = []
    for color in CONTROL_COLORS:
        coords = np.argwhere(g == color)
        assert len(coords) == 1, f"expected exactly one control of color {color}"
        r, c = map(int, coords[0])
        controls.append((r, c, color))

    control_positions = {(r, c) for r, c, _ in controls}
    assert len(control_positions) == 4, "controls must occupy distinct cells"
    assert len({r for r, _, _ in controls}) == 4, "controls must not share rows"
    assert len({c for _, c, _ in controls}) == 4, "controls must not share columns"

    top = _unique_extreme(controls, 0, min, "top")
    bottom = _unique_extreme(controls, 0, max, "bottom")
    left = _unique_extreme(controls, 1, min, "left")
    right = _unique_extreme(controls, 1, max, "right")

    roles = {(r, c) for r, c, _ in (top, bottom, left, right)}
    assert len(roles) == 4, "one control cannot serve multiple extreme roles"

    r_top = top[0]
    r_bot = bottom[0]
    c_left = left[1]
    c_right = right[1]
    assert r_top < pivot_r < r_bot, "pivot must be strictly between top and bottom controls"
    assert c_left < pivot_c < c_right, "pivot must be strictly between left and right controls"

    return {
        "pivot": (pivot_r, pivot_c),
        "controls": controls,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "r_top": r_top,
        "r_bot": r_bot,
        "c_left": c_left,
        "c_right": c_right,
    }


def _ray_cells_until_blocked(
    canvas: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if int(canvas[r, c]) != BACKGROUND:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _expected_output_from_spec(grid: np.ndarray) -> np.ndarray:
    spec = _extract_structure(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r_top = int(spec["r_top"])
    r_bot = int(spec["r_bot"])
    c_left = int(spec["c_left"])
    c_right = int(spec["c_right"])
    pivot_r, pivot_c = spec["pivot"]
    top = spec["top"]
    bottom = spec["bottom"]
    left = spec["left"]
    right = spec["right"]

    out[r_top, c_left : c_right + 1] = top[2]
    out[r_bot, c_left : c_right + 1] = bottom[2]
    out[r_top + 1 : r_bot, c_left] = left[2]
    out[r_top + 1 : r_bot, c_right] = right[2]

    for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        for r, c in _ray_cells_until_blocked(out, (pivot_r, pivot_c), direction):
            out[r, c] = RAY_COLOR

    out[pivot_r, pivot_c] = PIVOT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the extreme-defined frame and stop four diagonal rays at that frame.

    Input structure:
    - The grid has a black background, exactly four single-pixel control markers in colors 1/2/3/4, and exactly one orange pivot pixel.
    - Among the controls, one is uniquely topmost, one uniquely bottommost, one uniquely leftmost, and one uniquely rightmost.
    - The pivot lies strictly inside the rectangle determined by those extreme rows and columns.

    Rule:
    - Use the top/bottom/left/right extreme controls to define a rectangular frame and color each side with that control's color; horizontal sides cover the corners.
    - From the orange pivot, cast one ray along each diagonal direction, repainting only black cells to grey until the next step would hit a non-black frame/control pixel.
    - Keep the pivot unchanged and leave everything outside the frame untouched.

    Key insight:
    - The controls matter only through their geometric extrema and side colors: they create the stopping boundary that determines the exact diagonal ray lengths.
    """
    return _expected_output_from_spec(np.asarray(grid, dtype=int))


def _valid_side_pairs(size: int, span_range: tuple[int, int]) -> list[tuple[int, int]]:
    lo, hi = span_range
    pairs = []
    for start in range(FRAME_MARGIN, size - FRAME_MARGIN):
        max_end = min(size - 1 - FRAME_MARGIN, start + hi - 1)
        min_end = start + lo - 1
        for end in range(min_end, max_end + 1):
            pairs.append((start, end))
    return pairs


def _corner_hit_count(pivot_r: int, pivot_c: int, r_top: int, r_bot: int, c_left: int, c_right: int) -> int:
    return sum(
        [
            pivot_r - r_top == pivot_c - c_left,
            pivot_r - r_top == c_right - pivot_c,
            r_bot - pivot_r == pivot_c - c_left,
            r_bot - pivot_r == c_right - pivot_c,
        ]
    )


def _ray_lengths(spec: dict[str, object]) -> tuple[int, int, int, int]:
    pivot_r, pivot_c = spec["pivot"]
    r_top = int(spec["r_top"])
    r_bot = int(spec["r_bot"])
    c_left = int(spec["c_left"])
    c_right = int(spec["c_right"])
    return (
        min(pivot_r - r_top, pivot_c - c_left) - 1,
        min(pivot_r - r_top, c_right - pivot_c) - 1,
        min(r_bot - pivot_r, pivot_c - c_left) - 1,
        min(r_bot - pivot_r, c_right - pivot_c) - 1,
    )


def _pivot_candidates(
    r_top: int,
    r_bot: int,
    c_left: int,
    c_right: int,
    mode: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for pivot_r in range(r_top + PIVOT_CLEARANCE, r_bot - PIVOT_CLEARANCE + 1):
        for pivot_c in range(c_left + PIVOT_CLEARANCE, c_right - PIVOT_CLEARANCE + 1):
            hit_count = _corner_hit_count(pivot_r, pivot_c, r_top, r_bot, c_left, c_right)
            if mode == 0 and hit_count != 0:
                continue
            if mode == 1 and hit_count != 1:
                continue
            if mode == 2 and hit_count < 2:
                continue
            candidates.append((pivot_r, pivot_c))
    random.shuffle(candidates)
    return candidates


def _validate_input_semantics(grid: np.ndarray) -> dict[str, object]:
    spec = _extract_structure(grid)
    pivot_r, pivot_c = spec["pivot"]
    r_top = int(spec["r_top"])
    r_bot = int(spec["r_bot"])
    c_left = int(spec["c_left"])
    c_right = int(spec["c_right"])

    assert pivot_r - r_top >= PIVOT_CLEARANCE
    assert r_bot - pivot_r >= PIVOT_CLEARANCE
    assert pivot_c - c_left >= PIVOT_CLEARANCE
    assert c_right - pivot_c >= PIVOT_CLEARANCE
    assert spec["top"][1] not in (c_left, c_right, pivot_c)
    assert spec["bottom"][1] not in (c_left, c_right, pivot_c)
    assert spec["left"][0] not in (r_top, r_bot, pivot_r)
    assert spec["right"][0] not in (r_top, r_bot, pivot_r)

    lengths = _ray_lengths(spec)
    assert min(lengths) >= 1, "each ray must have visible length"
    assert sum(lengths) >= MIN_TOTAL_RAY_PIXELS, "rays should not collapse into a trivial pattern"
    assert len(set(lengths)) >= 2, "ray lengths should not all be identical"
    return spec


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    spec = _validate_input_semantics(inp)
    r_top = int(spec["r_top"])
    r_bot = int(spec["r_bot"])
    c_left = int(spec["c_left"])
    c_right = int(spec["c_right"])
    pivot_r, pivot_c = spec["pivot"]
    top = spec["top"]
    bottom = spec["bottom"]
    left = spec["left"]
    right = spec["right"]

    assert out.shape == inp.shape
    assert int(out[pivot_r, pivot_c]) == PIVOT_COLOR
    assert np.all(out[r_top, c_left : c_right + 1] == top[2])
    assert np.all(out[r_bot, c_left : c_right + 1] == bottom[2])
    assert np.all(out[r_top + 1 : r_bot, c_left] == left[2])
    assert np.all(out[r_top + 1 : r_bot, c_right] == right[2])

    outside = np.ones(inp.shape, dtype=bool)
    outside[r_top : r_bot + 1, c_left : c_right + 1] = False
    assert np.array_equal(out[outside], inp[outside]), "outside the frame nothing should change"

    expected_ray_mask = np.zeros(inp.shape, dtype=bool)
    frame_then_rays = inp.copy()
    frame_then_rays[r_top, c_left : c_right + 1] = top[2]
    frame_then_rays[r_bot, c_left : c_right + 1] = bottom[2]
    frame_then_rays[r_top + 1 : r_bot, c_left] = left[2]
    frame_then_rays[r_top + 1 : r_bot, c_right] = right[2]
    for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ray = _ray_cells_until_blocked(frame_then_rays, (pivot_r, pivot_c), direction)
        assert ray, "each direction must contribute at least one ray pixel"
        for r, c in ray:
            expected_ray_mask[r, c] = True

    actual_ray_mask = out == RAY_COLOR
    assert np.array_equal(actual_ray_mask, expected_ray_mask), "grey pixels must be exactly the four diagonal rays"

    expected = _expected_output_from_spec(inp)
    assert np.array_equal(out, expected), "output must match the puzzle semantics exactly"


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)

    row_pairs = _valid_side_pairs(rows, FRAME_HEIGHT)
    col_pairs = _valid_side_pairs(cols, FRAME_WIDTH)
    if not row_pairs or not col_pairs:
        return None

    random.shuffle(row_pairs)
    random.shuffle(col_pairs)
    branch_modes = [0, 1, 2]
    random.shuffle(branch_modes)

    for r_top, r_bot in row_pairs[:80]:
        for c_left, c_right in col_pairs[:80]:
            if (r_bot - r_top) < 2 * PIVOT_CLEARANCE or (c_right - c_left) < 2 * PIVOT_CLEARANCE:
                continue
            for mode in branch_modes:
                pivots = _pivot_candidates(r_top, r_bot, c_left, c_right, mode)
                if not pivots:
                    continue
                pivot_r, pivot_c = pivots[0]

                top_choices = [c for c in range(c_left + 1, c_right) if c != pivot_c]
                side_choices = [r for r in range(r_top + 1, r_bot) if r != pivot_r]
                if len(top_choices) < 2 or len(side_choices) < 2:
                    continue

                top_c, bottom_c = random.sample(top_choices, 2)
                left_r, right_r = random.sample(side_choices, 2)
                top_color, bottom_color, left_color, right_color = random.sample(CONTROL_COLORS, 4)

                grid = np.zeros((rows, cols), dtype=int)
                grid[pivot_r, pivot_c] = PIVOT_COLOR
                grid[r_top, top_c] = top_color
                grid[r_bot, bottom_c] = bottom_color
                grid[left_r, c_left] = left_color
                grid[right_r, c_right] = right_color

                try:
                    out = main(grid)
                    _validate_output_semantics(grid, out)
                except AssertionError:
                    continue

                input_sig = _sig(grid)
                output_sig = _sig(out)
                if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
                    continue

                _SEEN_INPUT_SIGS.add(input_sig)
                _SEEN_OUTPUT_SIGS.add(output_sig)
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
