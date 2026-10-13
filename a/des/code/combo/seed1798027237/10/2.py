import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))
RECT_SIZE_MIN = 1
RECT_SIZE_MAX = 6
SAFE_MARGIN_MIN = 2
SAFE_MARGIN_EXTRA = 4
MAX_GENERATION_ATTEMPTS = 300


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    row_candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_candidates) == 1, "expected one horizontal grey separator"
    assert len(col_candidates) == 1, "expected one vertical grey separator"

    cross_r = int(row_candidates[0])
    cross_c = int(col_candidates[0])
    assert 0 < cross_r < grid.shape[0] - 1, "separator row must split the grid"
    assert 0 < cross_c < grid.shape[1] - 1, "separator column must split the grid"
    return cross_r, cross_c


def _quadrant_name(row: int, col: int, cross_r: int, cross_c: int) -> str:
    if row < cross_r and col < cross_c:
        return "tl"
    if row < cross_r and col > cross_c:
        return "tr"
    if row > cross_r and col < cross_c:
        return "bl"
    if row > cross_r and col > cross_c:
        return "br"
    raise AssertionError("seed may not lie on the grey cross")


def _decode_seeds(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]]]:
    cross_r, cross_c = _find_cross(grid)
    seed_positions = np.argwhere((grid != BLACK) & (grid != GREY))
    assert len(seed_positions) == 4, "expected one colored seed per quadrant"

    seeds: list[dict[str, int]] = []
    seen_quadrants: set[str] = set()
    seen_colors: set[int] = set()
    for row, col in seed_positions:
        r = int(row)
        c = int(col)
        quadrant = _quadrant_name(r, c, cross_r, cross_c)
        color = int(grid[r, c])
        assert quadrant not in seen_quadrants, "each quadrant must contain exactly one seed"
        assert color not in seen_colors, "seed colors must be distinct"
        seen_quadrants.add(quadrant)
        seen_colors.add(color)
        seeds.append(
            {
                "row": r,
                "col": c,
                "color": color,
                "vdir": -1 if r < cross_r else 1,
                "hdir": -1 if c < cross_c else 1,
            }
        )

    assert len(seen_quadrants) == 4, "missing a quadrant seed"
    return cross_r, cross_c, seeds


def _span_start(index: int, size: int, direction: int) -> int:
    if direction < 0:
        return index - size + 1
    return index


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_r, cross_c, seeds = _decode_seeds(g)
    out = np.full_like(g, BLACK)
    out[cross_r, :] = GREY
    out[:, cross_c] = GREY

    for seed in seeds:
        row_matches = [
            other for other in seeds if other["row"] == seed["row"] and other["col"] != seed["col"]
        ]
        col_matches = [
            other for other in seeds if other["col"] == seed["col"] and other["row"] != seed["row"]
        ]
        assert len(row_matches) == 1, "each seed must have one opposite seed in the same row"
        assert len(col_matches) == 1, "each seed must have one opposite seed in the same column"

        width = abs(seed["col"] - row_matches[0]["col"]) - 2
        height = abs(seed["row"] - col_matches[0]["row"]) - 2
        assert width >= 1 and height >= 1, "rectangle dimensions must be positive"

        r0 = _span_start(seed["row"], height, seed["vdir"])
        c0 = _span_start(seed["col"], width, seed["hdir"])
        r1 = r0 + height
        c1 = c0 + width

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(g.shape[0], r1)
        cc1 = min(g.shape[1], c1)
        if rr0 < rr1 and cc0 < cc1:
            out[rr0:rr1, cc0:cc1] = seed["color"]

    return out


def _sample_split(total: int) -> tuple[int, int]:
    first = random.randint(0, total)
    return first, total - first


def _sample_margin(required: int, clipped: bool) -> int:
    if clipped:
        assert required >= 1, "cannot force clipping when no extra reach is needed"
        return random.randint(0, required - 1)
    lower = max(required, SAFE_MARGIN_MIN)
    upper = required + SAFE_MARGIN_EXTRA
    return random.randint(lower, upper)


def _build_input(
    rect_h: int,
    rect_w: int,
    top_gap: int,
    bottom_gap: int,
    left_gap: int,
    right_gap: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    height = top_margin + 1 + top_gap + 1 + bottom_gap + 1 + bottom_margin
    width = left_margin + 1 + left_gap + 1 + right_gap + 1 + right_margin

    cross_r = top_margin + 1 + top_gap
    cross_c = left_margin + 1 + left_gap
    top_row = top_margin
    bottom_row = cross_r + 1 + bottom_gap
    left_col = left_margin
    right_col = cross_c + 1 + right_gap

    grid = np.full((height, width), BLACK, dtype=int)
    grid[cross_r, :] = GREY
    grid[:, cross_c] = GREY

    colors = random.sample(SEED_COLORS, 4)
    placements = [
        (top_row, left_col, colors[0]),
        (top_row, right_col, colors[1]),
        (bottom_row, left_col, colors[2]),
        (bottom_row, right_col, colors[3]),
    ]
    for row, col, color in placements:
        grid[row, col] = color

    assert rect_h == top_gap + bottom_gap
    assert rect_w == left_gap + right_gap
    return grid


def _gen_attempt() -> np.ndarray:
    rect_h = random.randint(RECT_SIZE_MIN, RECT_SIZE_MAX)
    rect_w = random.randint(RECT_SIZE_MIN, RECT_SIZE_MAX)
    top_gap, bottom_gap = _sample_split(rect_h)
    left_gap, right_gap = _sample_split(rect_w)

    eligible_clip_sides = []
    if rect_h > 1:
        eligible_clip_sides.extend(["top", "bottom"])
    if rect_w > 1:
        eligible_clip_sides.extend(["left", "right"])

    want_clip = bool(eligible_clip_sides) and random.random() < 0.5
    clip_sides: set[str] = set()
    if want_clip:
        clip_count = random.randint(1, min(2, len(eligible_clip_sides)))
        clip_sides = set(random.sample(eligible_clip_sides, clip_count))

    top_margin = _sample_margin(rect_h - 1, "top" in clip_sides)
    bottom_margin = _sample_margin(rect_h - 1, "bottom" in clip_sides)
    left_margin = _sample_margin(rect_w - 1, "left" in clip_sides)
    right_margin = _sample_margin(rect_w - 1, "right" in clip_sides)

    grid = _build_input(
        rect_h,
        rect_w,
        top_gap,
        bottom_gap,
        left_gap,
        right_gap,
        top_margin,
        bottom_margin,
        left_margin,
        right_margin,
    )

    actual_clip = (
        top_margin < rect_h - 1
        or bottom_margin < rect_h - 1
        or left_margin < rect_w - 1
        or right_margin < rect_w - 1
    )
    assert actual_clip == want_clip

    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
