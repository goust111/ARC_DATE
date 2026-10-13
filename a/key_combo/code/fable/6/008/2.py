from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, coords_where, get_bg


RED = Color.RED
PALETTE = tuple(range(10))
GRID_SIZES = tuple(range(12, 29, 4))
MIN_DECOYS = 3
MAX_DECOYS = 7
MAX_GENERATION_ATTEMPTS = 400


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _quadrant_of(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    return (0 if int(r) < h // 2 else 1, 0 if int(c) < w // 2 else 1)


def _subquadrant_bounds(
    shape: tuple[int, int], quad_r: int, quad_c: int, sub_r: int, sub_c: int
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    half_h, half_w = h // 2, w // 2
    quarter_h, quarter_w = h // 4, w // 4
    r0 = int(quad_r) * half_h + int(sub_r) * quarter_h
    c0 = int(quad_c) * half_w + int(sub_c) * quarter_w
    return r0, r0 + quarter_h, c0, c0 + quarter_w


def _subquadrant_of(
    shape: tuple[int, int], quad_r: int, quad_c: int, r: int, c: int
) -> tuple[int, int]:
    r0, r1, c0, c1 = _subquadrant_bounds(shape, quad_r, quad_c, 0, 0)
    del r1, c1
    quarter_h, quarter_w = shape[0] // 4, shape[1] // 4
    sub_r = 0 if int(r) < r0 + quarter_h else 1
    sub_c = 0 if int(c) < c0 + quarter_w else 1
    return sub_r, sub_c


def _cells_in_subquadrant(
    shape: tuple[int, int], quad_r: int, quad_c: int, sub_r: int, sub_c: int
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _subquadrant_bounds(shape, quad_r, quad_c, sub_r, sub_c)
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]


def _render_output(
    shape: tuple[int, int], bg: int, red_quad: tuple[int, int], rel_sub: tuple[int, int], fill_color: int
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, r1, c0, c1 = _subquadrant_bounds(shape, red_quad[0], red_quad[1], rel_sub[0], rel_sub[1])
    out[r0:r1, c0:c1] = int(fill_color)
    return out


def _extract_features(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape
    assert h % 4 == 0 and w % 4 == 0, "grid must support two equal halving steps"

    bg = int(get_bg(g))
    red_coords = coords_where(g == RED)
    assert len(red_coords) == 1, "expected exactly one red control pixel"
    red = tuple(map(int, red_coords[0]))

    fg = [tuple(map(int, coord)) for coord in coords_where(g != bg)]
    assert len(fg) >= 1 + MIN_DECOYS + 1, "need one red pixel, one nearest pixel, and decoys"
    for i, a in enumerate(fg):
        for b in fg[i + 1 :]:
            assert _chebyshev(a, b) > 1, "foreground singletons must be mutually non-adjacent"

    others = []
    for coord in fg:
        if coord == red:
            continue
        color = int(g[coord])
        assert color != RED, "only the control pixel may be red"
        others.append(
            {
                "coord": coord,
                "color": color,
                "distance": _chebyshev(coord, red),
            }
        )

    assert others, "expected at least one non-red singleton"
    min_dist = min(obj["distance"] for obj in others)
    nearest = [obj for obj in others if obj["distance"] == min_dist]
    assert len(nearest) == 1, "nearest non-red singleton must be unique"
    nearest_obj = nearest[0]

    red_quad = _quadrant_of(g.shape, red[0], red[1])
    red_sub = _subquadrant_of(g.shape, red_quad[0], red_quad[1], red[0], red[1])
    nearest_quad = _quadrant_of(g.shape, nearest_obj["coord"][0], nearest_obj["coord"][1])
    nearest_sub = _subquadrant_of(
        g.shape,
        nearest_quad[0],
        nearest_quad[1],
        nearest_obj["coord"][0],
        nearest_obj["coord"][1],
    )
    decoys = [obj for obj in others if obj is not nearest_obj]
    for obj in decoys:
        quad = _quadrant_of(g.shape, obj["coord"][0], obj["coord"][1])
        obj["quadrant"] = quad
        obj["subquadrant"] = _subquadrant_of(g.shape, quad[0], quad[1], obj["coord"][0], obj["coord"][1])

    return {
        "grid": g,
        "shape": g.shape,
        "bg": bg,
        "red": red,
        "red_quad": red_quad,
        "red_sub": red_sub,
        "nearest": {
            **nearest_obj,
            "quadrant": nearest_quad,
            "subquadrant": nearest_sub,
        },
        "decoys": decoys,
    }


def _semantic_checks(grid: np.ndarray) -> None:
    features = _extract_features(grid)
    nearest = features["nearest"]
    decoys = features["decoys"]

    assert nearest["quadrant"] != features["red_quad"], "red coarse quadrant must matter"
    assert nearest["subquadrant"] != features["red_sub"], "nearest pixel pose must matter"
    assert decoys, "need distractors beyond the nearest singleton"

    out = main(grid)
    expected = _render_output(
        features["shape"],
        features["bg"],
        features["red_quad"],
        nearest["subquadrant"],
        nearest["color"],
    )
    assert np.array_equal(out, expected), "main() must match the intended rule"

    red_only = _render_output(
        features["shape"], features["bg"], features["red_quad"], features["red_sub"], nearest["color"]
    )
    assert not np.array_equal(out, red_only), "red pixel's own subquadrant shortcut must fail"

    donor_owned = _render_output(
        features["shape"], features["bg"], nearest["quadrant"], nearest["subquadrant"], nearest["color"]
    )
    assert not np.array_equal(out, donor_owned), "donor quadrant shortcut must fail"

    runner_up = min(decoys, key=lambda obj: obj["distance"])
    alt = _render_output(
        features["shape"],
        features["bg"],
        features["red_quad"],
        runner_up["subquadrant"],
        runner_up["color"],
    )
    assert not np.array_equal(out, alt), "nearest-neighbor choice must affect the output"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill one recursive subquadrant inside the red pixel's quadrant using the nearest distractor's pose and color.

    Input structure:
    - The grid has even-halvable dimensions, a dominant background, exactly one red singleton, and several other isolated non-red singletons.
    - The non-red singletons are mutually non-adjacent, and exactly one of them is strictly closest to the red pixel by Chebyshev distance.

    Rule:
    - Split the whole grid into four equal quadrants and locate the quadrant Q that contains the red pixel.
    - Find the unique nearest non-red singleton p; inside p's own quadrant, read which relative subquadrant (upper-left, upper-right, lower-left, lower-right) contains p.
    - Clear the canvas to background, then fill the matching relative subquadrant inside Q with p's color.

    Key insight:
    - The red pixel gives only the coarse destination quadrant; the nearest distractor simultaneously donates the fine subquadrant and the fill color.
    """
    features = _extract_features(np.asarray(grid, dtype=int))
    nearest = features["nearest"]
    return _render_output(
        features["shape"],
        features["bg"],
        features["red_quad"],
        nearest["subquadrant"],
        nearest["color"],
    )


def _pick_red_and_donor(
    rng: random.Random,
    shape: tuple[int, int],
    red_quad: tuple[int, int],
    red_sub: tuple[int, int],
    donor_quad: tuple[int, int],
    donor_sub: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    red_cells = _cells_in_subquadrant(shape, red_quad[0], red_quad[1], red_sub[0], red_sub[1])
    donor_cells = _cells_in_subquadrant(shape, donor_quad[0], donor_quad[1], donor_sub[0], donor_sub[1])

    pairs = []
    for red_cell in red_cells:
        for donor_cell in donor_cells:
            dist = _chebyshev(red_cell, donor_cell)
            if dist > 1:
                pairs.append((dist, red_cell, donor_cell))
    assert pairs, "selected subquadrants must admit separated singleton positions"

    pairs.sort(key=lambda item: item[0])
    short = pairs[: min(len(pairs), 10)]
    best_dist = short[0][0]
    weighted = [item for item in short if item[0] <= best_dist + 1]
    _dist, red_cell, donor_cell = rng.choice(weighted)
    return red_cell, donor_cell


def _candidate_decoy_cells(
    shape: tuple[int, int], occupied: set[tuple[int, int]], red: tuple[int, int], min_distance: int
) -> list[tuple[int, int]]:
    candidates = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            cell = (r, c)
            if cell in occupied:
                continue
            if _chebyshev(cell, red) <= min_distance:
                continue
            if any(_chebyshev(cell, other) <= 1 for other in occupied):
                continue
            candidates.append(cell)
    return candidates


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice(GRID_SIZES)
    w = rng.choice(GRID_SIZES)
    shape = (h, w)

    bg = rng.choice([color for color in PALETTE if color != RED])
    donor_color = rng.choice([color for color in PALETTE if color not in (bg, RED)])
    decoy_palette = [color for color in PALETTE if color not in (bg, RED, donor_color)]

    red_quad = (rng.randrange(2), rng.randrange(2))
    donor_quad_choices = [quad for quad in ((0, 0), (0, 1), (1, 0), (1, 1)) if quad != red_quad]
    donor_quad = rng.choice(donor_quad_choices)

    donor_sub = (rng.randrange(2), rng.randrange(2))
    red_sub_choices = [sub for sub in ((0, 0), (0, 1), (1, 0), (1, 1)) if sub != donor_sub]
    red_sub = rng.choice(red_sub_choices)

    red, donor = _pick_red_and_donor(rng, shape, red_quad, red_sub, donor_quad, donor_sub)
    grid = np.full(shape, bg, dtype=int)
    grid[red] = RED
    grid[donor] = donor_color

    donor_distance = _chebyshev(red, donor)
    occupied = {red, donor}
    candidate_cells = _candidate_decoy_cells(shape, occupied, red, donor_distance)
    if len(candidate_cells) < MIN_DECOYS:
        return None

    def score(cell: tuple[int, int]) -> tuple[int, int, int]:
        distance = _chebyshev(cell, red)
        quad = _quadrant_of(shape, cell[0], cell[1])
        sub = _subquadrant_of(shape, quad[0], quad[1], cell[0], cell[1])
        mismatch = int(sub != donor_sub)
        return (distance, mismatch, (cell[0] * 31 + cell[1]) % 17)

    candidate_cells.sort(key=score)
    runner_up = candidate_cells[0]
    grid[runner_up] = decoy_palette[0]
    occupied.add(runner_up)

    remaining = _candidate_decoy_cells(shape, occupied, red, donor_distance)
    rng.shuffle(remaining)
    target_count = rng.randint(MIN_DECOYS, MAX_DECOYS)
    decoy_cells = [runner_up]
    for cell in remaining:
        if len(decoy_cells) >= target_count:
            break
        if any(_chebyshev(cell, other) <= 1 for other in occupied):
            continue
        decoy_cells.append(cell)
        occupied.add(cell)

    if len(decoy_cells) < MIN_DECOYS:
        return None

    for idx, cell in enumerate(decoy_cells[1:], start=1):
        color = decoy_palette[idx % len(decoy_palette)]
        grid[cell] = color

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recursive-quadrant sample")
