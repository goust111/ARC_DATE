from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, in_bounds


BG = Color.BLACK
TRACE = Color.RED
SEED_COLORS = [
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

GRID_SIDE_RANGE = (16, 30)
REAL_SQUARE_COUNT_RANGE = (1, 4)
REAL_SIDE_RANGE = (3, 9)
REAL_PADDING = 2
PARTIAL_PADDING = 1
PARTIAL_SIDE_RANGE = (3, 8)
PARTIAL_COUNT_RANGE = (1, 3)
PARTIAL_TRACE_COUNT_RANGE = (5, 7)
SINGLE_NOISE_RANGE = (2, 8)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _trace_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _interior_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _expanded_box(
    r0: int, c0: int, r1: int, c1: int, pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _mark_reserved(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _square_candidate(
    grid: np.ndarray,
    seed_lookup: dict[tuple[int, int], int],
    red_set: set[tuple[int, int]],
    r0: int,
    c0: int,
    side: int,
) -> dict | None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    traces = _trace_coords(r0, c0, r1, c1)
    if not all(in_bounds(grid.shape, r, c) for r, c in traces):
        return None
    if not all((r, c) in red_set for r, c in traces):
        return None

    border = _border_coords(r0, c0, r1, c1)
    if any(int(grid[r, c]) != BG for r, c in border):
        return None

    seed_cells = [
        (r, c)
        for r, c in _interior_coords(r0, c0, r1, c1)
        if int(grid[r, c]) != BG
    ]
    if len(seed_cells) != 1:
        return None
    seed = seed_cells[0]
    color = seed_lookup.get(seed)
    if color is None or color == TRACE:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "traces": traces,
        "trace_set": frozenset(traces),
        "seed": seed,
        "color": int(color),
    }


def _find_reconstructable_squares(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    seed_mask = (g != bg) & (g != TRACE)
    seed_components = components_mask(seed_mask, connectivity=4)
    assert all(comp.shape[0] == 1 for comp in seed_components), "seed pixels must be isolated"
    seed_lookup = {
        (int(comp[0, 0]), int(comp[0, 1])): int(g[int(comp[0, 0]), int(comp[0, 1])])
        for comp in seed_components
    }
    red_set = {(int(r), int(c)) for r, c in np.argwhere(g == TRACE)}

    h, w = g.shape
    candidates: list[dict] = []
    max_side = min(h, w) - 2
    for side in range(3, max_side + 1):
        for r0 in range(1, h - side):
            for c0 in range(1, w - side):
                candidate = _square_candidate(g, seed_lookup, red_set, r0, c0, side)
                if candidate is not None:
                    candidates.append(candidate)

    accepted: list[dict] = []
    for i, candidate in enumerate(candidates):
        ambiguous = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if candidate["seed"] == other["seed"]:
                ambiguous = True
                break
            if candidate["trace_set"] & other["trace_set"]:
                ambiguous = True
                break
        if not ambiguous:
            accepted.append(candidate)

    accepted.sort(key=lambda item: item["bbox"])
    return accepted


def _apply_reconstruction(grid: np.ndarray, squares: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for square in squares:
        for r, c in square["traces"]:
            out[r, c] = BG
        sr, sc = square["seed"]
        out[sr, sc] = BG
    for square in squares:
        color = int(square["color"])
        for r, c in square["border"]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each hidden square frame from complete red corner traces and erase the clues.

    Input structure:
    - A black background contains many red pixels plus a few isolated non-red seed pixels.
    - Each valid hidden square is represented only by eight red corner-trace pixels:
      two orthogonally adjacent outside pixels per corner of an unseen square frame.
    - Exactly one isolated non-red seed pixel lies strictly inside each valid square, while
      some extra red pixels do not complete any consistent square.

    Rule:
    - Search for square positions whose eight required corner-trace pixels are all red,
      whose frame cells are currently empty, and whose interior contains exactly one seed pixel.
    - For every uniquely determined square, draw a one-pixel-wide hollow square border using
      that seed pixel's color.
    - Erase the square's eight red trace pixels and its seed pixel; leave unmatched red noise unchanged.

    Key insight:
    - The red pixels are not the object itself: each orthogonal trace pair points diagonally inward
      to a square corner, so only a full consistent octet plus a single interior seed justifies reconstruction.
    """
    squares = _find_reconstructable_squares(grid)
    assert squares, "expected at least one reconstructable square"
    return _apply_reconstruction(grid, squares)


def _paint_points(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _can_place_square(
    reserved: np.ndarray,
    r0: int,
    c0: int,
    side: int,
    pad: int,
) -> bool:
    h, w = reserved.shape
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    box = _expanded_box(r0 - 1, c0 - 1, r1 + 1, c1 + 1, pad, reserved.shape)
    br0, bc0, br1, bc1 = box
    return not np.any(reserved[br0 : br1 + 1, bc0 : bc1 + 1])


def _place_real_square(
    grid: np.ndarray,
    reserved: np.ndarray,
    side: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(1, h - side - 1)
        c0 = random.randint(1, w - side - 1)
        if not _can_place_square(reserved, r0, c0, side, REAL_PADDING):
            continue

        r1 = r0 + side - 1
        c1 = c0 + side - 1
        seed = random.choice(_interior_coords(r0, c0, r1, c1))
        color = random.choice(SEED_COLORS)
        traces = _trace_coords(r0, c0, r1, c1)
        _paint_points(grid, traces, TRACE)
        grid[seed] = int(color)
        _mark_reserved(
            reserved,
            _expanded_box(r0 - 1, c0 - 1, r1 + 1, c1 + 1, REAL_PADDING, grid.shape),
        )
        return {
            "bbox": (r0, c0, r1, c1),
            "seed": seed,
            "color": int(color),
            "traces": traces,
            "border": _border_coords(r0, c0, r1, c1),
        }
    return None


def _place_partial_square(
    grid: np.ndarray,
    reserved: np.ndarray,
    side: int,
    min_missing: int = 1,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(1, h - side - 1)
        c0 = random.randint(1, w - side - 1)
        if not _can_place_square(reserved, r0, c0, side, PARTIAL_PADDING):
            continue
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        traces = _trace_coords(r0, c0, r1, c1)
        keep_n = random.randint(PARTIAL_TRACE_COUNT_RANGE[0], PARTIAL_TRACE_COUNT_RANGE[1])
        keep_n = min(keep_n, len(traces) - min_missing)
        kept = set(random.sample(traces, keep_n))
        _paint_points(grid, list(kept), TRACE)
        _mark_reserved(
            reserved,
            _expanded_box(r0 - 1, c0 - 1, r1 + 1, c1 + 1, PARTIAL_PADDING, grid.shape),
        )
        return kept
    return None


def _place_single_noise(
    grid: np.ndarray,
    reserved: np.ndarray,
    count: int,
) -> set[tuple[int, int]]:
    h, w = grid.shape
    placed: set[tuple[int, int]] = set()
    for _ in range(count * 20):
        if len(placed) >= count:
            break
        r = random.randrange(h)
        c = random.randrange(w)
        if reserved[r, c] or int(grid[r, c]) != BG:
            continue
        grid[r, c] = TRACE
        reserved[r, c] = True
        placed.add((r, c))
    return placed


def _expected_output(grid: np.ndarray, real_squares: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for square in real_squares:
        for r, c in square["traces"]:
            out[r, c] = BG
        sr, sc = square["seed"]
        out[sr, sc] = BG
    for square in real_squares:
        _paint_points(out, square["border"], square["color"])
    return out


def _generate_case(return_meta: bool = False):
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    target_count = random.randint(*REAL_SQUARE_COUNT_RANGE)
    max_side = min(REAL_SIDE_RANGE[1], h - 2, w - 2)
    if max_side < REAL_SIDE_RANGE[0]:
        return None

    side_choices = []
    for _ in range(target_count):
        if random.random() < 0.35:
            side_choices.append(REAL_SIDE_RANGE[0])
        else:
            side_choices.append(random.randint(REAL_SIDE_RANGE[0], max_side))
    side_choices.sort(reverse=True)

    real_squares: list[dict] = []
    for side in side_choices:
        square = _place_real_square(grid, reserved, side)
        if square is None:
            return None
        real_squares.append(square)

    partials: list[set[tuple[int, int]]] = []
    partial_target = random.randint(*PARTIAL_COUNT_RANGE)
    for _ in range(partial_target):
        side = random.randint(PARTIAL_SIDE_RANGE[0], min(PARTIAL_SIDE_RANGE[1], h - 2, w - 2))
        partial = _place_partial_square(grid, reserved, side)
        if partial is not None:
            partials.append(partial)
    if not partials:
        return None

    single_noise = _place_single_noise(grid, reserved, random.randint(*SINGLE_NOISE_RANGE))
    expected = _expected_output(grid, real_squares)

    found = _find_reconstructable_squares(grid)
    found_keys = sorted(
        (tuple(item["bbox"]), tuple(item["seed"]), int(item["color"])) for item in found
    )
    true_keys = sorted(
        (tuple(item["bbox"]), tuple(item["seed"]), int(item["color"])) for item in real_squares
    )
    if found_keys != true_keys:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    out = expected
    for r, c in single_noise:
        if int(out[r, c]) != TRACE:
            return None
    for partial in partials:
        if not all(int(out[r, c]) == TRACE for r, c in partial):
            return None
        if len(partial) >= 7:
            break
    else:
        if random.random() < 0.7:
            return None

    if return_meta:
        return {
            "grid": grid,
            "output": expected,
            "real_squares": real_squares,
            "partials": partials,
            "single_noise": single_noise,
        }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case(return_meta=False)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid input")
