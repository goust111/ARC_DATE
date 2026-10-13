from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(grid != BLACK)
    assert box is not None, "expected a framed object"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    perimeter = np.zeros_like(grid, dtype=bool)
    perimeter[r0, c0 : c1 + 1] = True
    perimeter[r1, c0 : c1 + 1] = True
    perimeter[r0 : r1 + 1, c0] = True
    perimeter[r0 : r1 + 1, c1] = True

    assert not np.any((grid != BLACK) & (~perimeter)), "non-black pixels must lie on the frame perimeter"
    return r0, c0, r1, c1


def _collect_axes(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict[int, tuple[list[int], list[int]]]:
    r0, c0, r1, c1 = box
    axes: dict[int, tuple[list[int], list[int]]] = {}
    for color in (BLUE, RED):
        rows = [
            int(r)
            for r in range(r0 + 1, r1)
            if int(grid[r, c0]) == color and int(grid[r, c1]) == color
        ]
        cols = [
            int(c)
            for c in range(c0 + 1, c1)
            if int(grid[r0, c]) == color and int(grid[r1, c]) == color
        ]
        axes[color] = (rows, cols)
    return axes


def _coverage_from_axes(
    rows: list[int], cols: list[int], box: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    covered: set[tuple[int, int]] = set()
    for r in rows:
        for c in cols:
            covered.add((int(r), int(c)))
            for dr, dc in DIRS4:
                rr = int(r + 2 * dr)
                cc = int(c + 2 * dc)
                while r0 < rr < r1 and c0 < cc < c1:
                    covered.add((rr, cc))
                    rr += 2 * dr
                    cc += 2 * dc
    return covered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate frame-interior seed crossings from matched edge markers, then spread alternating-gap stripes from them.

    Input structure:
    - A black background contains one grey rectangular frame with a black interior.
    - Red and blue single-pixel markers replace some cells on opposite frame edges.
    - Matching marker rows on the left/right edges and matching marker columns on the top/bottom edges define interior seed crossings.

    Rule:
    - For each color C in {blue, red}, collect all interior rows whose left and right frame markers are both C, and all interior columns whose top and bottom frame markers are both C.
    - Every Cartesian-product crossing of those rows and columns is a seed. From each seed, paint the seed itself and then extend in the four cardinal directions with step size 2, stopping at the frame boundary; the skipped cells remain black.
    - If a pixel is covered by both colors, keep the stripe color whose seed-count product is larger. If the two seed counts are equal, red wins. Frame and marker pixels stay unchanged.

    Key insight:
    - The local stripe picture is not enough: collisions are resolved by a global count, namely how many seed crossings that color owns across the whole frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    box = _frame_bbox(g)
    axes = _collect_axes(g, box)
    blue_rows, blue_cols = axes[BLUE]
    red_rows, red_cols = axes[RED]

    blue_cover = _coverage_from_axes(blue_rows, blue_cols, box)
    red_cover = _coverage_from_axes(red_rows, red_cols, box)

    blue_count = len(blue_rows) * len(blue_cols)
    red_count = len(red_rows) * len(red_cols)
    overlap_winner = RED if red_count >= blue_count else BLUE

    for r, c in blue_cover - red_cover:
        out[int(r), int(c)] = BLUE
    for r, c in red_cover - blue_cover:
        out[int(r), int(c)] = RED
    for r, c in blue_cover & red_cover:
        out[int(r), int(c)] = overlap_winner

    return out


SCENARIOS = [
    {"winner": RED, "conflict": True, "red_shape": (2, 2), "blue_shape": (1, 2)},
    {"winner": BLUE, "conflict": True, "red_shape": (1, 2), "blue_shape": (2, 2)},
    {"winner": RED, "conflict": True, "red_shape": (1, 3), "blue_shape": (3, 1), "tie": True},
    {"winner": RED, "conflict": False, "red_shape": (3, 1), "blue_shape": (1, 2)},
    {"winner": BLUE, "conflict": False, "red_shape": (2, 1), "blue_shape": (1, 3)},
    {"winner": RED, "conflict": False, "red_shape": (2, 1), "blue_shape": (1, 2), "tie": True},
]


def _sample_positions(candidates: list[int], count: int, avoid: set[int]) -> list[int] | None:
    pool = [int(v) for v in candidates if int(v) not in avoid]
    if len(pool) < count:
        return None
    return sorted(random.sample(pool, count))


def _pick_axes(
    positions: list[int],
    red_count: int,
    blue_count: int,
    parity_mode: tuple[int, int],
) -> tuple[list[int], list[int]] | None:
    red_parity, blue_parity = parity_mode
    by_parity = {
        0: [int(v) for v in positions if int(v) % 2 == 0],
        1: [int(v) for v in positions if int(v) % 2 == 1],
    }

    red_vals = _sample_positions(by_parity[red_parity], red_count, set())
    if red_vals is None:
        return None
    blue_vals = _sample_positions(by_parity[blue_parity], blue_count, set(red_vals))
    if blue_vals is None:
        return None
    return red_vals, blue_vals


def _compute_overlap_summary(grid: np.ndarray) -> dict[str, object]:
    box = _frame_bbox(grid)
    axes = _collect_axes(grid, box)
    blue_rows, blue_cols = axes[BLUE]
    red_rows, red_cols = axes[RED]
    blue_cover = _coverage_from_axes(blue_rows, blue_cols, box)
    red_cover = _coverage_from_axes(red_rows, red_cols, box)
    return {
        "box": box,
        "axes": axes,
        "blue_cover": blue_cover,
        "red_cover": red_cover,
        "overlap": blue_cover & red_cover,
        "blue_count": len(blue_rows) * len(blue_cols),
        "red_count": len(red_rows) * len(red_cols),
    }


def _build_case() -> np.ndarray | None:
    scenario = random.choice(SCENARIOS + SCENARIOS[:3])

    H = random.randint(16, 24)
    W = random.randint(16, 24)
    top = random.randint(2, 3)
    bottom = random.randint(2, 3)
    left = random.randint(2, 3)
    right = random.randint(2, 3)
    r0, c0 = top, left
    r1, c1 = H - 1 - bottom, W - 1 - right
    if r1 - r0 - 1 < 7 or c1 - c0 - 1 < 7:
        return None

    row_positions = list(range(r0 + 1, r1))
    col_positions = list(range(c0 + 1, c1))

    if scenario["conflict"]:
        row_parity = random.randint(0, 1)
        col_parity = random.randint(0, 1)
        row_axes = _pick_axes(
            row_positions,
            scenario["red_shape"][0],
            scenario["blue_shape"][0],
            (row_parity, row_parity),
        )
        col_axes = _pick_axes(
            col_positions,
            scenario["red_shape"][1],
            scenario["blue_shape"][1],
            (col_parity, col_parity),
        )
    else:
        if random.random() < 0.5:
            row_parities = (0, 1)
            col_parities = (random.randint(0, 1), random.randint(0, 1))
        else:
            row_parities = (random.randint(0, 1), random.randint(0, 1))
            col_parities = (0, 1)
        row_axes = _pick_axes(
            row_positions,
            scenario["red_shape"][0],
            scenario["blue_shape"][0],
            row_parities,
        )
        col_axes = _pick_axes(
            col_positions,
            scenario["red_shape"][1],
            scenario["blue_shape"][1],
            col_parities,
        )

    if row_axes is None or col_axes is None:
        return None

    red_rows, blue_rows = row_axes
    red_cols, blue_cols = col_axes

    grid = np.full((H, W), BLACK, dtype=int)
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for r in red_rows:
        grid[int(r), c0] = RED
        grid[int(r), c1] = RED
    for r in blue_rows:
        grid[int(r), c0] = BLUE
        grid[int(r), c1] = BLUE
    for c in red_cols:
        grid[r0, int(c)] = RED
        grid[r1, int(c)] = RED
    for c in blue_cols:
        grid[r0, int(c)] = BLUE
        grid[r1, int(c)] = BLUE

    summary = _compute_overlap_summary(grid)
    overlap = summary["overlap"]
    red_count = int(summary["red_count"])
    blue_count = int(summary["blue_count"])
    tie = bool(scenario.get("tie", False))

    if tie and red_count != blue_count:
        return None
    if not tie and red_count == blue_count:
        return None
    if scenario["winner"] == RED and red_count < blue_count:
        return None
    if scenario["winner"] == BLUE and blue_count < red_count:
        return None
    if bool(scenario["conflict"]) != bool(overlap):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    axes = summary["axes"]
    red_rows2, red_cols2 = axes[RED]
    blue_rows2, blue_cols2 = axes[BLUE]
    if not (1 <= len(red_rows2) <= 3 and 1 <= len(red_cols2) <= 3):
        return None
    if not (1 <= len(blue_rows2) <= 3 and 1 <= len(blue_cols2) <= 3):
        return None
    if set(red_rows2) & set(blue_rows2):
        return None
    if set(red_cols2) & set(blue_cols2):
        return None

    red_seed_cells = {(r, c) for r in red_rows2 for c in red_cols2}
    blue_seed_cells = {(r, c) for r in blue_rows2 for c in blue_cols2}
    if red_seed_cells & blue_seed_cells:
        return None

    red_cover = summary["red_cover"]
    blue_cover = summary["blue_cover"]
    if len(red_cover) <= len(red_seed_cells) or len(blue_cover) <= len(blue_seed_cells):
        return None

    if overlap:
        winner = RED if red_count >= blue_count else BLUE
        for r, c in overlap:
            if int(out[r, c]) != winner:
                return None
        loser = BLUE if winner == RED else RED
        if winner == RED and not (red_count >= blue_count):
            return None
        if winner == BLUE and not (blue_count > red_count):
            return None
        if any(int(out[r, c]) == loser for r, c in overlap):
            return None

    non_frame_changes = np.argwhere((out != grid) & (grid == BLACK))
    if len(non_frame_changes) < 6:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid framed stripe instance")
