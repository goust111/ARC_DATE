from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

INPUT_COLORS = (3, 4, 6, 7, 8, 9)
MAIN_DIAGONAL_MISSING = ("tl", "br")
ANTI_DIAGONAL_MISSING = ("tr", "bl")
ALL_MISSING_KINDS = ("tl", "tr", "bl", "br")

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
SQUARE_COUNT_CHOICES = (2, 3)
SQUARE_SIDE_RANGE = (5, 9)
SQUARE_GAP = 2
GREY_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 400

SquareSpec = dict[str, int | str]
BBox = tuple[int, int, int, int]


def _corner_map(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }


def _infer_square(coords: np.ndarray) -> tuple[int, int, int, int, tuple[int, int]] | None:
    if coords.shape != (3, 2):
        return None

    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != 3:
        return None

    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return None

    r0, r1 = rows
    c0, c1 = cols
    if r1 - r0 != c1 - c0 or r1 == r0:
        return None

    corners = {(r, c) for r in rows for c in cols}
    if not pts < corners:
        return None

    missing = next(iter(corners - pts))
    return r0, c0, r1, c1, missing


def _fill_color_for_missing(r0: int, c0: int, r1: int, c1: int, missing: tuple[int, int]) -> int:
    return RED if missing in {(r0, c0), (r1, c1)} else BLUE


def _draw_completed_square(
    grid: np.ndarray,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    border_color: int,
    missing: tuple[int, int],
) -> None:
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = _fill_color_for_missing(r0, c0, r1, c1, missing)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each three-corner square and fill its interior by missing-corner diagonal class.

    Input structure:
    - A mostly black grid contains several isolated colored pixels and a few grey distractor pixels.
    - Each meaningful non-grey color appears exactly three times, occupying three corners of an axis-aligned square.
    - The grey pixels are unrelated clutter and do not participate in any square.

    Rule:
    - Ignore grey distractors.
    - For each non-black, non-grey color group that forms exactly three corners of one axis-aligned square, infer the missing fourth corner.
    - Draw the full square border in that group's color, then fill the square interior red when the missing corner is on the main diagonal (top-left or bottom-right) and blue when it is on the anti-diagonal (top-right or bottom-left).

    Key insight:
    - The color used for filling does not depend on the marker color or square size; it is determined only by which diagonal the missing corner belongs to.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)):
        info = _infer_square(np.argwhere(g == color))
        if info is None:
            continue
        r0, c0, r1, c1, missing = info
        _draw_completed_square(
            out,
            r0=r0,
            c0=c0,
            r1=r1,
            c1=c1,
            border_color=color,
            missing=missing,
        )

    return out


def _bbox_separated(a: BBox, b: BBox, gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_square_specs(height: int, width: int, count: int) -> list[SquareSpec] | None:
    missing_kinds = [random.choice(MAIN_DIAGONAL_MISSING), random.choice(ANTI_DIAGONAL_MISSING)]
    while len(missing_kinds) < count:
        missing_kinds.append(random.choice(ALL_MISSING_KINDS))
    random.shuffle(missing_kinds)

    colors = random.sample(INPUT_COLORS, count)
    specs: list[SquareSpec] = []

    for color, missing_kind in zip(colors, missing_kinds):
        placed = False
        for _ in range(120):
            max_side = min(SQUARE_SIDE_RANGE[1], height - 2, width - 2)
            if max_side < SQUARE_SIDE_RANGE[0]:
                return None
            side = random.randint(SQUARE_SIDE_RANGE[0], max_side)
            r0 = random.randint(0, height - side - 1)
            c0 = random.randint(0, width - side - 1)
            r1 = r0 + side
            c1 = c0 + side
            bbox = (r0, c0, r1, c1)

            if any(not _bbox_separated(bbox, (int(s["r0"]), int(s["c0"]), int(s["r1"]), int(s["c1"])), SQUARE_GAP) for s in specs):
                continue

            specs.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "color": color,
                    "missing_kind": missing_kind,
                }
            )
            placed = True
            break

        if not placed:
            return None

    return specs


def _render_input(height: int, width: int, squares: list[SquareSpec], greys: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    for spec in squares:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        color = int(spec["color"])
        corners = _corner_map(r0, c0, r1, c1)
        missing_kind = str(spec["missing_kind"])
        for kind, (r, c) in corners.items():
            if kind != missing_kind:
                grid[r, c] = color

    for r, c in greys:
        grid[r, c] = GREY

    return grid


def _render_expected(inp: np.ndarray, squares: list[SquareSpec]) -> np.ndarray:
    out = inp.copy()
    for spec in squares:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        color = int(spec["color"])
        missing = _corner_map(r0, c0, r1, c1)[str(spec["missing_kind"])]
        _draw_completed_square(
            out,
            r0=r0,
            c0=c0,
            r1=r1,
            c1=c1,
            border_color=color,
            missing=missing,
        )
    return out


def _forbidden_for_grey(height: int, width: int, squares: list[SquareSpec]) -> np.ndarray:
    forbidden = np.zeros((height, width), dtype=bool)
    for spec in squares:
        r0 = max(0, int(spec["r0"]) - 1)
        c0 = max(0, int(spec["c0"]) - 1)
        r1 = min(height - 1, int(spec["r1"]) + 1)
        c1 = min(width - 1, int(spec["c1"]) + 1)
        forbidden[r0 : r1 + 1, c0 : c1 + 1] = True
    return forbidden


def _sample_greys(height: int, width: int, squares: list[SquareSpec]) -> list[tuple[int, int]] | None:
    count = random.randint(*GREY_COUNT_RANGE)
    forbidden = _forbidden_for_grey(height, width, squares)
    spots = [(r, c) for r in range(height) for c in range(width) if not forbidden[r, c]]
    random.shuffle(spots)

    chosen: list[tuple[int, int]] = []
    for r, c in spots:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            return chosen
    return None


def _branch_coverage_ok(squares: list[SquareSpec]) -> bool:
    kinds = {str(spec["missing_kind"]) for spec in squares}
    return bool(kinds & set(MAIN_DIAGONAL_MISSING)) and bool(kinds & set(ANTI_DIAGONAL_MISSING))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    square_count = random.choice(SQUARE_COUNT_CHOICES)

    squares = _sample_square_specs(height, width, square_count)
    if squares is None or not _branch_coverage_ok(squares):
        return None

    greys = _sample_greys(height, width, squares)
    if greys is None:
        return None

    inp = _render_input(height, width, squares, greys)
    expected = _render_expected(inp, squares)

    if len({int(spec["color"]) for spec in squares}) != square_count:
        return None
    if int((inp == GREY).sum()) != len(greys):
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if int((expected == RED).sum()) == 0 or int((expected == BLUE).sum()) == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
