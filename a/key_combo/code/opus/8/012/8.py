from __future__ import annotations

from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_task_8_012",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


BG = Color.BLACK
MARK = Color.TEAL
SQUARE_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.ORANGE,
    Color.PINK,
]
ODD_SIDES = (3, 5)
EVEN_SIDES = (2, 4)
GRID_SIDE_RANGE = (16, 22)
GEN_RETRY_LIMIT = 400

_CROSS_PATCH = np.array(
    [
        [0, 8, 0],
        [8, 8, 8],
        [0, 8, 0],
    ],
    dtype=int,
)


@dataclass(frozen=True)
class SquareSpec:
    color: int
    top: int
    left: int
    side: int

    @property
    def bottom(self) -> int:
        return self.top + self.side - 1

    @property
    def right(self) -> int:
        return self.left + self.side - 1

    @property
    def center(self) -> tuple[int, int]:
        mid = self.side // 2
        return self.top + mid, self.left + mid

    @property
    def is_odd(self) -> bool:
        return self.side % 2 == 1


def _is_plus_cross(comp: dict) -> bool:
    if int(comp["color"]) != MARK or int(comp["area"]) != 5:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    center = (r0 + 1, c0 + 1)
    expected = {
        center,
        (center[0] - 1, center[1]),
        (center[0] + 1, center[1]),
        (center[0], center[1] - 1),
        (center[0], center[1] + 1),
    }
    return coords == expected


def _is_solid_square(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    return side_r == side_c and int(comp["area"]) == side_r * side_c


def _extract_scene(grid: np.ndarray) -> tuple[list[SquareSpec], tuple[int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
    )

    squares: list[SquareSpec] = []
    crosses: list[tuple[int, int]] = []
    for comp in comps:
        color = int(comp["color"])
        if color == MARK:
            assert _is_plus_cross(comp), "teal object must be the unique 3x3 plus marker"
            r0, c0, _r1, _c1 = map(int, comp["bbox"])
            crosses.append((r0, c0))
            continue

        assert _is_solid_square(comp), "every non-marker object must be a solid square"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = r1 - r0 + 1
        assert side in {2, 3, 4, 5}, "square side must be in 2..5"
        assert color in SQUARE_COLORS, "unexpected square color"
        squares.append(SquareSpec(color=color, top=r0, left=c0, side=side))

    assert len(crosses) == 1, "expected exactly one teal cross marker"
    assert 5 <= len(squares) <= 7, "expected 5 to 7 square objects"
    odd_count = sum(spec.is_odd for spec in squares)
    even_count = len(squares) - odd_count
    assert odd_count >= 2 and even_count >= 2 and odd_count != even_count
    return squares, crosses[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the majority parity group of solid squares, marking surviving odd squares and binding the teal cross to that odd group.

    Input structure:
    - The grid has black background, one separate teal 3x3 plus marker, and 5 to 7 non-touching solid single-color squares.
    - Each square is a 4-connected component with side length 2 to 5, does not touch the border, and uses a non-teal foreground color.
    - Squares are classified only by their side-length parity: odd side lengths versus even side lengths.

    Rule:
    - Identify every non-teal solid square and measure its side length.
    - Split the squares into the odd-length group and the even-length group.
    - Compare the group counts; keep every square from the larger group and erase every square from the smaller group to black.
    - For each kept odd square, recolor its unique center pixel to teal.
    - Keep the original teal plus marker only when the odd-length group is the majority; otherwise erase that marker too.

    Key insight:
    - The teal color is attached to the odd group rather than being an independent object: odd squares gain teal centers, and the separate teal plus survives only if odd squares win the count comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    squares, cross_top_left = _extract_scene(g)
    odd_squares = [spec for spec in squares if spec.is_odd]
    even_squares = [spec for spec in squares if not spec.is_odd]
    odd_wins = len(odd_squares) > len(even_squares)

    out = g.copy()
    if not odd_wins:
        cr, cc = cross_top_left
        out[cr : cr + 3, cc : cc + 3][_CROSS_PATCH == MARK] = BG

    for spec in squares:
        if spec.is_odd != odd_wins:
            out[spec.top : spec.bottom + 1, spec.left : spec.right + 1] = BG
            continue
        if spec.is_odd:
            rr, cc = spec.center
            out[rr, cc] = MARK

    return out


def _choose_group_counts() -> tuple[int, int]:
    total = random.choice((5, 6))
    odd_majority = random.choice((True, False))
    if total == 5:
        return (3, 2) if odd_majority else (2, 3)
    return (4, 2) if odd_majority else (2, 4)


def _sample_sizes(count: int, parity: str, majority_area_small: bool) -> list[int]:
    if parity == "odd":
        if majority_area_small:
            patterns = {
                2: ([5, 5], [3, 5], [5, 3]),
                3: ([3, 3, 3], [3, 3, 5], [3, 5, 3]),
                4: ([3, 3, 3, 3], [3, 3, 3, 5], [3, 5, 3, 3]),
            }
        else:
            patterns = {
                2: ([3, 3], [3, 5], [5, 3]),
                3: ([5, 5, 5], [5, 5, 3], [5, 3, 5]),
                4: ([5, 5, 5, 5], [5, 5, 5, 3], [5, 5, 3, 5]),
            }
    else:
        if majority_area_small:
            patterns = {
                2: ([4, 4], [2, 4], [4, 2]),
                3: ([2, 2, 2], [2, 2, 4], [2, 4, 2]),
                4: ([2, 2, 2, 2], [2, 2, 2, 4], [2, 4, 2, 2]),
            }
        else:
            patterns = {
                2: ([2, 2], [2, 4], [4, 2]),
                3: ([4, 4, 4], [4, 4, 2], [4, 2, 4]),
                4: ([4, 4, 4, 4], [4, 4, 4, 2], [4, 2, 4, 4]),
            }
    base = list(random.choice(patterns[count]))
    if count >= 3 and random.random() < 0.5:
        base = random.sample(base, len(base))
    return base


def _render_expected(inp: np.ndarray, squares: list[SquareSpec], cross_top_left: tuple[int, int]) -> np.ndarray:
    out = inp.copy()
    odd_squares = [spec for spec in squares if spec.is_odd]
    even_squares = [spec for spec in squares if not spec.is_odd]
    odd_wins = len(odd_squares) > len(even_squares)

    if not odd_wins:
        cr, cc = cross_top_left
        out[cr : cr + 3, cc : cc + 3][_CROSS_PATCH == MARK] = BG

    for spec in squares:
        if spec.is_odd != odd_wins:
            out[spec.top : spec.bottom + 1, spec.left : spec.right + 1] = BG
        elif spec.is_odd:
            rr, cc = spec.center
            out[rr, cc] = MARK
    return out


def _validate_case(inp: np.ndarray, squares: list[SquareSpec], cross_top_left: tuple[int, int]) -> bool:
    parsed_squares, parsed_cross = _extract_scene(inp)
    expected_specs = sorted((s.top, s.left, s.side, s.color) for s in squares)
    parsed_specs = sorted((s.top, s.left, s.side, s.color) for s in parsed_squares)
    if expected_specs != parsed_specs or parsed_cross != cross_top_left:
        return False

    expected = _render_expected(inp, squares, cross_top_left)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    odd_wins = sum(s.is_odd for s in squares) > sum((not s.is_odd) for s in squares)
    cr, cc = cross_top_left
    cross_mask = _CROSS_PATCH == MARK
    cross_block = actual[cr : cr + 3, cc : cc + 3]
    if odd_wins != np.array_equal(cross_block[cross_mask], np.full(5, MARK, dtype=int)):
        return False

    for spec in squares:
        block = actual[spec.top : spec.bottom + 1, spec.left : spec.right + 1]
        if spec.is_odd != odd_wins:
            if np.any(block != BG):
                return False
        elif spec.is_odd:
            rr, cc = spec.center
            center_r = rr - spec.top
            center_c = cc - spec.left
            if int(block[center_r, center_c]) != MARK:
                return False
            block_copy = block.copy()
            block_copy[center_r, center_c] = spec.color
            if not np.all(block_copy == spec.color):
                return False
        else:
            if not np.all(block == spec.color):
                return False

    return True


def _place_square(canvas: np.ndarray, color: int, side: int) -> SquareSpec:
    patch = np.full((side, side), color, dtype=int)
    top, left = random_free_location(
        canvas,
        patch,
        transparent=BG,
        border=1,
        padding=1,
        padding_connectivity=8,
    )
    paste(canvas, patch, top, left, transparent=BG)
    return SquareSpec(color=color, top=top, left=left, side=side)


def _place_cross(canvas: np.ndarray) -> tuple[int, int]:
    top, left = random_free_location(
        canvas,
        _CROSS_PATCH,
        transparent=BG,
        border=1,
        padding=1,
        padding_connectivity=8,
    )
    paste(canvas, _CROSS_PATCH, top, left, transparent=BG)
    return top, left


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    canvas = np.full((height, width), BG, dtype=int)

    odd_count, even_count = _choose_group_counts()
    odd_wins = odd_count > even_count
    majority_area_small = random.choice((True, False))

    odd_sizes = _sample_sizes(odd_count, "odd", majority_area_small if odd_wins else not majority_area_small)
    even_sizes = _sample_sizes(even_count, "even", majority_area_small if not odd_wins else not majority_area_small)

    total_area_odd = sum(side * side for side in odd_sizes)
    total_area_even = sum(side * side for side in even_sizes)
    if majority_area_small:
        if odd_wins and total_area_odd >= total_area_even:
            return None
        if (not odd_wins) and total_area_even >= total_area_odd:
            return None
    else:
        if odd_wins and total_area_odd <= total_area_even:
            return None
        if (not odd_wins) and total_area_even <= total_area_odd:
            return None

    colors = random.sample(SQUARE_COLORS, odd_count + even_count)
    items: list[tuple[int, int]] = []
    for side in odd_sizes:
        items.append((side, colors.pop()))
    for side in even_sizes:
        items.append((side, colors.pop()))
    random.shuffle(items)
    items.sort(key=lambda item: item[0], reverse=True)

    squares: list[SquareSpec] = []
    try:
        for side, color in items:
            squares.append(_place_square(canvas, color, side))
        cross_top_left = _place_cross(canvas)
    except ValueError:
        return None

    if not _validate_case(canvas, squares, cross_top_left):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
