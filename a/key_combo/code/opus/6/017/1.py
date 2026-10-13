from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
FLOOR = 5
MARK = 7
SQUARE_COLORS = (1, 2, 3, 4, 6)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (18, 26)
# The prompt asks for 4-6 objects, but it also requires distinct colors drawn
# from a 5-color palette, so 5 is the true feasible maximum.
OBJECT_COUNT_CHOICES = (4, 5)
SIDE_CHOICES = (2, 3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400
MIN_LIFT = 2
MAX_EXTRA_LIFT = 6


@dataclass(frozen=True)
class Square:
    color: int
    r0: int
    c0: int
    side: int

    @property
    def r1(self) -> int:
        return self.r0 + self.side - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.side - 1

    @property
    def is_odd(self) -> bool:
        return self.side % 2 == 1

    @property
    def center(self) -> tuple[int, int]:
        return self.r0 + self.side // 2, self.c0 + self.side // 2

    @property
    def floor_gap(self) -> int:
        return 2 if self.is_odd else 0


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_squares(grid: np.ndarray) -> list[Square]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must be at most 30x30"
    assert h >= 6 and w >= 6, "grid too small for floor and squares"
    assert np.all(g[-1, :] == FLOOR), "bottom row must be the grey floor"
    assert not np.any(g[:-1, :] == FLOOR), "grey floor color may only appear on the bottom row"

    non_bg = {int(v) for v in np.unique(g)} - {BG, FLOOR}
    assert non_bg, "expected foreground squares above the floor"
    assert MARK not in non_bg, "input must not already contain the output mark color"
    assert non_bg.issubset(set(SQUARE_COLORS)), "unexpected square colors"

    squares: list[Square] = []
    seen_columns: list[tuple[int, int]] = []
    for color in sorted(non_bg):
        coords_list = _find_components(g == color)
        assert len(coords_list) == 1, "each square color must form exactly one component"
        coords = coords_list[0]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side_r = int(r1 - r0 + 1)
        side_c = int(c1 - c0 + 1)
        assert side_r == side_c, "foreground component must be square"
        side = side_r
        assert 2 <= side <= 5, "square side must be between 2 and 5"
        assert coords.shape[0] == side * side, "square must be solid"
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert np.all(patch == color), "square bounding box must be uniformly filled"
        col_span = (int(c0), int(c1))
        for other_c0, other_c1 in seen_columns:
            assert col_span[1] < other_c0 or other_c1 < col_span[0], (
                "square column ranges must be disjoint"
            )
        seen_columns.append(col_span)
        squares.append(Square(int(color), int(r0), int(c0), int(side)))

    squares.sort(key=lambda sq: sq.c0)
    return squares


def _apply_transform(grid: np.ndarray, squares: list[Square]) -> np.ndarray:
    h, _w = grid.shape
    out = np.full_like(grid, BG)
    out[-1, :] = FLOOR
    floor_row = h - 1

    for sq in squares:
        landed_r0 = floor_row - sq.floor_gap - sq.side
        assert landed_r0 >= 0, "square does not fit above the floor"
        out[landed_r0 : landed_r0 + sq.side, sq.c0 : sq.c0 + sq.side] = sq.color
        if sq.is_odd:
            cr = landed_r0 + sq.side // 2
            cc = sq.c0 + sq.side // 2
            out[cr, cc] = MARK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark odd-sized solid squares, then drop every square so the mark controls how far it floats above the floor.

    Input structure:
    - The background is black, and the bottom row is a full grey floor.
    - Above the floor are 4 to 5 monochromatic solid squares, each using a distinct color from the square palette.
    - Square column spans do not overlap, so every square can fall straight down independently.

    Rule:
    - Identify each foreground component as a solid square and read its side length.
    - If a square has odd side length, recolor its single center cell orange; even-sized squares receive no mark.
    - Keep each square's columns fixed and move the whole square downward.
    - Odd squares stop with exactly two black rows between their bottom edge and the grey floor; even squares stop flush against the floor.

    Key insight:
    - The orange center mark is not decorative: odd/even side length decides whether the square gains a two-row gravity buffer, so parity must be inferred before the drop height is known.
    """
    g = np.asarray(grid, dtype=int)
    squares = _extract_squares(g)
    return _apply_transform(g, squares)


def _sample_sides(count: int) -> list[int]:
    base = [2, 3, 4, 5]
    if count == 4:
        sides = base
    else:
        sides = base + [random.choice(SIDE_CHOICES)]
    random.shuffle(sides)
    odd_sides = [s for s in sides if s % 2 == 1]
    even_sides = [s for s in sides if s % 2 == 0]
    assert len(odd_sides) >= 2 and len(even_sides) >= 2
    assert min(odd_sides) < max(even_sides)
    assert max(odd_sides) > min(even_sides)
    return sides


def _sample_width_for_sides(sides: list[int]) -> int:
    min_w = sum(sides) + (len(sides) - 1)
    max_w = GRID_W_RANGE[1]
    return random.randint(max(min_w, GRID_W_RANGE[0]), max_w)


def _sample_column_starts(width: int, sides: list[int]) -> list[int] | None:
    free = width - sum(sides)
    gap_count = len(sides) + 1
    if free < gap_count - 1:
        return None
    gaps = [0] * gap_count
    remaining = free
    for i in range(1, gap_count - 1):
        gaps[i] = 1
        remaining -= 1
    for _ in range(remaining):
        gaps[random.randrange(gap_count)] += 1
    starts = []
    c = gaps[0]
    for side, gap_after in zip(sides, gaps[1:]):
        starts.append(c)
        c += side + gap_after
    if c != width:
        return None
    return starts


def _landed_top(height: int, side: int) -> int:
    return height - 1 - (2 if side % 2 == 1 else 0) - side


def _build_output_template(
    height: int, width: int, colors: list[int], sides: list[int], starts: list[int]
) -> tuple[np.ndarray, list[Square]]:
    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = FLOOR
    squares: list[Square] = []
    for color, side, c0 in zip(colors, sides, starts):
        r0 = _landed_top(height, side)
        grid[r0 : r0 + side, c0 : c0 + side] = color
        if side % 2 == 1:
            grid[r0 + side // 2, c0 + side // 2] = MARK
        squares.append(Square(color=color, r0=r0, c0=c0, side=side))
    return grid, squares


def _lifted_input_from_output(
    output_squares: list[Square], height: int, width: int
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = FLOOR
    lifts = []
    top_half_limit = max(0, height // 2 - 1)
    for sq in output_squares:
        min_r0 = 0
        max_r0 = min(_landed_top(height, sq.side) - MIN_LIFT, top_half_limit - sq.side + 1)
        assert max_r0 >= min_r0, "height too small to place lifted square in upper half"
        max_lift = min(_landed_top(height, sq.side) - min_r0, MIN_LIFT + MAX_EXTRA_LIFT)
        lift = random.randint(MIN_LIFT, max_lift)
        lifts.append(lift)
        r0 = sq.r0 - lift
        grid[r0 : r0 + sq.side, sq.c0 : sq.c0 + sq.side] = sq.color

    assert lifts and min(lifts) >= MIN_LIFT
    assert max(lifts) > MIN_LIFT
    assert np.any(grid[: max(1, height // 3), :] != BG), "at least one square should start high"
    return grid


def _analyze_output(output: np.ndarray, input_squares: list[Square]) -> None:
    restored = output.copy()
    for sq in input_squares:
        if sq.is_odd:
            rr = output.shape[0] - 1 - 2 - sq.side + sq.side // 2
            cc = sq.c0 + sq.side // 2
            restored[rr, cc] = sq.color
    out_squares = _extract_squares(restored)
    by_color = {sq.color: sq for sq in out_squares}
    assert set(by_color) == {sq.color for sq in input_squares}
    h = output.shape[0]
    floor_top = h - 1
    marked_centers = set(map(tuple, np.argwhere(output == MARK)))
    expected_marked = 0
    for sq in input_squares:
        out_sq = by_color[sq.color]
        assert out_sq.side == sq.side
        assert out_sq.c0 == sq.c0
        expected_r0 = floor_top - (2 if sq.is_odd else 0) - sq.side
        assert out_sq.r0 == expected_r0
        if sq.is_odd:
            expected_marked += 1
            assert (out_sq.r0 + sq.side // 2, out_sq.c0 + sq.side // 2) in marked_centers
        else:
            rr = out_sq.r0 + sq.side // 2
            cc = out_sq.c0 + sq.side // 2
            assert output[rr, cc] != MARK
    assert expected_marked >= 2


def _validate_generated(grid: np.ndarray) -> None:
    squares = _extract_squares(grid)
    assert len(squares) in OBJECT_COUNT_CHOICES
    assert len({sq.color for sq in squares}) == len(squares)
    odd = [sq for sq in squares if sq.is_odd]
    even = [sq for sq in squares if not sq.is_odd]
    assert len(odd) >= 2 and len(even) >= 2
    assert min(sq.side for sq in odd) < max(sq.side for sq in even)
    assert max(sq.side for sq in odd) > min(sq.side for sq in even)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.all(out[-1, :] == FLOOR)
    assert not np.array_equal(out, grid), "generator must not produce fixed-point inputs"
    _analyze_output(out, squares)
    floor_top = out.shape[0] - 1
    moves = []
    for sq in squares:
        landed_r0 = floor_top - sq.floor_gap - sq.side
        moves.append(landed_r0 - sq.r0)
    assert all(delta >= 2 for delta in moves), "every square should visibly fall"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        count = random.choice(OBJECT_COUNT_CHOICES)
        sides = _sample_sides(count)
        width = _sample_width_for_sides(sides)
        starts = _sample_column_starts(width, sides)
        if starts is None:
            continue
        min_height = max(
            GRID_H_RANGE[0],
            1 + max(_landed_top(GRID_H_RANGE[1], side) for side in sides) + MIN_LIFT,
            max(side + MIN_LIFT + 4 for side in sides),
        )
        height = random.randint(max(GRID_H_RANGE[0], min_height), GRID_H_RANGE[1])
        colors = random.sample(list(SQUARE_COLORS), count)
        random.shuffle(colors)
        try:
            expected_output, settled_squares = _build_output_template(height, width, colors, sides, starts)
            grid = _lifted_input_from_output(settled_squares, height, width)
            _validate_generated(grid)
            assert np.array_equal(main(grid), expected_output)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
