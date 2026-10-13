from __future__ import annotations

import random

import numpy as np


BG = 0
CANVAS_SIZE = 30
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_RAIL_LENGTH = 3
MAX_RAIL_LENGTH = 8
PADDING = 1
MAX_LAYOUT_ATTEMPTS = 400
MAX_GENERATION_ATTEMPTS = 300
MAX_EXTRA_MARGIN = 4


def _color_specs(grid: np.ndarray) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == color)
        assert len(coords) >= MIN_RAIL_LENGTH + 1

        rows, counts = np.unique(coords[:, 0], return_counts=True)
        rail_rows = rows[counts > 1]
        assert len(rail_rows) == 1
        rail_row = int(rail_rows[0])

        rail_cols = sorted(int(c) for c in coords[coords[:, 0] == rail_row][:, 1])
        length = len(rail_cols)
        start_col = rail_cols[0]
        assert rail_cols == list(range(start_col, start_col + length))
        assert MIN_RAIL_LENGTH <= length <= MAX_RAIL_LENGTH

        seed_coords = coords[coords[:, 0] != rail_row]
        assert len(seed_coords) == 1
        seed_row, seed_col = map(int, seed_coords[0])
        assert abs(seed_row - rail_row) == 1
        assert start_col < seed_col < start_col + length - 1

        direction = -1 if seed_row < rail_row else 1
        specs.append(
            {
                "color": color,
                "rail_row": rail_row,
                "start_col": start_col,
                "length": length,
                "direction": direction,
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """Grow same-color vertical branches from each horizontal rail, using the seed side to choose the growth direction.

    Input structure
    - The background is black.
    - Each foreground color appears as one horizontal one-cell-thick rail plus one
      same-color seed pixel directly above or below an interior rail column.
    - Rail lengths vary from 3 to 8, and different rails use different colors.

    Rule
    - For each color, identify the unique multi-cell row as the rail and the
      remaining same-color pixel as its seed.
    - If the seed is above the rail, draw from every rail cell a vertical segment
      of rail-length upward, counting the rail cell itself; if the seed is below,
      do the same downward.
    - Keep the original rail in place, so the growth fills the rail-side square
      whose height equals the rail length.

    Key insight
    - Color pairs each seed with its rail, the seed side determines up versus
      down, and the rail length is reused as every branch length, so each rail
      expands into a same-color square on that side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for spec in _color_specs(g):
        color = spec["color"]
        rail_row = spec["rail_row"]
        start_col = spec["start_col"]
        length = spec["length"]
        direction = spec["direction"]

        if direction == -1:
            top = rail_row - length + 1
            assert top >= 0
            out[top : rail_row + 1, start_col : start_col + length] = color
        else:
            bottom = rail_row + length
            assert bottom <= g.shape[0]
            out[rail_row:bottom, start_col : start_col + length] = color

    return out


def _footprint(length: int, rail_row: int, start_col: int, direction: int) -> tuple[int, int, int, int]:
    if direction == -1:
        return rail_row - length + 1, start_col, rail_row, start_col + length - 1
    return rail_row, start_col, rail_row + length - 1, start_col + length - 1


def _sample_directions(count: int) -> list[int]:
    directions = [random.choice([-1, 1]) for _ in range(count)]
    if all(direction == directions[0] for direction in directions):
        directions[random.randrange(count)] *= -1
    return directions


def _place_specs(lengths: list[int], colors: list[int], directions: list[int]) -> list[dict[str, int]] | None:
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    specs: list[dict[str, int] | None] = [None] * len(lengths)
    order = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)

    for idx in order:
        length = lengths[idx]
        color = colors[idx]
        direction = directions[idx]

        for _ in range(MAX_LAYOUT_ATTEMPTS):
            if direction == -1:
                rail_row = random.randint(length - 1, CANVAS_SIZE - 1)
            else:
                rail_row = random.randint(0, CANVAS_SIZE - length)
            start_col = random.randint(0, CANVAS_SIZE - length)
            r0, c0, r1, c1 = _footprint(length, rail_row, start_col, direction)

            pr0 = max(0, r0 - PADDING)
            pc0 = max(0, c0 - PADDING)
            pr1 = min(CANVAS_SIZE - 1, r1 + PADDING)
            pc1 = min(CANVAS_SIZE - 1, c1 + PADDING)
            if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            specs[idx] = {
                "color": color,
                "rail_row": rail_row,
                "start_col": start_col,
                "length": length,
                "direction": direction,
                "seed_col": random.randint(start_col + 1, start_col + length - 2),
            }
            break
        else:
            return None

    return [spec for spec in specs if spec is not None]


def _render_input(specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    for spec in specs:
        color = spec["color"]
        rail_row = spec["rail_row"]
        start_col = spec["start_col"]
        length = spec["length"]
        seed_row = rail_row - 1 if spec["direction"] == -1 else rail_row + 1
        grid[rail_row, start_col : start_col + length] = color
        grid[seed_row, spec["seed_col"]] = color
    return grid


def _render_output(specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = _footprint(spec["length"], spec["rail_row"], spec["start_col"], spec["direction"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
    return grid


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(out != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_margin = random.randint(0, min(MAX_EXTRA_MARGIN, int(r0)))
    left_margin = random.randint(0, min(MAX_EXTRA_MARGIN, int(c0)))
    bottom_margin = random.randint(0, min(MAX_EXTRA_MARGIN, CANVAS_SIZE - 1 - int(r1)))
    right_margin = random.randint(0, min(MAX_EXTRA_MARGIN, CANVAS_SIZE - 1 - int(c1)))

    rs = slice(int(r0) - top_margin, int(r1) + bottom_margin + 1)
    cs = slice(int(c0) - left_margin, int(c1) + right_margin + 1)
    return inp[rs, cs], out[rs, cs]


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    lengths = random.sample(range(MIN_RAIL_LENGTH, MAX_RAIL_LENGTH + 1), count)
    colors = random.sample(range(1, 10), count)
    directions = _sample_directions(count)

    specs = _place_specs(lengths, colors, directions)
    if specs is None:
        return None

    inp = _render_input(specs)
    out = _render_output(specs)
    inp, out = _crop_pair(inp, out)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-growth input")
