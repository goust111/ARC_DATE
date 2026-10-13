import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generator bounds.
GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (11, 28)
LINE_COUNT_RANGE = (2, 5)
MAX_ARM = 6
PLACEMENT_TRIES = 250
GENERATION_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _fold_cells(row: int, center: int, arm: int) -> list[tuple[int, int]]:
    cells = [(row, center)]
    for offset in range(1, arm + 1):
        drop = offset if row % 2 == 0 else (offset + 1) // 2
        cells.append((row + drop, center - offset))
        cells.append((row + drop, center + offset))
    return cells


def _segments_in_rows(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    segments: list[dict[str, int]] = []
    for r in range(h):
        c = 0
        while c < w:
            color = int(grid[r, c])
            if color == bg:
                c += 1
                continue

            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == color:
                c += 1
            c1 = c
            length = c1 - c0 + 1
            assert length % 2 == 1, "each input segment must have odd length"
            segments.append({"row": r, "c0": c0, "c1": c1, "color": color})
            c += 1

    colors = [seg["color"] for seg in segments]
    assert segments, "expected at least one horizontal segment"
    assert len(colors) == len(set(colors)), "each segment must have a distinct color"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each horizontal colored segment into a downward V whose slope depends on row parity.

    Input structure:
    - The background is black.
    - The grid contains several single-color horizontal line segments of odd length.
    - Each segment is one pixel thick, has a unique color, and is separated from the others.

    Rule:
    - Erase each horizontal segment after locating its midpoint cell.
    - If the segment's row index is even, redraw it as a 45-degree downward V with the midpoint as the apex.
    - If the segment's row index is odd, redraw it as a shallower downward V where each pair of horizontal steps shares one lower row.
    - Keep the same color and preserve the segment's total pixel count.

    Key insight:
    - The midpoint stays fixed as the apex, while row parity chooses between a steep fold and a flatter fold.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full((h, w), bg, dtype=int)

    for seg in _segments_in_rows(g, bg):
        row = int(seg["row"])
        c0 = int(seg["c0"])
        c1 = int(seg["c1"])
        color = int(seg["color"])
        center = (c0 + c1) // 2
        arm = (c1 - c0) // 2

        for rr, cc in _fold_cells(row, center, arm):
            assert 0 <= rr < h and 0 <= cc < w, "folded V must stay inside the grid"
            assert int(out[rr, cc]) == bg, "folded segments must not overlap"
            out[rr, cc] = color

    return out


def _input_cells(row: int, center: int, arm: int) -> list[tuple[int, int]]:
    return [(row, c) for c in range(center - arm, center + arm + 1)]


def _neighbor_cells(cells: list[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _render_input(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        row = int(spec["row"])
        center = int(spec["center"])
        arm = int(spec["arm"])
        color = int(spec["color"])
        for rr, cc in _input_cells(row, center, arm):
            assert grid[rr, cc] == BACKGROUND
            grid[rr, cc] = color
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        row = int(spec["row"])
        center = int(spec["center"])
        arm = int(spec["arm"])
        color = int(spec["color"])
        for rr, cc in _fold_cells(row, center, arm):
            assert grid[rr, cc] == BACKGROUND
            grid[rr, cc] = color
    return grid


def _sample_parities(n: int) -> list[int]:
    start = random.randrange(2)
    parities = [(start + i) % 2 for i in range(n)]
    random.shuffle(parities)
    return parities


def _place_specs(shape: tuple[int, int], n_lines: int) -> list[dict[str, int]] | None:
    h, w = shape
    specs: list[dict[str, int]] = []
    used_rows: set[int] = set()
    input_forbidden: set[tuple[int, int]] = set()
    output_used: set[tuple[int, int]] = set()

    colors = random.sample(PALETTE, n_lines)
    parities = _sample_parities(n_lines)

    for color, parity in zip(colors, parities):
        placed = False
        candidate_rows = [r for r in range(h - 1) if r % 2 == parity and r not in used_rows]
        if not candidate_rows:
            return None

        for _ in range(PLACEMENT_TRIES):
            row = random.choice(candidate_rows)
            center = random.randint(1, w - 2)
            horiz_limit = min(center, w - 1 - center, MAX_ARM)
            vert_limit = h - 1 - row if parity == 0 else 2 * (h - 1 - row)
            arm_max = min(horiz_limit, vert_limit)
            if arm_max < 1:
                continue

            arm = random.randint(1, arm_max)
            input_cells = _input_cells(row, center, arm)
            if any(cell in input_forbidden for cell in input_cells):
                continue

            output_cells = _fold_cells(row, center, arm)
            if any(cell in output_used for cell in output_cells):
                continue

            specs.append({"row": row, "center": center, "arm": arm, "color": color})
            used_rows.add(row)
            input_forbidden |= _neighbor_cells(input_cells, shape, radius=1)
            output_used.update(output_cells)
            placed = True
            break

        if not placed:
            return None

    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_lines = random.randint(*LINE_COUNT_RANGE)
    n_lines = min(n_lines, len(PALETTE), h - 1)

    specs = _place_specs((h, w), n_lines)
    if specs is None:
        return None

    inp = _render_input((h, w), specs)
    expected = _render_output((h, w), specs)

    try:
        produced = main(inp)
    except Exception:
        return None

    if not np.array_equal(produced, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-lines puzzle")
