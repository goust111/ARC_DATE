import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
PLACEMENT_RETRIES = 120


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    candidates = [
        r
        for r in range(h)
        if r == h - r - 1 and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected one full-width separator row between equal halves"
    row = int(candidates[0])
    return row, int(grid[row, 0])


def _instruction_axes(top: np.ndarray, bg: int) -> dict[int, tuple[str, int]]:
    axes: dict[int, tuple[str, int]] = {}
    for color in (int(c) for c in np.unique(top) if int(c) != bg):
        coords = np.argwhere(top == color)
        assert len(coords) >= 3, "every instruction line must have length at least 3"
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1:
            row = int(rows[0])
            span = np.sort(coords[:, 1])
            assert int(span[-1]) - int(span[0]) + 1 == len(span), "horizontal instruction must be contiguous"
            axes[color] = ("horizontal", top.shape[0] - 1 - row)
        elif len(cols) == 1:
            col = int(cols[0])
            span = np.sort(coords[:, 0])
            assert int(span[-1]) - int(span[0]) + 1 == len(span), "vertical instruction must be contiguous"
            axes[color] = ("vertical", col)
        else:
            raise AssertionError("instruction components must be straight horizontal or vertical lines")

    assert axes, "instruction area must contain at least one line"
    return axes


def _reflect_pixel(r: int, c: int, axis: tuple[str, int]) -> tuple[int, int]:
    kind, pos = axis
    if kind == "horizontal":
        return 2 * pos - r, c
    return r, 2 * pos - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each lower-half colored object across the axis encoded by its same-color upper-half line segment.

    Input structure:
    - The background is black.
    - A full-width grey separator row splits the grid into equal top and bottom halves.
    - The top half contains several same-color instruction components, each a single horizontal or vertical line segment.
    - The bottom half contains colored objects whose colors pair them with exactly one instruction line.

    Rule:
    - Interpret each top-half line segment as a fold axis descriptor: its orientation chooses horizontal vs vertical folding, and its position is mirrored across the grey separator into the bottom half.
    - For each bottom-half color, collect all pixels of that color and reflect them across the paired axis.
    - Keep the original object pixels and also paint the reflected pixels in the same color.
    - Remove the grey separator and output only the transformed bottom half.

    Key insight:
    - The upper half is not copied into the output; it is a coordinate-encoded instruction set whose line orientation and mirrored position define a per-color reflection axis for the lower-half objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_row, _sep_color = _find_separator(g)
    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    assert top.shape == bottom.shape, "separator must split the grid into equal halves"

    axes = _instruction_axes(top, BACKGROUND)
    bottom_colors = {int(c) for c in np.unique(bottom) if int(c) != BACKGROUND}
    assert bottom_colors == set(axes), "every instruction color must pair with one bottom-half object color"

    out = bottom.copy()
    h, w = bottom.shape
    for color, axis in axes.items():
        coords = np.argwhere(bottom == color)
        assert len(coords) > 0, "paired object color must appear in the lower half"
        for r_raw, c_raw in coords:
            rr, cc = _reflect_pixel(int(r_raw), int(c_raw), axis)
            assert 0 <= rr < h and 0 <= cc < w, "reflected pixels must stay inside the output grid"
            current = int(out[rr, cc])
            assert current in {BACKGROUND, color}, "different colors may not overlap after folding"
            out[rr, cc] = color
    return out


def _mark_halo(mask: np.ndarray, coords: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        mask[r0:r1, c0:c1] = True


def _axis_interval(size: int, axis_pos: int) -> tuple[int, int]:
    low = max(0, 2 * axis_pos - (size - 1))
    high = min(size - 1, 2 * axis_pos)
    return low, high


def _random_connected_shape(box_h: int, box_w: int, cell_count: int) -> set[tuple[int, int]] | None:
    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < cell_count:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _place_line(
    top: np.ndarray,
    blocked: np.ndarray,
    color: int,
    axis: tuple[str, int],
) -> bool:
    h, w = top.shape
    kind, pos = axis
    for _ in range(PLACEMENT_RETRIES):
        if kind == "horizontal":
            row = h - 1 - pos
            max_len = w - 2
            if max_len < 3:
                return False
            length = random.randint(3, max_len)
            start = random.randint(0, w - length)
            coords = {(row, c) for c in range(start, start + length)}
        else:
            col = pos
            length = random.randint(3, h)
            start = random.randint(0, h - length)
            coords = {(r, col) for r in range(start, start + length)}

        if any(blocked[r, c] for r, c in coords):
            continue

        for r, c in coords:
            top[r, c] = color
        _mark_halo(blocked, coords)
        return True
    return False


def _place_object(
    bottom: np.ndarray,
    blocked: np.ndarray,
    color: int,
    axis: tuple[str, int],
) -> bool:
    h, w = bottom.shape
    kind, pos = axis

    for _ in range(PLACEMENT_RETRIES):
        if kind == "horizontal":
            low, high = _axis_interval(h, pos)
            side_ranges = []
            if pos - low + 1 >= 2:
                side_ranges.append((low, pos))
            if high - pos + 1 >= 2:
                side_ranges.append((pos, high))
            if not side_ranges:
                return False
            row_low, row_high = random.choice(side_ranges)
            box_h = random.randint(2, min(5, row_high - row_low + 1))
            box_w = random.randint(2, min(6, w))
            r0 = random.randint(row_low, row_high - box_h + 1)
            c0 = random.randint(0, w - box_w)
        else:
            low, high = _axis_interval(w, pos)
            side_ranges = []
            if pos - low + 1 >= 2:
                side_ranges.append((low, pos))
            if high - pos + 1 >= 2:
                side_ranges.append((pos, high))
            if not side_ranges:
                return False
            col_low, col_high = random.choice(side_ranges)
            box_h = random.randint(2, min(5, h))
            box_w = random.randint(2, min(6, col_high - col_low + 1))
            r0 = random.randint(0, h - box_h)
            c0 = random.randint(col_low, col_high - box_w + 1)

        max_cells = min(8, box_h * box_w)
        if max_cells < 3:
            continue
        cell_count = random.randint(3, max_cells)
        shape = _random_connected_shape(box_h, box_w, cell_count)
        if shape is None:
            continue

        coords = {(r0 + r, c0 + c) for r, c in shape}
        mirrored = {_reflect_pixel(r, c, axis) for r, c in coords}
        if coords == mirrored:
            continue
        union = coords | mirrored

        if any(not (0 <= r < h and 0 <= c < w) for r, c in union):
            continue
        if any(blocked[r, c] for r, c in union):
            continue

        for r, c in coords:
            bottom[r, c] = color
        _mark_halo(blocked, union)
        return True
    return False


def _choose_axes(half: int, width: int, count: int) -> list[tuple[str, int]]:
    orientations = [random.choice(("horizontal", "vertical")) for _ in range(count)]
    if len(set(orientations)) == 1:
        orientations[0] = "horizontal"
        orientations[1] = "vertical"
        random.shuffle(orientations)

    h_needed = orientations.count("horizontal")
    v_needed = orientations.count("vertical")
    h_positions = random.sample(list(range(1, half - 1)), h_needed)
    v_positions = random.sample(list(range(1, width - 1)), v_needed)

    axes = []
    h_idx = 0
    v_idx = 0
    for kind in orientations:
        if kind == "horizontal":
            axes.append((kind, h_positions[h_idx]))
            h_idx += 1
        else:
            axes.append((kind, v_positions[v_idx]))
            v_idx += 1
    return axes


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    half = random.randint(6 + count // 2, 14)
    width = random.randint(10 + 3 * count, 30)
    colors = random.sample(PALETTE, count)
    axes = _choose_axes(half, width, count)

    top = np.full((half, width), BACKGROUND, dtype=int)
    bottom = np.full((half, width), BACKGROUND, dtype=int)
    top_blocked = np.zeros((half, width), dtype=bool)
    bottom_blocked = np.zeros((half, width), dtype=bool)

    items = list(zip(colors, axes))
    random.shuffle(items)
    for color, axis in items:
        if not _place_object(bottom, bottom_blocked, color, axis):
            return None
        if not _place_line(top, top_blocked, color, axis):
            return None

    grid = np.full((2 * half + 1, width), BACKGROUND, dtype=int)
    grid[:half, :] = top
    grid[half, :] = SEPARATOR
    grid[half + 1 :, :] = bottom

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, bottom):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-axis ARC input")
