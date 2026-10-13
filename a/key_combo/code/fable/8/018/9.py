import random

import numpy as np


BLACK = 0
GREY = 5
BORDER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIDE_RANGE = (10, 24)
FRAME_MARGIN = 2
FRAME_OUTER_MIN = 5
FRAME_OUTER_MAX = 14
KEY_LENGTH_MAX = 9
GEN_RETRIES = 400


def _border_clockwise_path(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    path = [(0, c) for c in range(w)]
    path.extend((r, w - 1) for r in range(1, h))
    path.extend((h - 1, c) for c in range(w - 2, -1, -1))
    path.extend((r, 0) for r in range(h - 2, 0, -1))
    return path


def _spiral_coords(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top = r0
    bottom = r0 + h - 1
    left = c0
    right = c0 + w - 1

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        for r in range(top + 1, bottom + 1):
            coords.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                coords.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                coords.append((r, left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1

    return coords


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == GREY
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a grey frame"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    box = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(box[0, :]) and np.all(box[-1, :]), "frame top/bottom edges must be solid"
    assert np.all(box[:, 0]) and np.all(box[:, -1]), "frame left/right edges must be solid"
    assert not np.any(box[1:-1, 1:-1]), "frame must be hollow"

    expected_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    assert int(np.count_nonzero(mask)) == expected_area, "grey pixels must form one rectangle border"
    return r0, c0, r1, c1


def _key_sequence(grid: np.ndarray) -> list[int]:
    path = _border_clockwise_path(grid.shape)
    seq = [
        int(grid[r, c])
        for r, c in path
        if int(grid[r, c]) not in (BLACK, GREY)
    ]
    assert seq, "expected colored boundary pixels"
    return seq


def _expected_output_from_key(grid: np.ndarray, key: list[int]) -> np.ndarray:
    r0, c0, r1, c1 = _find_frame(grid)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h > 0 and inner_w > 0
    assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "frame interior must start black"

    out = np.array(grid, copy=True)
    for i, (r, c) in enumerate(_spiral_coords(r0 + 1, c0 + 1, inner_h, inner_w)):
        out[r, c] = key[i % len(key)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colored outer-border pixels clockwise and replay that sequence as a clockwise spiral inside the hollow frame.

    Input structure
    - The background is black.
    - A single grey hollow rectangle appears away from the grid edge, with a fully black interior.
    - Colored pixels are scattered only on the outermost border of the grid.

    Rule
    - Starting at the top-left corner, walk clockwise around the grid border and record each non-black, non-grey pixel in encounter order.
    - Starting at the top-left cell inside the grey frame, fill the frame interior in clockwise spiral order.
    - Repeat the recorded border-color sequence cyclically until the interior is full, while leaving the border pixels and grey frame unchanged.

    Key insight
    - The border pixels matter only through their clockwise reading order; that same ordered cycle is replayed on a smaller clockwise path inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must satisfy ARC size limits"
    assert int(np.min(g)) >= 0, "color ids must be non-negative"

    r0, c0, r1, c1 = _find_frame(g)
    h, w = g.shape
    assert 0 < r0 < r1 < h - 1 and 0 < c0 < c1 < w - 1, "frame must be strictly inside the border"

    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    colored_mask = (g != BLACK) & (g != GREY)
    assert np.all(border_mask[colored_mask]), "all colored non-frame pixels must lie on the outer border"

    key = _key_sequence(g)
    return _expected_output_from_key(g, key)


def _border_indices_by_side(shape: tuple[int, int]) -> dict[str, list[int]]:
    h, w = shape
    side_to_indices = {"top": [], "right": [], "bottom": [], "left": []}
    for i, (r, c) in enumerate(_border_clockwise_path((h, w))):
        if r == 0:
            side_to_indices["top"].append(i)
        elif c == w - 1:
            side_to_indices["right"].append(i)
        elif r == h - 1:
            side_to_indices["bottom"].append(i)
        else:
            side_to_indices["left"].append(i)
    return side_to_indices


def _sample_key_length(area: int, border_slots: int) -> int | None:
    upper = min(KEY_LENGTH_MAX, border_slots, area - 1)
    candidates = [k for k in range(2, upper + 1) if area % k != 0]
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_border_indices(shape: tuple[int, int], key_len: int) -> list[int] | None:
    side_indices = _border_indices_by_side(shape)
    non_empty_sides = [side for side, indices in side_indices.items() if indices]
    side_count = min(key_len, random.randint(2, min(4, len(non_empty_sides))))
    chosen_sides = random.sample(non_empty_sides, side_count)

    chosen: set[int] = set()
    for side in chosen_sides:
        chosen.add(random.choice(side_indices[side]))

    remaining = [i for i in range(len(_border_clockwise_path(shape))) if i not in chosen]
    if len(remaining) < key_len - len(chosen):
        return None
    chosen.update(random.sample(remaining, key_len - len(chosen)))
    return sorted(chosen)


def _sample_key_colors(key_len: int) -> list[int]:
    distinct = random.randint(2, min(5, key_len))
    palette = random.sample(BORDER_COLORS, distinct)
    colors = palette[:]
    while len(colors) < key_len:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    max_frame_h = min(FRAME_OUTER_MAX, h - 2 * FRAME_MARGIN)
    max_frame_w = min(FRAME_OUTER_MAX, w - 2 * FRAME_MARGIN)
    if max_frame_h < FRAME_OUTER_MIN or max_frame_w < FRAME_OUTER_MIN:
        return None

    frame_h = random.randint(FRAME_OUTER_MIN, max_frame_h)
    frame_w = random.randint(FRAME_OUTER_MIN, max_frame_w)
    r0 = random.randint(FRAME_MARGIN, h - FRAME_MARGIN - frame_h)
    c0 = random.randint(FRAME_MARGIN, w - FRAME_MARGIN - frame_w)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    inner_h = frame_h - 2
    inner_w = frame_w - 2
    area = inner_h * inner_w

    border_path = _border_clockwise_path((h, w))
    key_len = _sample_key_length(area, len(border_path))
    if key_len is None:
        return None

    border_indices = _sample_border_indices((h, w), key_len)
    if border_indices is None:
        return None
    key_colors = _sample_key_colors(key_len)

    grid = np.full((h, w), BLACK, dtype=int)
    _draw_frame(grid, r0, c0, r1, c1)

    for idx, color in zip(border_indices, key_colors):
        br, bc = border_path[idx]
        grid[br, bc] = color

    expected = _expected_output_from_key(grid, key_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
