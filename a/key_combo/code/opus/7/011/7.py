from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 1
CONTROL = 2
PATTERN = 3

BBox = tuple[int, int, int, int]

GEN_RETRIES = 400
GRID_RANGE = (16, 24)
MARKER_GAP_RANGE = (2, 5)
SEED_SHAPES = ((2, 2), (2, 3))


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _draw_border(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _tile(seed: np.ndarray, height: int, width: int) -> np.ndarray:
    row_idx = np.arange(height) % seed.shape[0]
    col_idx = np.arange(width) % seed.shape[1]
    return seed[row_idx[:, None], col_idx[None, :]].copy()


def _axis_period(arr: np.ndarray, axis: int) -> int:
    length = int(arr.shape[axis])
    for period in range(1, length + 1):
        ok = True
        if axis == 0:
            for r in range(length):
                if not np.array_equal(arr[r, :], arr[r % period, :]):
                    ok = False
                    break
        else:
            for c in range(length):
                if not np.array_equal(arr[:, c], arr[:, c % period]):
                    ok = False
                    break
        if ok:
            return period
    raise AssertionError("period search failed")


def _minimal_seed(pattern_grid: np.ndarray) -> np.ndarray:
    ph = _axis_period(pattern_grid, axis=0)
    pw = _axis_period(pattern_grid, axis=1)
    seed = pattern_grid[:ph, :pw].copy()
    assert np.array_equal(_tile(seed, *pattern_grid.shape), pattern_grid)
    return seed


def _classify_controls(control_coords: np.ndarray, box: BBox) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    top: list[tuple[int, int]] = []
    bottom: list[tuple[int, int]] = []
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []

    for r, c in control_coords:
        rr = int(r)
        cc = int(c)
        if rr < r0 and c0 <= cc <= c1:
            top.append((rr, cc))
        elif rr > r1 and c0 <= cc <= c1:
            bottom.append((rr, cc))
        elif cc < c0 and r0 <= rr <= r1:
            left.append((rr, cc))
        elif cc > c1 and r0 <= rr <= r1:
            right.append((rr, cc))
        else:
            raise AssertionError(f"control pixel {(rr, cc)} is not aligned to exactly one side")

    assert len(top) == len(bottom) == len(left) == len(right) == 1
    return top[0], bottom[0], left[0], right[0]


def _new_box_from_controls(box: BBox, controls: np.ndarray, shape: tuple[int, int]) -> BBox:
    _ = shape
    top, bottom, left, right = _classify_controls(controls, box)
    new_r0 = top[0] + 1
    new_r1 = bottom[0] - 1
    new_c0 = left[1] + 1
    new_c1 = right[1] - 1
    assert new_r0 <= box[0] <= box[2] <= new_r1
    assert new_c0 <= box[1] <= box[3] <= new_c1
    return int(new_r0), int(new_c0), int(new_r1), int(new_c1)


def _render_output(shape: tuple[int, int], new_box: BBox, seed: np.ndarray, controls: np.ndarray) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[controls[:, 0], controls[:, 1]] = CONTROL

    _draw_border(out, new_box, FRAME)
    r0, c0, r1, c1 = new_box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    tiled = _tile(seed, inner_h, inner_w)
    out[r0 + 1 : r1, c0 + 1 : c1] = tiled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the blue frame to the control-defined rectangle, then tile its enlarged interior with the seed pattern extracted from the original interior.

    Input structure:
    - The background is black.
    - A blue hollow rectangle encloses a cropped repetition of a green-on-black seed pattern.
    - Four red isolated control pixels lie outside the rectangle, one aligned to each side.

    Rule:
    - Detect the blue rectangle and read its inner pattern.
    - Extract the smallest top-left seed block whose repetitions reproduce the whole original interior.
    - Move the frame sides so they become adjacent to the corresponding red control pixels.
    - Fill the new interior from its top-left corner by repeating the extracted seed and keep the red control pixels.

    Key insight:
    - The red pixels determine only the new container, while the repeated content must be recovered independently as the minimal periodic unit hidden inside the old frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_mask = g == FRAME
    old_box = _bbox(frame_mask)
    assert old_box is not None
    r0, c0, r1, c1 = old_box
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    expected_frame = np.zeros_like(frame_mask)
    _draw_border(expected_frame, old_box, True)
    assert np.array_equal(frame_mask, expected_frame)

    controls = np.argwhere(g == CONTROL)
    assert controls.shape == (4, 2)
    new_box = _new_box_from_controls(old_box, controls, g.shape)

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0
    assert set(map(int, np.unique(interior))) <= {BG, PATTERN}
    assert np.any(interior == PATTERN)

    seed = _minimal_seed(interior)
    return _render_output(g.shape, new_box, seed, controls)


def _sample_seed() -> np.ndarray:
    seed_h, seed_w = random.choice(SEED_SHAPES)
    while True:
        seed = np.zeros((seed_h, seed_w), dtype=int)
        n_green = random.randint(2, seed_h * seed_w - 1)
        coords = random.sample([(r, c) for r in range(seed_h) for c in range(seed_w)], n_green)
        for r, c in coords:
            seed[r, c] = PATTERN
        reduced = _minimal_seed(seed)
        if reduced.shape == seed.shape:
            return seed


def _sample_inner_shape(seed: np.ndarray) -> tuple[int, int]:
    seed_h, seed_w = map(int, seed.shape)
    options: list[tuple[int, int]] = []
    for inner_h in range(3, 6):
        for inner_w in range(3, 6):
            if seed_w == 3 and inner_w < 4:
                continue
            if inner_h % seed_h == 0 and inner_w % seed_w == 0:
                continue
            options.append((inner_h, inner_w))
    assert options
    return random.choice(options)


def _attempt_instance() -> tuple[np.ndarray, np.ndarray] | None:
    seed = _sample_seed()
    inner_h, inner_w = _sample_inner_shape(seed)
    old_h = inner_h + 2
    old_w = inner_w + 2

    gaps = [random.randint(*MARKER_GAP_RANGE) for _ in range(4)]
    top_gap, bottom_gap, left_gap, right_gap = gaps
    if len(set(gaps)) == 1:
        return None

    new_h = old_h + top_gap + bottom_gap - 2
    new_w = old_w + left_gap + right_gap - 2
    if new_h * new_w < 2 * old_h * old_w:
        return None

    height = random.randint(max(GRID_RANGE[0], new_h + 2), min(GRID_RANGE[1], 30))
    width = random.randint(max(GRID_RANGE[0], new_w + 2), min(GRID_RANGE[1], 30))
    if height < new_h + 2 or width < new_w + 2:
        return None

    new_r0 = random.randint(1, height - new_h - 1)
    new_c0 = random.randint(1, width - new_w - 1)
    new_box = (new_r0, new_c0, new_r0 + new_h - 1, new_c0 + new_w - 1)

    old_r0 = new_r0 + top_gap - 1
    old_c0 = new_c0 + left_gap - 1
    old_box = (old_r0, old_c0, old_r0 + old_h - 1, old_c0 + old_w - 1)
    _, _, old_r1, old_c1 = old_box
    _, _, new_r1, new_c1 = new_box

    controls = np.array(
        [
            (new_r0 - 1, random.randint(old_c0, old_c1)),
            (new_r1 + 1, random.randint(old_c0, old_c1)),
            (random.randint(old_r0, old_r1), new_c0 - 1),
            (random.randint(old_r0, old_r1), new_c1 + 1),
        ],
        dtype=int,
    )

    grid = np.full((height, width), BG, dtype=int)
    _draw_border(grid, old_box, FRAME)
    grid[old_r0 + 1 : old_r1, old_c0 + 1 : old_c1] = _tile(seed, inner_h, inner_w)
    grid[controls[:, 0], controls[:, 1]] = CONTROL

    expected = _render_output(grid.shape, new_box, seed, controls)
    return grid, expected


def _semantic_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    frame_box = _bbox(inp == FRAME)
    if frame_box is None:
        return False
    new_box = _new_box_from_controls(frame_box, np.argwhere(inp == CONTROL), inp.shape)

    r0, c0, r1, c1 = frame_box
    interior = inp[r0 + 1 : r1, c0 + 1 : c1]
    seed = _minimal_seed(interior)
    if seed.shape == interior.shape:
        return False
    if tuple(seed.shape) not in set(SEED_SHAPES):
        return False
    if not np.any(seed == BG) or not np.any(seed == PATTERN):
        return False

    full_tile_out = _render_output(inp.shape, new_box, interior, np.argwhere(inp == CONTROL))
    if np.array_equal(full_tile_out, out):
        return False

    nr0, nc0, nr1, nc1 = new_box
    extension = np.zeros((nr1 - nr0 - 1, nc1 - nc0 - 1), dtype=bool)
    extension[:, :] = True
    extension[(r0 - nr0 - 1) : (r1 - nr0 - 1), (c0 - nc0 - 1) : (c1 - nc0 - 1)] = False
    extended_area = out[nr0 + 1 : nr1, nc0 + 1 : nc1][extension]
    if extended_area.size == 0:
        return False
    if not np.any(extended_area == BG) or not np.any(extended_area == PATTERN):
        return False

    if np.array_equal(inp, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _attempt_instance()
        if built is None:
            continue
        inp, expected = built
        if not np.array_equal(main(inp), expected):
            continue
        if not _semantic_ok(inp, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
