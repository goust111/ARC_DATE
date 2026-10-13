from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

GEN_RETRIES = 400

SIDE_TO_MARKER_SLOTS = {
    0: ("nw", "ne"),
    1: ("ne", "se"),
    2: ("sw", "se"),
    3: ("nw", "sw"),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shift_row_clip(row: np.ndarray, dc: int) -> np.ndarray:
    out = np.zeros_like(row)
    cols = np.flatnonzero(row == RED)
    dst = cols + int(dc)
    keep = (0 <= dst) & (dst < row.size)
    out[dst[keep]] = RED
    return out


def _shift_row_wrap(row: np.ndarray, dc: int) -> np.ndarray:
    out = np.zeros_like(row)
    cols = np.flatnonzero(row == RED)
    if row.size == 0:
        return out
    out[(cols + int(dc)) % row.size] = RED
    return out


def _find_period_and_shift(seed: np.ndarray) -> tuple[int, int]:
    h, w = map(int, seed.shape)
    for period in range(1, h):
        shifts: list[int] = []
        for dc in range(-(w - 1), w):
            if all(np.array_equal(_shift_row_clip(seed[r], dc), seed[r + period]) for r in range(h - period)):
                shifts.append(int(dc))
        if shifts:
            assert len(shifts) == 1, "expected a unique shift for the minimal period"
            return int(period), int(shifts[0])
    raise AssertionError("expected a translated row-periodic seed")


def _frame_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    box = _bbox(grid == color)
    assert box is not None
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] >= 3 and patch.shape[1] >= 3
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.all(patch[border] == color)
    assert not np.any(patch[1:-1, 1:-1] == color)
    return box


def _corner_slots(rect: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {
        "nw": (r0 - 1, c0 - 1),
        "ne": (r0 - 1, c1 + 1),
        "se": (r1 + 1, c1 + 1),
        "sw": (r1 + 1, c0 - 1),
    }


def _decode_rotation(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    markers = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)}
    assert len(markers) == 2
    slots = _corner_slots(rect)
    matches = [
        k
        for k, pair in SIDE_TO_MARKER_SLOTS.items()
        if markers == {slots[pair[0]], slots[pair[1]]}
    ]
    assert len(matches) == 1, "expected one adjacent-corner blue orientation code"
    return int(matches[0])


def _extract_seed(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0
    return np.where(inner == RED, RED, BLACK).astype(int)


def _target_size(grid: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _frame_bbox(grid, ORANGE)
    return int(r1 - r0 - 1), int(c1 - c0 - 1)


def _extend_seed(seed: np.ndarray, out_h: int, out_w: int, period: int, dc: int) -> np.ndarray:
    out = np.zeros((out_h, out_w), dtype=int)
    seed_h, seed_w = map(int, seed.shape)
    copy_h = min(out_h, seed_h)
    copy_w = min(out_w, seed_w)
    out[:copy_h, :copy_w] = seed[:copy_h, :copy_w]
    for r in range(seed_h, out_h):
        out[r] = _shift_row_wrap(out[r - period], dc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the green frame, restore the seed's orientation from the blue corner code, then extend its row-periodic pattern downward.

    Input structure:
    - The background is black.
    - One green rectangular frame contains a red seed pattern on black cells inside it, and two blue pixels sit at adjacent outer-corner diagonals of that frame.
    - A separate orange rectangular frame marks the target output size by its empty interior.

    Rule:
    - Crop the inside of the green frame and keep only the red pixels.
    - Decode the blue outer-corner pair as a quarter-turn instruction, rotate the cropped seed accordingly, and infer the minimal vertical period plus the unique horizontal shift relating rows one period apart.
    - Read the orange frame's interior size, copy the rotated seed into the top of that black canvas with width cropping if needed, then continue lower rows from the row one period above using the same horizontal shift with wraparound across the target width.

    Key insight:
    - The blue corner pair is an orientation legend: only after the indicated rotation does the cropped red pattern become a coherent translated row-periodic seed that can be extended.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    rect = _frame_bbox(g, GREEN)
    rotation = _decode_rotation(g, rect)
    seed = np.rot90(_extract_seed(g, rect), rotation)
    period, dc = _find_period_and_shift(seed)
    out_h, out_w = _target_size(g)
    return _extend_seed(seed, out_h, out_w, period, dc)


def _count_valid_rotations(seed: np.ndarray) -> int:
    total = 0
    for k in range(4):
        try:
            _find_period_and_shift(np.rot90(seed, k))
        except Exception:
            continue
        total += 1
    return total


def _seed_attempt(rng: np.random.Generator) -> np.ndarray | None:
    period = int(rng.choice(np.array([2, 3], dtype=int)))
    shift = int(rng.choice(np.array([1, 2], dtype=int)))
    h = int(rng.integers(max(2 * period + 1, 5), min(2 * period + 4, 9)))
    w = int(rng.integers(max(5, shift + 4), 9))

    base_rows: list[list[int]] = []
    top_cols = {0, w - 1}
    if w > 4 and int(rng.integers(0, 2)) == 1:
        top_cols.add(int(rng.integers(1, w - 2)))
    base_rows.append(sorted(top_cols))

    for i in range(1, period):
        max_steps = (h - 1 - i) // period
        max_col = w - 1 - max_steps * shift
        if max_col < 1:
            return None
        pool = np.arange(0, max_col + 1, dtype=int)
        count = int(rng.integers(1, min(4, pool.size + 1)))
        cols = sorted(int(x) for x in rng.choice(pool, size=count, replace=False))
        if cols == base_rows[0]:
            return None
        base_rows.append(cols)

    seed = np.zeros((h, w), dtype=int)
    for base_index, cols in enumerate(base_rows):
        step = 0
        row = base_index
        while row < h:
            shifted = [c + step * shift for c in cols if c + step * shift < w]
            if not shifted or len(shifted) > 3:
                return None
            seed[row, shifted] = RED
            row += period
            step += 1

    if not np.all((seed == RED).sum(axis=1) >= 1):
        return None
    if seed[0, 0] != RED or seed[0, -1] != RED:
        return None
    if seed[-1, 0] == RED or seed[-1, -1] == RED:
        return None
    if len({tuple(row.tolist()) for row in seed}) <= period:
        return None

    try:
        actual_period, actual_shift = _find_period_and_shift(seed)
    except Exception:
        return None
    if (actual_period, actual_shift) != (period, shift):
        return None
    if _count_valid_rotations(seed) != 1:
        return None
    return seed


def _generate_seed(rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        seed = _seed_attempt(rng)
        if seed is not None:
            return seed
    raise RuntimeError("failed to generate a non-degenerate periodic seed")


def _choose_target_width(seed_w: int, rng: np.random.Generator) -> int:
    modes = ["smaller", "same", "larger"]
    mode = str(rng.choice(np.array(modes, dtype=object)))
    if mode == "smaller" and seed_w >= 6:
        return int(rng.integers(max(3, seed_w - 2), seed_w))
    if mode == "larger":
        return int(rng.integers(seed_w + 1, min(11, seed_w + 4)))
    return int(seed_w)


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _marker_positions(rect: tuple[int, int, int, int], rotation: int) -> list[tuple[int, int]]:
    slots = _corner_slots(rect)
    return [slots[name] for name in SIDE_TO_MARKER_SLOTS[int(rotation)]]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    seed = _generate_seed(rng)
    period, shift = _find_period_and_shift(seed)

    correction = int(rng.integers(0, 4))
    raw_seed = np.rot90(seed, -correction)
    raw_h, raw_w = map(int, raw_seed.shape)

    out_h = int(rng.integers(seed.shape[0] + 1, min(11, seed.shape[0] + 5)))
    out_w = _choose_target_width(seed.shape[1], rng)

    green_h, green_w = raw_h + 2, raw_w + 2
    orange_h, orange_w = out_h + 2, out_w + 2

    rect_r = int(rng.integers(1, 3))
    rect_c = int(rng.integers(1, 3))
    orange_r = rect_r + green_h + int(rng.integers(2, 4))
    orange_c = rect_c + green_w + int(rng.integers(2, 4))

    rows = orange_r + orange_h + int(rng.integers(1, 3))
    cols = orange_c + orange_w + int(rng.integers(1, 3))
    if max(rows, cols) > 30:
        return None

    grid = np.zeros((rows, cols), dtype=int)

    _draw_frame(grid, rect_r, rect_c, green_h, green_w, GREEN)
    interior = grid[rect_r + 1 : rect_r + 1 + raw_h, rect_c + 1 : rect_c + 1 + raw_w]
    interior[raw_seed == RED] = RED

    rect = (rect_r, rect_c, rect_r + green_h - 1, rect_c + green_w - 1)
    for mr, mc in _marker_positions(rect, correction):
        if not (0 <= mr < rows and 0 <= mc < cols):
            return None
        grid[mr, mc] = BLUE

    _draw_frame(grid, orange_r, orange_c, orange_h, orange_w, ORANGE)

    expected = _extend_seed(seed, out_h, out_w, period, shift)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, np.zeros_like(actual)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
