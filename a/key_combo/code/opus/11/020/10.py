import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

LENGTH_TO_CW_TURNS = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    turns = int(turns) % 4
    return np.rot90(grid, k=(-turns) % 4)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _read_lengths(lower: np.ndarray) -> list[int]:
    lengths: list[int] = []
    for row in lower:
        colored = np.flatnonzero(row == YELLOW)
        other = np.flatnonzero((row != BLACK) & (row != YELLOW))
        assert other.size == 0, "instruction area may only contain black or yellow"
        if colored.size == 0:
            continue
        expected = np.arange(int(colored[-1]) + 1)
        assert colored[0] == 0, "each instruction segment must start at the left edge"
        assert np.array_equal(colored, expected), "each instruction segment must be contiguous"
        seg_len = int(colored.size)
        assert 1 <= seg_len <= 4, "instruction length must be between 1 and 4"
        lengths.append(seg_len)
    assert 2 <= len(lengths) <= 4, "expected 2-4 instruction rows"
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode rotation angles from increasing segment lengths and concatenate the rotated sprite copies.

    Input structure:
    - The grid has a black background and one full-width grey separator row.
    - Above the separator sits a single red 4-connected sprite on black.
    - Below the separator are 2-4 yellow horizontal segments, one per non-empty row, with blank black rows allowed between them.

    Rule:
    - Split the grid at the grey separator row.
    - Crop the red sprite in the upper region to its tight bounding box.
    - Read each yellow segment length from top to bottom and map 1/2/3/4 to 0/90/180/270 degrees clockwise.
    - Rotate the cropped sprite for each instruction and place the copies left-to-right with one black column between them, top-aligned.

    Key insight:
    - The yellow segment lengths are not decorative bars; their lengths encode rotation commands while their vertical order defines the concatenation order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert sep_rows.size == 1, "expected exactly one full-width grey separator row"
    sep_r = int(sep_rows[0])
    assert 0 < sep_r < g.shape[0] - 1, "separator must split upper and lower regions"

    upper = g[:sep_r]
    lower = g[sep_r + 1 :]

    sprite_mask = upper == RED
    sprite_box = _bbox(sprite_mask)
    assert sprite_box is not None, "upper region must contain a red sprite"
    sprite = _crop_bbox(upper, sprite_box)

    lengths = _read_lengths(lower)
    variants = [_rotate_cw(sprite, LENGTH_TO_CW_TURNS[length]) for length in lengths]

    out_h = max(int(variant.shape[0]) for variant in variants)
    out_w = sum(int(variant.shape[1]) for variant in variants) + len(variants) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    c0 = 0
    for variant in variants:
        h, w = map(int, variant.shape)
        patch = out[:h, c0 : c0 + w]
        mask = variant != BLACK
        patch[mask] = variant[mask]
        c0 += w + 1
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    return _crop_bbox(mask.astype(int), box).astype(bool)


def _is_rotation_distinct(mask: np.ndarray) -> bool:
    norm = _normalize_mask(mask)
    for turns in (1, 2, 3):
        if np.array_equal(norm, _rotate_cw(norm.astype(int), turns).astype(bool)):
            return False
    return True


def _random_sprite() -> np.ndarray:
    for _ in range(300):
        base_h, base_w = [(3, 3), (3, 4), (4, 3)][int(_RNG.integers(0, 3))]
        area = int(_RNG.integers(4, 7))
        start = (int(_RNG.integers(0, base_h)), int(_RNG.integers(0, base_w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < base_h and 0 <= nc < base_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(_RNG.integers(0, len(frontier)))])
        mask = np.zeros((base_h, base_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)
        if not (4 <= int(mask.sum()) <= 6):
            continue
        if min(mask.shape) < 2 or max(mask.shape) < 3:
            continue
        if not _mask_connected(mask):
            continue
        if not _is_rotation_distinct(mask):
            continue
        sprite = np.full(mask.shape, BLACK, dtype=int)
        sprite[mask] = RED
        return _rotate_cw(sprite, int(_RNG.integers(0, 4)))
    raise RuntimeError("failed to build a valid asymmetric sprite")


def _sample_lengths(count: int) -> list[int]:
    assert 2 <= count <= 4
    if count == 4 and _RNG.random() < 0.55:
        lengths = [1, 2, 3, 4]
        _RNG.shuffle(lengths)
        return [int(x) for x in lengths]

    for _ in range(100):
        mode = _RNG.random()
        if mode < 0.45:
            lengths = _RNG.choice(np.array([1, 2, 3, 4]), size=count, replace=False).tolist()
        elif mode < 0.8:
            lengths = _RNG.integers(1, 5, size=count).tolist()
        else:
            repeated = int(_RNG.integers(1, 5))
            lengths = [repeated] * count
            swap_count = int(_RNG.integers(1, count))
            for idx in _RNG.choice(np.arange(count), size=swap_count, replace=False):
                lengths[int(idx)] = int(_RNG.integers(1, 5))
        lengths = [int(x) for x in lengths]
        if len(set(lengths)) >= 2:
            return lengths
    raise RuntimeError("failed to sample diverse instruction lengths")


def _valid_instruction_layout(rows: list[int]) -> bool:
    return all(rows[i + 1] - rows[i] >= 2 for i in range(len(rows) - 1))


def generate_input() -> np.ndarray:
    for _ in range(300):
        sprite = _random_sprite()
        sprite_h, sprite_w = map(int, sprite.shape)

        instr_count = int(_RNG.integers(2, 5))
        lengths = _sample_lengths(instr_count)

        top_h = int(sprite_h + _RNG.integers(1, 4))
        lower_lead = int(_RNG.integers(0, 2))
        lower_tail = int(_RNG.integers(0, 3))
        lower_h = lower_lead + (2 * instr_count - 1) + lower_tail
        height = top_h + 1 + lower_h

        min_width = max(sprite_w + 2, 8)
        width = int(_RNG.integers(min_width, max(min_width, 16) + 1))
        if height > 30 or width > 30:
            continue

        grid = np.full((height, width), BLACK, dtype=int)

        max_r0 = top_h - sprite_h
        max_c0 = width - sprite_w
        sprite_r0 = int(_RNG.integers(0, max_r0 + 1))
        sprite_c0 = int(_RNG.integers(0, max_c0 + 1))
        sprite_patch = grid[sprite_r0 : sprite_r0 + sprite_h, sprite_c0 : sprite_c0 + sprite_w]
        sprite_mask = sprite != BLACK
        sprite_patch[sprite_mask] = sprite[sprite_mask]

        sep_r = top_h
        grid[sep_r, :] = GREY

        instr_rows = [sep_r + 1 + lower_lead + 2 * i for i in range(instr_count)]
        if not _valid_instruction_layout(instr_rows):
            continue
        for row, length in zip(instr_rows, lengths):
            grid[row, :length] = YELLOW

        if np.any(grid[:sep_r] == YELLOW) or np.any(grid[sep_r + 1 :] == RED):
            continue

        out = main(grid)
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if len(set(lengths)) < 2:
            continue
        if np.array_equal(out, _rotate_cw(out, 2)):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
