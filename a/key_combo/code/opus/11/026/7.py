import numpy as np


BACKGROUND = 0
DIVIDER = 5
SHAPE_COLORS = (1, 2, 3, 7)

# Quadrant / slot order is row-major: 0=TL, 1=TR, 2=BL, 3=BR.
PATTERNS = (
    ((0, 0), (0, 1)),  # domino h
    ((0, 0), (1, 0)),  # domino v
    ((0, 0), (0, 1), (1, 0)),  # L3
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # 2x2 square
    ((0, 0), (0, 1), (0, 2)),  # 3-line h
    ((0, 0), (1, 0), (2, 0)),  # 3-line v
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L4
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # T4
    ((1, 0), (1, 1), (1, 2), (0, 1)),
    ((0, 1), (1, 1), (2, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # zig4
    ((1, 0), (1, 1), (0, 1), (0, 2)),
)


def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups = [[indices[0]]]
    for value in indices[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _slot_bounds(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[int], list[int]]:
    row_divs = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    col_divs = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    row_groups = _consecutive_groups(row_divs)
    col_groups = _consecutive_groups(col_divs)
    assert len(row_groups) == 1 and len(col_groups) == 1

    row_band = row_groups[0]
    col_band = col_groups[0]
    top_rows = (0, row_band[0] - 1)
    bot_rows = (row_band[-1] + 1, grid.shape[0] - 1)
    left_cols = (0, col_band[0] - 1)
    right_cols = (col_band[-1] + 1, grid.shape[1] - 1)
    assert top_rows[0] <= top_rows[1] < bot_rows[0] <= bot_rows[1]
    assert left_cols[0] <= left_cols[1] < right_cols[0] <= right_cols[1]

    slot_h = top_rows[1] - top_rows[0] + 1
    slot_w = left_cols[1] - left_cols[0] + 1
    assert bot_rows[1] - bot_rows[0] + 1 == slot_h
    assert right_cols[1] - right_cols[0] + 1 == slot_w

    slots = [
        (top_rows[0], top_rows[1], left_cols[0], left_cols[1]),
        (top_rows[0], top_rows[1], right_cols[0], right_cols[1]),
        (bot_rows[0], bot_rows[1], left_cols[0], left_cols[1]),
        (bot_rows[0], bot_rows[1], right_cols[0], right_cols[1]),
    ]
    return slots, row_divs, col_divs


def _component_coords(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    seen = {tuple(coords[0])}
    stack = [tuple(coords[0])]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
    assert len(seen) == len(coords)
    return np.array(sorted(seen), dtype=int)


def _crop_from_coords(subgrid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return subgrid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _quadrant_from_coords(coords: np.ndarray, slot_shape: tuple[int, int]) -> int:
    center_r = (slot_shape[0] - 1) / 2.0
    center_c = (slot_shape[1] - 1) / 2.0
    rows = coords[:, 0]
    cols = coords[:, 1]

    if np.all(rows < center_r):
        vertical = 0
    elif np.all(rows > center_r):
        vertical = 1
    else:
        raise AssertionError("shape is not fully in one vertical half")

    if np.all(cols < center_c):
        horizontal = 0
    elif np.all(cols > center_c):
        horizontal = 1
    else:
        raise AssertionError("shape is not fully in one horizontal half")

    return vertical * 2 + horizontal


def _place_centered(out: np.ndarray, slot: tuple[int, int, int, int], sprite: np.ndarray) -> None:
    r0, r1, c0, c1 = slot
    slot_h = r1 - r0 + 1
    slot_w = c1 - c0 + 1
    sh, sw = sprite.shape
    start_r = r0 + (slot_h - sh) // 2
    start_c = c0 + (slot_w - sw) // 2
    mask = sprite != BACKGROUND
    out[start_r : start_r + sh, start_c : start_c + sw][mask] = sprite[mask]


def _extract_slot_shape(subgrid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    colors = sorted(int(x) for x in np.unique(subgrid) if x not in (BACKGROUND, DIVIDER))
    assert len(colors) == 1
    color = colors[0]
    mask = subgrid == color
    coords = _component_coords(mask)
    assert 2 <= len(coords) <= 4
    sprite = _crop_from_coords(subgrid, coords)
    return color, coords, sprite


def _identity_output(grid: np.ndarray) -> np.ndarray:
    slots, row_divs, col_divs = _slot_bounds(grid)
    out = np.full_like(grid, BACKGROUND)
    out[row_divs, :] = DIVIDER
    out[:, col_divs] = DIVIDER
    for slot in slots:
        r0, r1, c0, c1 = slot
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        _color, _coords, sprite = _extract_slot_shape(sub)
        _place_centered(out, slot, sprite)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange slot-contained shapes by decoding each shape's quadrant as a destination slot.

    Input structure:
    - A gray divider row and gray divider column split the grid into four equal slots.
    - Each slot contains exactly one small 4-connected monochrome shape on black background.
    - Every shape lies entirely inside one quadrant of its slot, and the four quadrant choices are all different.

    Rule:
    - Number the slots 0,1,2,3 in row-major order.
    - For each slot, detect the contained shape and determine whether it is in the slot's top-left, top-right, bottom-left, or bottom-right quadrant.
    - Map those quadrants to destination slots 0,1,2,3 in the same row-major order, clear the old slots, keep the gray dividers, and place each cropped shape centered in its destination slot.

    Key insight:
    - The position of a shape inside its source slot is not decorative: it encodes where that entire shape must be moved.
    """
    g = np.asarray(grid, dtype=int)
    slots, row_divs, col_divs = _slot_bounds(g)
    out = np.full_like(g, BACKGROUND)
    out[row_divs, :] = DIVIDER
    out[:, col_divs] = DIVIDER

    targets = []
    sprites = []
    for slot in slots:
        r0, r1, c0, c1 = slot
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        _color, coords, sprite = _extract_slot_shape(sub)
        target = _quadrant_from_coords(coords, sub.shape)
        targets.append(target)
        sprites.append(sprite)

    assert sorted(targets) == [0, 1, 2, 3]
    for target, sprite in zip(targets, sprites):
        _place_centered(out, slots[target], sprite)
    return out


def _pattern_to_sprite(pattern: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    coords = np.array(pattern, dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    sprite = np.zeros((h, w), dtype=int)
    sprite[coords[:, 0], coords[:, 1]] = color
    return sprite


def _compatible_patterns(quadrant_size: int) -> list[tuple[tuple[int, int], ...]]:
    options = []
    for pattern in PATTERNS:
        coords = np.array(pattern, dtype=int)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h <= quadrant_size and w <= quadrant_size:
            options.append(pattern)
    return options


def _quadrant_anchor(slot_size: int, quadrant: int) -> tuple[int, int]:
    q = slot_size // 2
    top = 0 if quadrant < 2 else slot_size - q
    left = 0 if quadrant % 2 == 0 else slot_size - q
    return top, left


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, start_r: int, start_c: int) -> None:
    mask = sprite != BACKGROUND
    canvas[start_r : start_r + sprite.shape[0], start_c : start_c + sprite.shape[1]][mask] = sprite[mask]


def _random_derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


def _build_expected_from_blueprint(
    slot_size: int, row_div: int, col_div: int, target_for_source: list[int], sprites: list[np.ndarray]
) -> np.ndarray:
    grid = np.zeros((2 * slot_size + 1, 2 * slot_size + 1), dtype=int)
    grid[row_div, :] = DIVIDER
    grid[:, col_div] = DIVIDER
    slots, _, _ = _slot_bounds(grid)
    for src_idx, sprite in enumerate(sprites):
        _place_centered(grid, slots[target_for_source[src_idx]], sprite)
    return grid


def _validate_generated(inp: np.ndarray, expected: np.ndarray, target_for_source: list[int]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, _identity_output(inp)):
        return False

    slots, row_divs, col_divs = _slot_bounds(inp)
    if len(row_divs) != 1 or len(col_divs) != 1:
        return False
    decoded = []
    for idx, slot in enumerate(slots):
        r0, r1, c0, c1 = slot
        sub = inp[r0 : r1 + 1, c0 : c1 + 1]
        _color, coords, _sprite = _extract_slot_shape(sub)
        decoded.append(_quadrant_from_coords(coords, sub.shape))
        if decoded[-1] == idx:
            return False
    return decoded == list(target_for_source)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        slot_size = int(rng.choice([5, 7]))
        quadrant_size = slot_size // 2
        row_div = slot_size
        col_div = slot_size

        grid = np.zeros((2 * slot_size + 1, 2 * slot_size + 1), dtype=int)
        grid[row_div, :] = DIVIDER
        grid[:, col_div] = DIVIDER

        patterns = _compatible_patterns(quadrant_size)
        chosen_patterns = rng.choice(len(patterns), size=4, replace=False)
        colors = rng.permutation(np.array(SHAPE_COLORS, dtype=int))
        target_for_source = _random_derangement(rng, 4).tolist()
        sprites = []

        slots, _, _ = _slot_bounds(grid)
        for src_idx, pattern_idx in enumerate(chosen_patterns):
            color = int(colors[src_idx])
            sprite = _pattern_to_sprite(patterns[int(pattern_idx)], color)
            sprites.append(sprite)

            quadrant = target_for_source[src_idx]
            base_r, base_c = _quadrant_anchor(slot_size, quadrant)
            qh, qw = sprite.shape
            dr = int(rng.integers(0, quadrant_size - qh + 1))
            dc = int(rng.integers(0, quadrant_size - qw + 1))
            slot_r0, _slot_r1, slot_c0, _slot_c1 = slots[src_idx]
            _paste_sprite(grid, sprite, slot_r0 + base_r + dr, slot_c0 + base_c + dc)

        expected = _build_expected_from_blueprint(slot_size, row_div, col_div, target_for_source, sprites)
        if _validate_generated(grid, expected, target_for_source):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
