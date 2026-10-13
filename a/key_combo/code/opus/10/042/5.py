import math
from collections import Counter, deque

import numpy as np


BLACK = 0
QUAD_SIZE = 6
SEP_INDEX = QUAD_SIZE
INPUT_SIZE = QUAD_SIZE * 2 + 1
SQUARE_COUNTS = (16, 25, 36, 49, 64)
MIN_COMMON_COUNT = 3
MAX_COMMON_COUNT = 24
MAX_GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 200


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BLACK)
    if box is None:
        raise AssertionError("expected non-black pixels")
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _quadrant_slices(index: int) -> tuple[slice, slice]:
    row_slice = slice(0, QUAD_SIZE) if index < 2 else slice(SEP_INDEX + 1, INPUT_SIZE)
    col_slice = slice(0, QUAD_SIZE) if index % 2 == 0 else slice(SEP_INDEX + 1, INPUT_SIZE)
    return row_slice, col_slice


def _split_quadrants(grid: np.ndarray, sep_r: int, sep_c: int) -> list[np.ndarray]:
    return [
        grid[:sep_r, :sep_c],
        grid[:sep_r, sep_c + 1 :],
        grid[sep_r + 1 :, :sep_c],
        grid[sep_r + 1 :, sep_c + 1 :],
    ]


def _quadrant_info(quadrant: np.ndarray) -> dict | None:
    colors = [int(color) for color in np.unique(quadrant) if int(color) != BLACK]
    if len(colors) != 1:
        return None
    count = int(np.count_nonzero(quadrant))
    if count == 0:
        return None
    return {
        "color": colors[0],
        "count": count,
        "crop": _crop_nonblack(quadrant),
    }


def _find_partition(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == BLACK)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == BLACK)]
    matches = []

    for sep_r in sep_rows:
        for sep_c in sep_cols:
            if sep_r <= 0 or sep_c <= 0 or sep_r >= g.shape[0] - 1 or sep_c >= g.shape[1] - 1:
                continue
            quadrants = _split_quadrants(g, sep_r, sep_c)
            if any(quadrant.size == 0 for quadrant in quadrants):
                continue
            infos = [_quadrant_info(quadrant) for quadrant in quadrants]
            if any(info is None for info in infos):
                continue

            color_counts = Counter(info["color"] for info in infos)
            if sorted(color_counts.values()) != [1, 3]:
                continue
            outlier_color = next(color for color, freq in color_counts.items() if freq == 1)
            outlier_index = next(idx for idx, info in enumerate(infos) if info["color"] == outlier_color)
            sprite = infos[outlier_index]["crop"]
            if sprite.shape != (3, 3):
                continue

            common_total = sum(info["count"] for idx, info in enumerate(infos) if idx != outlier_index)
            side = math.isqrt(common_total)
            if side * side != common_total:
                continue

            matches.append(
                {
                    "sep_r": sep_r,
                    "sep_c": sep_c,
                    "outlier_index": outlier_index,
                    "sprite": sprite,
                    "copies": common_total,
                    "side": side,
                }
            )

    if len(matches) != 1:
        raise AssertionError(f"expected exactly one valid partition, got {len(matches)}")
    return matches[0]


def _render_output(sprite: np.ndarray, copies: int, side: int) -> np.ndarray:
    sprite_h, sprite_w = sprite.shape
    out = np.zeros((sprite_h * side, sprite_w * side), dtype=int)
    if copies > side * side:
        raise AssertionError("copy count exceeds slot count")
    for index in range(copies):
        slot_r, slot_c = divmod(index, side)
        r0 = slot_r * sprite_h
        c0 = slot_c * sprite_w
        out[r0 : r0 + sprite_h, c0 : c0 + sprite_w] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the uniquely colored quadrant sprite and tile it into a square array sized by the shared-color pixel count.

    Input structure:
    - A black grid is split into four quadrants by one full black separator row and one full black separator column.
    - Each quadrant contains black background plus exactly one non-black foreground color.
    - Three quadrants share one foreground color, while the remaining quadrant uses a different color and contains a compact 3x3-bbox pattern.

    Rule:
    - Find the separator row and separator column, then inspect the four quadrants.
    - Identify the unique-color quadrant and crop its non-black pixels to their minimal bounding box to obtain the sprite.
    - Count all non-black pixels across the other three same-color quadrants; this total must be a perfect square N.
    - Build a sqrt(N) by sqrt(N) array of sprite slots and paste the sprite into the first N row-major slots, leaving any later slots black.

    Key insight:
    - The outlier quadrant tells you what to copy, while the three matching quadrants tell you how many copies and how large the square sprite array must be.
    """
    data = _find_partition(np.asarray(grid, dtype=int))
    return _render_output(data["sprite"], data["copies"], data["side"])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return len(seen) == int(mask.sum())


def _build_sprite_masks() -> list[np.ndarray]:
    masks = []
    for bits in range(1 << 9):
        count = bits.bit_count()
        if count < 4 or count > 8:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for index in range(9):
            if bits & (1 << index):
                mask[index // 3, index % 3] = True
        if _bbox(mask) != (0, 0, 2, 2):
            continue
        if not _is_connected(mask):
            continue
        masks.append(mask)
    return masks


SPRITE_MASKS = _build_sprite_masks()


def _common_count_splits(total: int) -> list[tuple[int, int, int]]:
    splits = []
    upper = min(MAX_COMMON_COUNT, total - 2 * MIN_COMMON_COUNT)
    for first in range(MIN_COMMON_COUNT, upper + 1):
        second_upper = min(MAX_COMMON_COUNT, total - first - MIN_COMMON_COUNT)
        for second in range(MIN_COMMON_COUNT, second_upper + 1):
            third = total - first - second
            if MIN_COMMON_COUNT <= third <= MAX_COMMON_COUNT:
                splits.append((first, second, third))
    return splits


def _random_common_mask(count: int, rng: np.random.Generator) -> np.ndarray | None:
    min_span = 4 if count >= 8 else 3
    max_full_rows = 2 if count <= 12 else 3
    for _ in range(MASK_ATTEMPTS):
        cells = rng.choice(QUAD_SIZE * QUAD_SIZE, size=count, replace=False)
        mask = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool)
        mask.flat[cells] = True
        row_span = int(np.count_nonzero(mask.any(axis=1)))
        col_span = int(np.count_nonzero(mask.any(axis=0)))
        if row_span < min_span or col_span < min_span:
            continue
        if int(np.count_nonzero(mask.all(axis=1))) > max_full_rows:
            continue
        if int(np.count_nonzero(mask.all(axis=0))) > max_full_rows:
            continue
        return mask
    return None


def _full_black_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLACK)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLACK)]
    return rows, cols


def _assemble_input(
    common_color: int,
    outlier_color: int,
    outlier_index: int,
    sprite_mask: np.ndarray,
    common_masks: dict[int, np.ndarray],
    outlier_offset: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=int)
    sprite_patch = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    rr, cc = outlier_offset
    sprite_patch[rr : rr + 3, cc : cc + 3] = np.where(sprite_mask, outlier_color, BLACK)

    for quadrant_index in range(4):
        row_slice, col_slice = _quadrant_slices(quadrant_index)
        if quadrant_index == outlier_index:
            grid[row_slice, col_slice] = sprite_patch
        else:
            grid[row_slice, col_slice] = np.where(common_masks[quadrant_index], common_color, BLACK)
    return grid


def _expected_from_parts(sprite_mask: np.ndarray, outlier_color: int, common_total: int) -> np.ndarray:
    side = math.isqrt(common_total)
    sprite = np.where(sprite_mask, outlier_color, BLACK).astype(int)
    return _render_output(sprite, common_total, side)


def _semantics_are_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    outlier_index: int,
    common_total: int,
) -> bool:
    rows, cols = _full_black_lines(grid)
    if rows != [SEP_INDEX] or cols != [SEP_INDEX]:
        return False
    try:
        found = _find_partition(grid)
    except AssertionError:
        return False
    if found["outlier_index"] != outlier_index:
        return False
    if found["copies"] != common_total:
        return False
    if found["sprite"].shape != (3, 3):
        return False
    actual = main(grid)
    return actual.shape == expected.shape and np.array_equal(actual, expected)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    common_color, outlier_color = rng.choice(np.arange(1, 10), size=2, replace=False)
    common_total = int(rng.choice(SQUARE_COUNTS))
    splits = _common_count_splits(common_total)
    if not splits:
        return None

    counts = list(splits[int(rng.integers(len(splits)))])
    rng.shuffle(counts)
    outlier_index = int(rng.integers(4))
    sprite_mask = SPRITE_MASKS[int(rng.integers(len(SPRITE_MASKS)))]
    outlier_offset = (int(rng.integers(0, 4)), int(rng.integers(0, 4)))

    common_slots = [index for index in range(4) if index != outlier_index]
    common_masks: dict[int, np.ndarray] = {}
    for quadrant_index, count in zip(common_slots, counts):
        mask = _random_common_mask(count, rng)
        if mask is None:
            return None
        common_masks[quadrant_index] = mask

    grid = _assemble_input(
        int(common_color),
        int(outlier_color),
        outlier_index,
        sprite_mask,
        common_masks,
        outlier_offset,
    )
    expected = _expected_from_parts(sprite_mask, int(outlier_color), common_total)
    if not _semantics_are_valid(grid, expected, outlier_index, common_total):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
