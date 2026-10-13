from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
TARGET_BORDER = 5
BLOCK_SIZE = 5
INTERIOR_SLICE = (slice(1, 4), slice(1, 4))
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
HOLE_COUNT_VALUES = [1, 2, 3, 4, 5]
MAX_GEN_ATTEMPTS = 400

INTERIOR_CELLS = [(r, c) for r in range(3) for c in range(3)]
PATTERNS_BY_COUNT = {
    count: [tuple(cells) for cells in combinations(INTERIOR_CELLS, count)]
    for count in HOLE_COUNT_VALUES
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(block: np.ndarray) -> int:
    return int(np.sum(block[INTERIOR_SLICE] == BG))


def _is_valid_block_crop(crop: np.ndarray, color: int) -> bool:
    if crop.shape != (BLOCK_SIZE, BLOCK_SIZE):
        return False
    if not np.all((crop == BG) | (crop == int(color))):
        return False
    if not np.all(crop[0, :] == int(color)):
        return False
    if not np.all(crop[-1, :] == int(color)):
        return False
    if not np.all(crop[:, 0] == int(color)):
        return False
    if not np.all(crop[:, -1] == int(color)):
        return False
    return True


def _find_target_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components_of_color(grid, TARGET_BORDER):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        actual = {tuple(map(int, rc)) for rc in comp}
        if actual != expected:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.all(interior == BG):
            continue
        if (h - 2) % BLOCK_SIZE != 0 or (w - 2) % BLOCK_SIZE != 0:
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one grey rectangular target border"
    return candidates[0]


def _find_blocks(grid: np.ndarray) -> list[dict]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, TARGET_BORDER)]
    blocks: list[dict] = []
    for color in colors:
        for comp in _components_of_color(grid, color):
            bbox = _bbox(comp)
            crop = _crop(grid, bbox)
            assert _is_valid_block_crop(crop, color), "unexpected non-block foreground object"
            holes = _count_holes(crop)
            blocks.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "crop": crop,
                    "holes": holes,
                }
            )

    assert 4 <= len(blocks) <= 6, "expected 4-6 foreground blocks"
    block_colors = {block["color"] for block in blocks}
    assert len(block_colors) == 1, "expected all blocks to share one foreground color"

    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        assert r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1, "blocks may not touch the grid edge"

    blocks.sort(key=lambda block: (block["bbox"][0], block["bbox"][1]))
    hole_freq = Counter(block["holes"] for block in blocks)
    unique_counts = [count for count, freq in hole_freq.items() if freq == 1]
    assert len(unique_counts) == 1, "expected exactly one unique hole count"
    for count, freq in hole_freq.items():
        if count != unique_counts[0]:
            assert freq >= 2, "every non-outlier hole count must repeat"
    return blocks


def _analyze(grid: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int], dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(np.max(g.shape)) <= 30

    target_bbox = _find_target_bbox(g)
    blocks = _find_blocks(g)
    hole_freq = Counter(block["holes"] for block in blocks)
    unique_hole_count = next(count for count, freq in hole_freq.items() if freq == 1)
    outlier = next(block for block in blocks if block["holes"] == unique_hole_count)

    max_block_bottom = max(block["bbox"][2] for block in blocks)
    assert max_block_bottom < target_bbox[0], "blocks must appear above the target region"

    return blocks, target_bbox, outlier


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the 5x5 block with the unique interior hole count and tile it into the grey target frame.

    Input structure:
    - Black background fills most of the grid.
    - The upper region contains 4-6 disconnected same-colored 5x5 square blocks with solid borders and 0-valued holes only inside their 3x3 interiors.
    - The lower region contains one grey rectangular frame whose black interior has height and width that are integer multiples of 5.

    Rule:
    - Count the number of black cells inside each block's inner 3x3 area.
    - Select the single block whose hole count occurs exactly once, while every other hole count is repeated by at least one partner.
    - Copy that entire 5x5 block pattern, including its border and hole layout, into every 5x5 tile slot of the grey frame's interior from left to right and top to bottom.

    Key insight:
    - The holes are not output directly as a crop; they identify one source block whose full pattern becomes the seed for a separate tiled target area.
    """
    blocks, target_bbox, outlier = _analyze(grid)
    _ = blocks

    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = target_bbox
    tile = outlier["crop"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    for rr in range(r0 + 1, r0 + 1 + inner_h, BLOCK_SIZE):
        for cc in range(c0 + 1, c0 + 1 + inner_w, BLOCK_SIZE):
            out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = tile
    return out


def _distribute_slack(slack: int, buckets: int) -> list[int]:
    parts = [0] * buckets
    for _ in range(slack):
        parts[random.randrange(buckets)] += 1
    return parts


def _row_positions(total_width: int, count: int, gap: int) -> list[int] | None:
    base = count * BLOCK_SIZE + (count - 1) * gap + 2
    if base > total_width:
        return None
    extras = _distribute_slack(total_width - base, count + 1)
    positions: list[int] = []
    cursor = 1 + extras[0]
    for idx in range(count):
        positions.append(cursor)
        if idx + 1 < count:
            cursor += BLOCK_SIZE + gap + extras[idx + 1]
    return positions


def _make_block(color: int, holes: tuple[tuple[int, int], ...]) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), color, dtype=int)
    for r, c in holes:
        block[r + 1, c + 1] = BG
    return block


def _sample_count_profile(block_count: int) -> list[int]:
    values = random.sample(HOLE_COUNT_VALUES, k=3)
    if block_count == 4:
        if random.random() < 0.5:
            common, outlier = values[0], values[1]
            counts = [common, common, common, outlier]
        else:
            a, b, outlier = values
            counts = [a, a, b, outlier]
            counts[2] = a  # keep only one true outlier
        random.shuffle(counts)
        return counts

    if block_count == 5:
        if random.random() < 0.5:
            common, outlier = values[0], values[1]
            counts = [common, common, common, common, outlier]
        else:
            a, b, outlier = values
            counts = [a, a, b, b, outlier]
        random.shuffle(counts)
        return counts

    assert block_count == 6
    if random.random() < 0.5:
        common, outlier = values[0], values[1]
        counts = [common, common, common, common, common, outlier]
    else:
        a, b, outlier = values
        if random.random() < 0.5:
            counts = [a, a, a, b, b, outlier]
        else:
            counts = [a, a, b, b, b, outlier]
    random.shuffle(counts)
    return counts


def _sample_row_counts(block_count: int) -> list[int]:
    if block_count == 4 and random.random() < 0.45:
        return [4]
    if block_count == 4:
        return [2, 2]
    if block_count == 5:
        return random.choice([[3, 2], [2, 3]])
    return [3, 3]


def _expected_from_metadata(grid: np.ndarray, target_bbox: tuple[int, int, int, int], outlier_block: np.ndarray) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = target_bbox
    for rr in range(r0 + 1, r1, BLOCK_SIZE):
        for cc in range(c0 + 1, c1, BLOCK_SIZE):
            out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = outlier_block
    return out


def _attempt_generate() -> np.ndarray | None:
    block_count = random.randint(4, 6)
    row_counts = _sample_row_counts(block_count)
    block_rows = len(row_counts)
    block_color = random.choice(BLOCK_COLORS)
    hole_counts = _sample_count_profile(block_count)

    target_tile_rows = 2 if block_rows == 2 else random.choice([2, 3, 4])
    target_tile_cols = random.randint(2, 4)
    target_h = target_tile_rows * BLOCK_SIZE + 2
    target_w = target_tile_cols * BLOCK_SIZE + 2

    row_gap = random.randint(1, 3)
    col_gap = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    gap_to_target = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    block_region_h = block_rows * BLOCK_SIZE + (block_rows - 1) * row_gap
    height = top_margin + block_region_h + gap_to_target + target_h + bottom_margin
    if height > 30:
        return None

    widest_row = max(count * BLOCK_SIZE + (count - 1) * col_gap + 2 for count in row_counts)
    min_width = max(widest_row, target_w + 2)
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)

    row_xs = []
    for count in row_counts:
        xs = _row_positions(width, count, col_gap)
        if xs is None:
            return None
        row_xs.append(xs)

    max_target_x = width - target_w - 1
    if max_target_x < 1:
        return None
    target_x = random.randint(1, max_target_x)
    target_y = top_margin + block_region_h + gap_to_target

    unique_index = next(i for i, count in enumerate(hole_counts) if Counter(hole_counts)[count] == 1)
    pattern_pool = {count: random.sample(PATTERNS_BY_COUNT[count], k=hole_counts.count(count)) for count in set(hole_counts)}
    block_specs = []
    cursor = 0
    outlier_block: np.ndarray | None = None
    for row_idx, count in enumerate(row_counts):
        y = top_margin + row_idx * (BLOCK_SIZE + row_gap)
        for x in row_xs[row_idx][:count]:
            hole_count = hole_counts[cursor]
            holes = pattern_pool[hole_count].pop()
            block = _make_block(block_color, holes)
            block_specs.append((y, x, block, hole_count))
            if cursor == unique_index:
                outlier_block = block
            cursor += 1

    assert outlier_block is not None

    grid = np.zeros((height, width), dtype=int)
    for y, x, block, _hole_count in block_specs:
        grid[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = block

    r0, c0 = target_y, target_x
    r1 = r0 + target_h - 1
    c1 = c0 + target_w - 1
    grid[r0 : r1 + 1, c0] = TARGET_BORDER
    grid[r0 : r1 + 1, c1] = TARGET_BORDER
    grid[r0, c0 : c1 + 1] = TARGET_BORDER
    grid[r1, c0 : c1 + 1] = TARGET_BORDER

    expected = _expected_from_metadata(grid, (r0, c0, r1, c1), outlier_block)

    try:
        blocks, target_bbox, detected_outlier = _analyze(grid)
    except AssertionError:
        return None

    repeated_groups = {}
    for block in blocks:
        repeated_groups.setdefault(block["holes"], []).append(block["crop"].tobytes())
    for holes, crops in repeated_groups.items():
        if len(crops) > 1 and len(set(crops)) < 2:
            return None

    if target_bbox != (r0, c0, r1, c1):
        return None
    if not np.array_equal(detected_outlier["crop"], outlier_block):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    out_unique_colors = {int(v) for v in np.unique(expected[r0 + 1 : r1, c0 + 1 : c1])}
    if block_color not in out_unique_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
