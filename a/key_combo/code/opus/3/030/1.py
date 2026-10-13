from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
OUTPUT_HOLE_COLOR = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

BLOCK_SIZE = 5
INTERIOR_CELLS = tuple((r, c) for r in range(1, 4) for c in range(1, 4))
HOLE_COUNTS = (1, 2, 3, 4)
MIN_GAP = 1

LAYOUTS_BY_BLOCK_COUNT = {
    4: ((2, 2), (2, 3), (3, 2)),
    5: ((2, 3), (3, 2)),
    6: ((2, 3), (3, 2)),
}

MAX_GENERATION_ATTEMPTS = 400
MIN_CANVAS_DIM = 14
CANVAS_EXTRA = 6


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    objects: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )

    return objects


def _is_contiguous(block: np.ndarray, fg: int = 1) -> bool:
    return len(_components_from_mask(block == fg)) == 1


def _build_pattern_catalog() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    catalog: dict[int, list[tuple[tuple[int, int], ...]]] = {count: [] for count in HOLE_COUNTS}
    for hole_count in HOLE_COUNTS:
        for cells in combinations(INTERIOR_CELLS, hole_count):
            block = np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
            for r, c in cells:
                block[r, c] = BG
            if _is_contiguous(block, fg=1):
                catalog[hole_count].append(tuple(sorted(cells)))
    return catalog


PATTERN_CATALOG = _build_pattern_catalog()


def _make_block(fg: int, holes: tuple[tuple[int, int], ...]) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(fg), dtype=int)
    for r, c in holes:
        block[r, c] = BG
    return block


def _analyze_blocks(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _get_bg(grid)
    objects = _find_components(grid, bg)
    assert 4 <= len(objects) <= 6, "expected 4-6 foreground blocks"

    colors = {int(obj["color"]) for obj in objects}
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = next(iter(colors))
    assert fg != OUTPUT_HOLE_COLOR, "input foreground color must differ from the output hole color"

    size: tuple[int, int] | None = None
    analyzed: list[dict] = []
    for obj in objects:
        crop = _crop_bbox(grid, obj["bbox"])
        h, w = map(int, crop.shape)
        assert h == w and h >= 3, "expected square blocks"
        if size is None:
            size = (h, w)
        else:
            assert size == (h, w), "all blocks must share one size"
        assert np.all((crop == bg) | (crop == fg)), "blocks may contain only background holes"
        assert np.all(crop[0, :] == fg)
        assert np.all(crop[-1, :] == fg)
        assert np.all(crop[:, 0] == fg)
        assert np.all(crop[:, -1] == fg)

        hole_count = int(np.count_nonzero(crop == bg))
        info = dict(obj)
        info["crop"] = crop
        info["hole_count"] = hole_count
        analyzed.append(info)

    return bg, fg, analyzed


def _invert_crop(crop: np.ndarray, bg: int) -> np.ndarray:
    return np.where(crop == bg, OUTPUT_HOLE_COLOR, bg).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert in place the one same-colored block whose number of black hole pixels is unique.

    Input structure:
    - The background is black.
    - Exactly one non-background color forms 4-6 disconnected square blocks of the same size.
    - Every block has a solid foreground border and may contain black hole pixels only inside its bounding box.

    Rule:
    - Find each foreground block and crop it to its inclusive bounding box.
    - Count the black pixels inside every cropped block bbox.
    - Select the unique block whose black-pixel count occurs exactly once.
    - Replace only that block bbox in place: its old foreground pixels become black, and its old black hole pixels become red.
    - Leave all other blocks and the exterior background unchanged.

    Key insight:
    - The internal hole layout does double duty: its pixel count identifies the outlier block, and the same hole mask becomes the red negative after the in-place inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, _fg, blocks = _analyze_blocks(g)
    hole_counts = [int(block["hole_count"]) for block in blocks]
    frequencies = Counter(hole_counts)
    unique_counts = [count for count, freq in frequencies.items() if freq == 1]
    assert len(unique_counts) == 1, "expected exactly one outlier hole count"

    target_count = unique_counts[0]
    target_blocks = [block for block in blocks if int(block["hole_count"]) == target_count]
    assert len(target_blocks) == 1, "expected exactly one target block"
    target = target_blocks[0]

    out = g.copy()
    r0, c0, r1, c1 = target["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = _invert_crop(target["crop"], bg)
    return out


def _axis_positions(total: int, slots: int) -> list[int] | None:
    base = BLOCK_SIZE * slots + MIN_GAP * (slots - 1)
    if base > total:
        return None

    extras = [0] * (slots + 1)
    for _ in range(total - base):
        extras[random.randrange(slots + 1)] += 1

    positions: list[int] = []
    cursor = extras[0]
    for idx in range(slots):
        positions.append(cursor)
        if idx + 1 < slots:
            cursor += BLOCK_SIZE + MIN_GAP + extras[idx + 1]
    return positions


def _choose_slots(rows: int, cols: int, block_count: int) -> list[tuple[int, int]]:
    slot_ids = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    for _ in range(60):
        chosen = random.sample(slot_ids, block_count)
        row_ids = {rr for rr, _ in chosen}
        col_ids = {cc for _, cc in chosen}
        if len(row_ids) == rows and len(col_ids) == cols:
            return chosen
    return random.sample(slot_ids, block_count)


def _sample_layout(block_count: int) -> tuple[int, int]:
    return random.choice(LAYOUTS_BY_BLOCK_COUNT[block_count])


def _sample_canvas(block_count: int) -> tuple[int, int, list[tuple[int, int]]]:
    rows, cols = _sample_layout(block_count)
    min_height = rows * BLOCK_SIZE + (rows - 1) * MIN_GAP
    min_width = cols * BLOCK_SIZE + (cols - 1) * MIN_GAP
    height_lo = max(min_height, MIN_CANVAS_DIM)
    width_lo = max(min_width, MIN_CANVAS_DIM)
    height = random.randint(height_lo, min(30, max(height_lo, min_height + CANVAS_EXTRA)))
    width = random.randint(width_lo, min(30, max(width_lo, min_width + CANVAS_EXTRA)))

    row_positions = _axis_positions(height, rows)
    col_positions = _axis_positions(width, cols)
    assert row_positions is not None and col_positions is not None

    chosen_slots = _choose_slots(rows, cols, block_count)
    locations = [(row_positions[rr], col_positions[cc]) for rr, cc in chosen_slots]
    random.shuffle(locations)
    return height, width, locations


def _sample_patterns(block_count: int, common_count: int, unique_count: int) -> tuple[list[tuple[tuple[int, int], ...]], int]:
    common_needed = block_count - 1
    common_options = PATTERN_CATALOG[common_count]
    unique_options = PATTERN_CATALOG[unique_count]

    common_patterns = [random.choice(common_options) for _ in range(common_needed)]
    if common_needed >= 2 and len(common_options) >= 2:
        forced = random.sample(common_options, 2)
        common_patterns[0] = forced[0]
        common_patterns[1] = forced[1]
    random.shuffle(common_patterns)

    unique_index = random.randrange(block_count)
    all_patterns: list[tuple[tuple[int, int], ...]] = []
    common_iter = iter(common_patterns)
    unique_pattern = random.choice(unique_options)
    for idx in range(block_count):
        all_patterns.append(unique_pattern if idx == unique_index else next(common_iter))
    return all_patterns, unique_index


def _expected_output(grid: np.ndarray, target_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = target_bbox
    crop = _crop_bbox(out, target_bbox)
    out[r0 : r1 + 1, c0 : c1 + 1] = _invert_crop(crop, BG)
    return out


def _semantic_ok(
    grid: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    common_count: int,
    unique_count: int,
) -> bool:
    try:
        bg, fg, blocks = _analyze_blocks(grid)
    except AssertionError:
        return False

    if bg != BG or fg == OUTPUT_HOLE_COLOR or OUTPUT_HOLE_COLOR in np.unique(grid):
        return False

    counts = [int(block["hole_count"]) for block in blocks]
    frequencies = Counter(counts)
    if len(frequencies) != 2:
        return False
    if frequencies.get(common_count) != len(blocks) - 1:
        return False
    if frequencies.get(unique_count) != 1:
        return False

    target_blocks = [block for block in blocks if block["bbox"] == target_bbox]
    if len(target_blocks) != 1 or int(target_blocks[0]["hole_count"]) != unique_count:
        return False

    common_signatures = {
        tuple(sorted(map(tuple, np.argwhere(block["crop"] == BG))))
        for block in blocks
        if int(block["hole_count"]) == common_count
    }
    if len(common_signatures) < 2:
        return False

    expected = _expected_output(grid, target_bbox)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    block_count = random.randint(4, 6)
    common_count = random.choice(HOLE_COUNTS)
    unique_count = random.choice([count for count in HOLE_COUNTS if count != common_count])
    fg = random.choice(INPUT_COLORS)

    height, width, locations = _sample_canvas(block_count)
    patterns, unique_index = _sample_patterns(block_count, common_count, unique_count)

    grid = np.zeros((height, width), dtype=int)
    target_bbox: tuple[int, int, int, int] | None = None
    for idx, ((r0, c0), holes) in enumerate(zip(locations, patterns)):
        block = _make_block(fg, holes)
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
        if idx == unique_index:
            target_bbox = (r0, c0, r0 + BLOCK_SIZE - 1, c0 + BLOCK_SIZE - 1)

    assert target_bbox is not None
    return grid if _semantic_ok(grid, target_bbox, common_count, unique_count) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
