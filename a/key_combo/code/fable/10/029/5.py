from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


MARKER_COLOR = 2
PLACEHOLDER_FILL = 1
BG_COLOR = 0
OUTPUT_SHAPE = (7, 7)
SPRITE_SHAPE = (3, 3)
SLOT_BY_COUNT = {
    1: (0, 0),
    2: (0, 4),
    3: (4, 0),
    4: (4, 4),
}
INPUT_SIZE_RANGE = (16, 24)
MAX_GENERATION_ATTEMPTS = 400
FILL_CHOICES = (1, 3, 4, 5, 6, 7, 8, 9)
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox(arr), "area": int(arr.shape[0])})
    return components


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                node = (nr, nc)
                if node not in visited:
                    visited.add(node)
                    stack.append(node)
    return len(visited) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _recolor_fill(crop: np.ndarray, fill_color: int) -> np.ndarray:
    out = crop.copy()
    out[out == PLACEHOLDER_FILL] = int(fill_color)
    return out


def _sprite_signature(crop: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in crop)


def _rotate_for_count(crop: np.ndarray, count: int) -> np.ndarray:
    return np.rot90(crop, -int(count - 1)).copy()


def _build_output(crops_by_count: dict[int, np.ndarray], bg: int, *, rotate: bool) -> np.ndarray:
    out = np.full(OUTPUT_SHAPE, bg, dtype=int)
    for count in range(1, 5):
        block = crops_by_count[count]
        if rotate:
            block = _rotate_for_count(block, count)
        r0, c0 = SLOT_BY_COUNT[count]
        out[r0 : r0 + 3, c0 : c0 + 3] = block
    return out


def _build_output_by_position(components: list[dict], grid: np.ndarray, bg: int) -> np.ndarray:
    ordered = sorted(components, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    out = np.full(OUTPUT_SHAPE, bg, dtype=int)
    for slot_idx, comp in enumerate(ordered, start=1):
        crop = _crop_bbox(grid, comp["bbox"])
        count = int(np.count_nonzero(crop == MARKER_COLOR))
        block = _rotate_for_count(crop, count)
        r0, c0 = SLOT_BY_COUNT[slot_idx]
        out[r0 : r0 + 3, c0 : c0 + 3] = block
    return out


def _valid_colored_sprite(crop: np.ndarray, count: int) -> bool:
    mask = crop != BG_COLOR
    if crop.shape != SPRITE_SHAPE:
        return False
    if not _is_connected(mask):
        return False
    if not _touches_all_sides(mask):
        return False
    if int(np.count_nonzero(crop == MARKER_COLOR)) != count:
        return False
    if int(np.count_nonzero(crop == PLACEHOLDER_FILL)) < 1:
        return False
    if count > 1 and np.array_equal(crop, _rotate_for_count(crop, count)):
        return False
    return True


def _make_sprite_bank() -> dict[int, list[np.ndarray]]:
    banks: dict[int, list[np.ndarray]] = {count: [] for count in range(1, 5)}
    cells = range(9)
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in cells], dtype=int).reshape(SPRITE_SHAPE).astype(bool)
        fg = int(mask.sum())
        if fg < 5 or fg > 8:
            continue
        if not _touches_all_sides(mask) or not _is_connected(mask):
            continue
        fg_cells = [idx for idx in cells if mask.reshape(-1)[idx]]
        for count in range(1, 5):
            if fg <= count:
                continue
            for red_cells in itertools.combinations(fg_cells, count):
                crop = np.full(SPRITE_SHAPE, BG_COLOR, dtype=int)
                crop[mask] = PLACEHOLDER_FILL
                flat = crop.reshape(-1)
                for idx in red_cells:
                    flat[idx] = MARKER_COLOR
                crop = flat.reshape(SPRITE_SHAPE)
                if _valid_colored_sprite(crop, count):
                    banks[count].append(crop.copy())
    return banks


SPRITE_BANK = _make_sprite_bank()
SPRITE_BANK_WITH_HOLE = {
    count: [crop for crop in bank if int(crop[1, 1]) == BG_COLOR]
    for count, bank in SPRITE_BANK.items()
}
DERANGEMENTS = [
    perm
    for perm in itertools.permutations((1, 2, 3, 4))
    if any(perm[i] != i + 1 for i in range(4))
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the four marked sprites into a 2x2 atlas addressed by marker counts.

    Input structure:
    - The grid has a dominant background color, one shared fill color, and red marker pixels.
    - Exactly four separated 8-connected multicolor sprites are present, each with a 3x3 bounding box.
    - Each sprite keeps its original holes, and the four red-marker counts are exactly 1, 2, 3, and 4.

    Rule:
    - Crop each sprite to its 3x3 bounding box.
    - If a crop contains k red markers, rotate that crop clockwise by (k-1) quarter-turns.
    - Paste the rotated crop into the kth slot of a 7x7 2x2 atlas in reading order, with one background row and column between blocks.

    Key insight:
    - The red-marker count is a hidden key used twice: it chooses both the crop's atlas address and its rotation amount, so input spatial order is irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) == 4

    fg_colors = set(map(int, np.unique(g))) - {bg}
    assert MARKER_COLOR in fg_colors and len(fg_colors) == 2

    crops_by_count: dict[int, np.ndarray] = {}
    for comp in components:
        box = comp["bbox"]
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1, c1 - c0 + 1) == SPRITE_SHAPE
        crop = _crop_bbox(g, box)
        count = int(np.count_nonzero(crop == MARKER_COLOR))
        assert count in {1, 2, 3, 4} and count not in crops_by_count
        assert _valid_colored_sprite(np.where(crop == bg, BG_COLOR, np.where(crop == MARKER_COLOR, MARKER_COLOR, PLACEHOLDER_FILL)), count)
        crops_by_count[count] = crop

    assert set(crops_by_count) == {1, 2, 3, 4}
    return _build_output(crops_by_count, bg, rotate=True)


def _choose_crops() -> dict[int, np.ndarray]:
    force_hole_count = int(RNG.integers(1, 5))
    crops = {}
    used_raw = set()
    used_rotated = set()

    for count in range(1, 5):
        if count == force_hole_count and SPRITE_BANK_WITH_HOLE[count]:
            bank = SPRITE_BANK_WITH_HOLE[count]
        else:
            bank = SPRITE_BANK[count]
        choices = RNG.permutation(len(bank))
        picked = None
        for idx in choices:
            crop = bank[int(idx)]
            raw_sig = _sprite_signature(crop)
            rotated_sig = _sprite_signature(_rotate_for_count(crop, count))
            if raw_sig in used_raw or rotated_sig in used_rotated:
                continue
            picked = crop.copy()
            used_raw.add(raw_sig)
            used_rotated.add(rotated_sig)
            break
        if picked is None:
            raise RuntimeError("failed to choose a non-degenerate sprite set")
        crops[count] = picked
    return crops


def _sample_positions(height: int, width: int) -> list[tuple[int, int]]:
    sites = [(r, c) for r in range(height - 2) for c in range(width - 2)]
    RNG.shuffle(sites)
    positions: list[tuple[int, int]] = []

    for r, c in sites:
        ok = True
        for rr, cc in positions:
            if abs(r - rr) <= 4 and abs(c - cc) <= 4:
                ok = False
                break
        if ok:
            positions.append((r, c))
        if len(positions) == 4:
            return positions
    raise RuntimeError("failed to place four separated sprites")


def _render_input(crops_by_count: dict[int, np.ndarray], fill_color: int, height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    positions = sorted(_sample_positions(height, width))
    permutation = DERANGEMENTS[int(RNG.integers(len(DERANGEMENTS)))]
    for position, count in zip(positions, permutation):
        r0, c0 = position
        grid[r0 : r0 + 3, c0 : c0 + 3] = _recolor_fill(crops_by_count[count], fill_color)
    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = _get_bg(grid)
    components = _find_components(grid, bg)
    if len(components) != 4:
        return False

    counts = sorted(int(np.count_nonzero(_crop_bbox(grid, comp["bbox"]) == MARKER_COLOR)) for comp in components)
    if counts != [1, 2, 3, 4]:
        return False

    by_position = _build_output_by_position(components, grid, bg)
    if np.array_equal(by_position, expected):
        return False

    crops_by_count = {}
    for comp in components:
        crop = _crop_bbox(grid, comp["bbox"])
        count = int(np.count_nonzero(crop == MARKER_COLOR))
        crops_by_count[count] = crop
    without_rotation = _build_output(crops_by_count, bg, rotate=False)
    if np.array_equal(without_rotation, expected):
        return False

    transformed = [_sprite_signature(_rotate_for_count(crops_by_count[count], count)) for count in range(1, 5)]
    if len(set(transformed)) != 4:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        fill_color = int(RNG.choice(FILL_CHOICES))
        crops_by_count = _choose_crops()
        height = int(RNG.integers(INPUT_SIZE_RANGE[0], INPUT_SIZE_RANGE[1] + 1))
        width = int(RNG.integers(INPUT_SIZE_RANGE[0], INPUT_SIZE_RANGE[1] + 1))
        grid = _render_input(crops_by_count, fill_color, height, width)
        expected = _build_output(
            {count: _recolor_fill(crop, fill_color) for count, crop in crops_by_count.items()},
            BG_COLOR,
            rotate=True,
        )
        if _validate_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid counting-atlas input")
