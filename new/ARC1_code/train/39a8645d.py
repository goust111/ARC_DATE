from collections import defaultdict

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, label_mask, paste
from itertools import combinations
import random


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, _r1, _c1 = component["bbox"]
    h = bbox_h(component["bbox"])
    w = bbox_w(component["bbox"])
    assert (h, w) == (3, 3)

    mask = np.zeros((h, w), dtype=bool)
    rel = component["coords"] - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 monochrome sprite type that appears most often in the input.

    Input structure
    The grid has a dominant background and several small monochrome foreground sprites.
    Each sprite is an 8-connected component whose bounding box is 3x3.

    Rule
    Normalize each sprite to its 3x3 bounding-box mask, keeping its color as part of
    the sprite identity. Count identical colored sprite types and output one copy of
    the uniquely most frequent type on the background.

    Key insight
    The output is not a crop from a particular location; it is the repeated motif
    selected by copy count.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert components

    groups: dict[tuple[int, tuple[tuple[int, ...], ...]], list[dict]] = defaultdict(list)
    masks: dict[tuple[int, tuple[tuple[int, ...], ...]], np.ndarray] = {}
    for component in components:
        color = int(component["color"])
        mask = _component_mask(component)
        pattern = tuple(tuple(int(x) for x in row) for row in mask)
        key = (color, pattern)
        groups[key].append(component)
        masks[key] = mask

    max_count = max(len(group) for group in groups.values())
    winners = [key for key, group in groups.items() if len(group) == max_count]
    assert len(winners) == 1

    color, _pattern = winners[0]
    mask = masks[winners[0]]
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas, sprite geometry, and palette.
_GEN_GRID_SHAPE = (14, 14)
_GEN_SPRITE_SHAPE = (3, 3)
_GEN_TOP_LEFT_ROWS = range(_GEN_GRID_SHAPE[0] - _GEN_SPRITE_SHAPE[0] + 1)
_GEN_TOP_LEFT_COLS = range(_GEN_GRID_SHAPE[1] - _GEN_SPRITE_SHAPE[1] + 1)
_GEN_BACKGROUND = 0
_GEN_COLORS = range(1, 10)

# Sprite shape constraints.
_GEN_CELL_RANGE = (4, 6)
_GEN_CONNECTIVITY = 8
_GEN_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

# Type count distribution and retry budget.
_GEN_COUNT_OPTIONS_BY_TOTAL = {
    3: ((2, 1),),
    4: ((2, 1, 1),),
    5: ((3, 2),),
    6: ((3, 2, 1),),
    7: ((4, 3), (3, 2, 2)),
    8: ((4, 3, 1),),
}
_GEN_RETRY_LIMIT = 300


def _gen_valid_masks_by_cells() -> dict[int, list[np.ndarray]]:
    pools: dict[int, list[np.ndarray]] = {cells: [] for cells in range(_GEN_CELL_RANGE[0], _GEN_CELL_RANGE[1] + 1)}
    for cells in range(_GEN_CELL_RANGE[0], _GEN_CELL_RANGE[1] + 1):
        for flat_coords in combinations(range(_GEN_SPRITE_SHAPE[0] * _GEN_SPRITE_SHAPE[1]), cells):
            mask = np.zeros(_GEN_SPRITE_SHAPE, dtype=bool)
            for k in flat_coords:
                mask[k // _GEN_SPRITE_SHAPE[1], k % _GEN_SPRITE_SHAPE[1]] = True
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            _, n_components = label_mask(mask, connectivity=_GEN_CONNECTIVITY)
            if n_components == 1:
                pools[cells].append(mask)
    return pools


_GEN_MASKS_BY_CELLS = _gen_valid_masks_by_cells()


def _gen_pattern_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask)


def _gen_pick_counts() -> list[int]:
    total = random.randint(min(_GEN_COUNT_OPTIONS_BY_TOTAL), max(_GEN_COUNT_OPTIONS_BY_TOTAL))
    counts = list(random.choice(_GEN_COUNT_OPTIONS_BY_TOTAL[total]))
    random.shuffle(counts)
    return counts


def _gen_pick_masks(k: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    used: set[tuple[tuple[int, ...], ...]] = set()
    for _ in range(k):
        cells = random.randint(*_GEN_CELL_RANGE)
        choices = [m for m in _GEN_MASKS_BY_CELLS[cells] if _gen_pattern_key(m) not in used]
        mask = random.choice(choices).copy()
        used.add(_gen_pattern_key(mask))
        masks.append(mask)
    return masks


def _gen_can_place(grid: np.ndarray, coords: np.ndarray, color: int, r0: int, c0: int) -> bool:
    rows = r0 + coords[:, 0]
    cols = c0 + coords[:, 1]
    if np.any(grid[rows, cols] != _GEN_BACKGROUND):
        return False

    height, width = grid.shape
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        for nr_delta, nc_delta in _GEN_NEIGHBORS8:
            nr = r + nr_delta
            nc = c + nc_delta
            if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) == int(color):
                return False
    return True


def _gen_valid_positions(grid: np.ndarray, mask: np.ndarray, color: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [
        (r, c)
        for r in _GEN_TOP_LEFT_ROWS
        for c in _GEN_TOP_LEFT_COLS
        if _gen_can_place(grid, coords, color, r, c)
    ]


def _gen_render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(_GEN_SPRITE_SHAPE, _GEN_BACKGROUND, dtype=int)
    sprite[mask] = color
    return sprite


def _gen_place_instances(grid: np.ndarray, masks: list[np.ndarray], colors: list[int], counts: list[int]) -> bool:
    """Place all shuffled sprite instances, returning False if any has no legal site."""
    instances = [idx for idx, count in enumerate(counts) for _ in range(count)]
    random.shuffle(instances)
    for idx in instances:
        positions = _gen_valid_positions(grid, masks[idx], colors[idx])
        if not positions:
            return False
        r, c = random.choice(positions)
        paste(grid, _gen_render_sprite(masks[idx], colors[idx]), r, c, transparent=_GEN_BACKGROUND)
    return True


def _gen_matches_oracle(grid: np.ndarray, masks: list[np.ndarray], colors: list[int], counts: list[int]) -> bool:
    winner_idx = max(range(len(counts)), key=lambda idx: counts[idx])
    expected = _gen_render_sprite(masks[winner_idx], colors[winner_idx])
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    counts = _gen_pick_counts()
    k = len(counts)
    colors = random.sample(_GEN_COLORS, k)
    masks = _gen_pick_masks(k)

    grid = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    if not _gen_place_instances(grid, masks, colors, counts):
        return None
    if _gen_matches_oracle(grid, masks, colors, counts):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 39a8645d input")
