from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

ORDERED_PAIR_COLORS = (RED, BLUE, GREEN, MAGENTA, ORANGE, MAROON)
PAIR_COUNT_CHOICES = (2, 3)
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (24, 34)
SHAPE_SIDE_RANGE = (3, 5)
COMMON_SIZE_MAX = 4
EXTRA_SIZE_MAX = 3
MAX_GEN_ATTEMPTS = 400

_CROSS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    result = []
    if r > 0:
        result.append((r - 1, c))
    if r + 1 < h:
        result.append((r + 1, c))
    if c > 0:
        result.append((r, c - 1))
    if c + 1 < w:
        result.append((r, c + 1))
    return result


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_sort_key(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _bbox_from_coords(coords)
    return (r0, c0)


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_yellow_cross(coords: np.ndarray) -> bool:
    if coords.shape[0] != 5:
        return False
    mask = _coords_to_mask(coords)
    return mask.shape == (3, 3) and np.array_equal(mask, _CROSS_MASK)


def _find_anchor_center(grid: np.ndarray) -> tuple[int, int]:
    comps = [comp for comp in _color_components(grid, YELLOW) if _is_yellow_cross(comp)]
    assert len(comps) == 1, "expected exactly one yellow cross anchor"
    r0, c0, _r1, _c1 = _bbox_from_coords(comps[0])
    return (r0 + 1, c0 + 1)


def _merge_masks(mask_a: np.ndarray, mask_b: np.ndarray, color: int) -> np.ndarray:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1])
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    a[: mask_a.shape[0], : mask_a.shape[1]] = mask_a
    b[: mask_b.shape[0], : mask_b.shape[1]] = mask_b

    out = np.full((h, w), BLACK, dtype=int)
    both = a & b
    only_a = a & ~b
    only_b = b & ~a
    out[only_a] = GREY
    out[only_b] = AZURE
    out[both] = color
    return out


def _build_strip(panels: list[np.ndarray]) -> np.ndarray:
    assert panels
    h = max(panel.shape[0] for panel in panels)
    w = sum(panel.shape[1] for panel in panels) + len(panels) - 1
    strip = np.full((h, w), BLACK, dtype=int)
    col = 0
    for panel in panels:
        ph, pw = panel.shape
        row = h - ph
        patch = panel != BLACK
        strip[row : row + ph, col : col + pw][patch] = panel[patch]
        col += pw + 1
    return strip


def _strip_top_left(anchor_center: tuple[int, int], strip_shape: tuple[int, int]) -> tuple[int, int]:
    anchor_r, anchor_c = anchor_center
    bottom = anchor_r - 2
    left = anchor_c + 2
    top = bottom - strip_shape[0] + 1
    return (top, left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, overlay their cropped masks, and stamp the merged strip near the yellow cross.

    The scene is deterministic only if the two fragments in a same-color pair are ordered. This implementation
    uses reading order: the upper-left fragment is A and the other is B. The merged panels are packed left-to-right
    in the fixed color order red, blue, green, magenta, orange, maroon, bottom-aligned in one strip whose lower-left
    corner sits one cell above and one cell to the right of the cross's bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    allowed = set(ORDERED_PAIR_COLORS) | {BLACK, YELLOW}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected input colors"

    anchor_center = _find_anchor_center(g)
    panels = []
    for color in ORDERED_PAIR_COLORS:
        comps = _color_components(g, color)
        if not comps:
            continue
        assert len(comps) == 2, "each fragment color must appear in exactly two components"
        comps.sort(key=_component_sort_key)
        mask_a = _coords_to_mask(comps[0])
        mask_b = _coords_to_mask(comps[1])
        panels.append(_merge_masks(mask_a, mask_b, color))

    assert panels, "expected at least one colored pair"
    strip = _build_strip(panels)
    top, left = _strip_top_left(anchor_center, strip.shape)
    sh, sw = strip.shape
    assert top >= 0 and left >= 0 and top + sh <= g.shape[0] and left + sw <= g.shape[1], "strip must fit"

    out = g.copy()
    patch = strip != BLACK
    out[top : top + sh, left : left + sw][patch] = strip[patch]
    return out


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _grow_connected(
    h: int,
    w: int,
    size: int,
    rng: random.Random,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    forbidden = forbidden or set()
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    if len(available) < size or size <= 0:
        return None

    start = rng.choice(available)
    cells = {start}
    frontier = {nb for nb in _neighbors(start[0], start[1], h, w) if nb not in forbidden}
    while len(cells) < size:
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors(cell[0], cell[1], h, w):
            if nb not in cells and nb not in forbidden:
                frontier.add(nb)
    return cells


def _grow_attached(
    h: int,
    w: int,
    base: set[tuple[int, int]],
    size: int,
    rng: random.Random,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    forbidden = forbidden or set()
    cells: set[tuple[int, int]] = set()
    frontier = {
        nb
        for r, c in base
        for nb in _neighbors(r, c, h, w)
        if nb not in base and nb not in forbidden
    }
    while len(cells) < size:
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors(cell[0], cell[1], h, w):
            if nb not in base and nb not in cells and nb not in forbidden:
                frontier.add(nb)
    return cells


def _apply_transform(mask: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return mask.copy()
    if code == 1:
        return np.rot90(mask, 1)
    if code == 2:
        return np.rot90(mask, 2)
    if code == 3:
        return np.rot90(mask, 3)
    if code == 4:
        return np.flipud(mask)
    if code == 5:
        return np.fliplr(mask)
    if code == 6:
        return mask.T
    if code == 7:
        return np.fliplr(mask.T)
    raise ValueError(code)


def _generate_pair_masks(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        h = rng.randint(SHAPE_SIDE_RANGE[0], SHAPE_SIDE_RANGE[1])
        w = rng.randint(SHAPE_SIDE_RANGE[0], SHAPE_SIDE_RANGE[1])
        total = h * w

        common_max = min(COMMON_SIZE_MAX, total - 2)
        if common_max < 1:
            continue
        common_size = rng.randint(1, common_max)

        a_max = min(EXTRA_SIZE_MAX, total - common_size - 1)
        if a_max < 1:
            continue
        a_size = rng.randint(1, a_max)

        b_max = min(EXTRA_SIZE_MAX, total - common_size - a_size)
        if b_max < 1:
            continue
        b_size = rng.randint(1, b_max)

        common = _grow_connected(h, w, common_size, rng)
        if common is None:
            continue
        a_only = _grow_attached(h, w, common, a_size, rng)
        if a_only is None:
            continue
        b_only = _grow_attached(h, w, common, b_size, rng, forbidden=a_only)
        if b_only is None:
            continue

        mask_a = _mask_from_cells(common | a_only, h, w)
        mask_b = _mask_from_cells(common | b_only, h, w)
        code = rng.randrange(8)
        mask_a = _crop_mask(_apply_transform(mask_a, code))
        mask_b = _crop_mask(_apply_transform(mask_b, code))
        if np.array_equal(mask_a, mask_b):
            continue

        panel = _merge_masks(mask_a, mask_b, RED)
        if not np.any(panel == RED):
            continue
        if not np.any(panel == GREY) or not np.any(panel == AZURE):
            continue
        return (mask_a, mask_b)
    return None


def _mark_blocked(
    blocked: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
    pad: int = 1,
) -> None:
    h, w = shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _valid_positions(mask: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    gh, gw = blocked.shape
    pos = []
    for top in range(gh - h + 1):
        for left in range(gw - w + 1):
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(gh, top + h + 1)
            c1 = min(gw, left + w + 1)
            if not blocked[r0:r1, c0:c1].any():
                pos.append((top, left))
    return pos


def _place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    rng: random.Random,
) -> bool:
    candidates_a = _valid_positions(mask_a, blocked)
    rng.shuffle(candidates_a)
    for top_a, left_a in candidates_a:
        temp = blocked.copy()
        _mark_blocked(temp, top_a, left_a, mask_a.shape)
        candidates_b = [pos for pos in _valid_positions(mask_b, temp) if pos > (top_a, left_a)]
        if not candidates_b:
            continue
        top_b, left_b = rng.choice(candidates_b)
        _draw_mask(grid, mask_a, top_a, left_a, color)
        _draw_mask(grid, mask_b, top_b, left_b, color)
        _mark_blocked(blocked, top_a, left_a, mask_a.shape)
        _mark_blocked(blocked, top_b, left_b, mask_b.shape)
        return True
    return False


def _draw_cross(grid: np.ndarray, center: tuple[int, int]) -> None:
    cr, cc = center
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        grid[cr + dr, cc + dc] = YELLOW


def _validate_input(grid: np.ndarray) -> bool:
    try:
        colors = [color for color in ORDERED_PAIR_COLORS if np.any(grid == color)]
        if len(colors) not in PAIR_COUNT_CHOICES:
            return False
        _find_anchor_center(grid)
        for color in colors:
            comps = _color_components(grid, color)
            if len(comps) != 2:
                return False
            comps.sort(key=_component_sort_key)
            panel = _merge_masks(_coords_to_mask(comps[0]), _coords_to_mask(comps[1]), color)
            if not np.any(panel == color):
                return False
            if not np.any(panel == GREY) or not np.any(panel == AZURE):
                return False
        out = main(grid)
        if out.shape != grid.shape:
            return False
        if not np.array_equal(out[grid != BLACK], grid[grid != BLACK]):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    colors = rng.sample(list(ORDERED_PAIR_COLORS), pair_count)

    pair_infos = []
    for color in colors:
        masks = _generate_pair_masks(rng)
        if masks is None:
            return None
        mask_a, mask_b = masks
        pair_infos.append(
            {
                "color": color,
                "mask_a": mask_a,
                "mask_b": mask_b,
                "panel": _merge_masks(mask_a, mask_b, color),
            }
        )

    ordered_panels = [
        info["panel"]
        for color in ORDERED_PAIR_COLORS
        for info in pair_infos
        if int(info["color"]) == color
    ]
    strip = _build_strip(ordered_panels)

    min_h = max(GRID_H_RANGE[0], strip.shape[0] + 8)
    min_w = max(GRID_W_RANGE[0], strip.shape[1] + 6)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    anchor_center = (h - 3, 2)
    strip_top, strip_left = _strip_top_left(anchor_center, strip.shape)
    if strip_top < 1 or strip_left < 1 or strip_top + strip.shape[0] > h - 1 or strip_left + strip.shape[1] > w - 1:
        return None

    _mark_blocked(blocked, strip_top, strip_left, strip.shape)
    _mark_blocked(blocked, anchor_center[0] - 1, anchor_center[1] - 1, (3, 3))

    pair_infos.sort(
        key=lambda info: -(
            int(np.count_nonzero(info["mask_a"])) + int(np.count_nonzero(info["mask_b"]))
        )
    )
    for info in pair_infos:
        if not _place_pair(grid, blocked, int(info["color"]), info["mask_a"], info["mask_b"], rng):
            return None

    _draw_cross(grid, anchor_center)
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
