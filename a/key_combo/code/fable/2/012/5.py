from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
COLORS = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (14, 20)
OBJECT_COUNT_RANGE = (3, 5)
INSTANCE_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80
ACTIVE_BAND_WIDTH_RANGE = (5, 7)


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


BASE_SHAPES = (
    _mask(["##", "#."]),
    _mask(["##", ".#"]),
    _mask(["###", ".#."]),
    _mask(["#.", "##", "#."]),
    _mask([".##", "##."]),
    _mask(["###", "#.."]),
    _mask(["###", "..#"]),
    _mask(["##", "##", "#."]),
    _mask(["##", "##", ".#"]),
    _mask(["###", ".##"]),
)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask
        for k in range(4):
            cand = np.rot90(cur, k)
            if not any(np.array_equal(cand, prev) for prev in variants):
                variants.append(cand)
    return variants


def _bottom_profile(mask: np.ndarray) -> tuple[int, ...]:
    profile = []
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0
        profile.append(int(rows.max()))
    return tuple(profile)


INTERESTING_SHAPES = [
    orient
    for base in BASE_SHAPES
    for orient in _all_orientations(base)
    if orient.shape[1] >= 2 and len(set(_bottom_profile(orient))) >= 2
]


_ROTATION_BAG: list[int] = []


@dataclass(frozen=True)
class Sprite:
    color: int
    patch: np.ndarray
    bbox: tuple[int, int, int, int]


def _next_rotation() -> int:
    global _ROTATION_BAG
    if not _ROTATION_BAG:
        _ROTATION_BAG = [0, 1, 2, 3]
        random.shuffle(_ROTATION_BAG)
    return _ROTATION_BAG.pop()


def _shape_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _fits(shape: tuple[int, int], patch_shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = map(int, patch_shape)
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def _can_place(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    if not _fits(out.shape, patch.shape, r0, c0):
        return False
    mask = patch != bg
    region = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    return bool(np.all(region[mask] == bg))


def _direction_delta(side: str) -> tuple[int, int]:
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _simulate(
    shape: tuple[int, int],
    legend_patch: np.ndarray,
    sprites: dict[int, Sprite],
    order: list[int],
    side: str,
    bg: int = BG,
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    out = np.full(shape, bg, dtype=int)
    paste(out, legend_patch, 0, 0, transparent=bg)
    placements: dict[int, tuple[int, int, int, int]] = {}
    dr, dc = _direction_delta(side)

    for color in order:
        sprite = sprites[color]
        r, c = sprite.bbox[0], sprite.bbox[1]
        if not _can_place(out, sprite.patch, r, c, bg):
            raise AssertionError("sprite start overlaps the legend or a previously settled sprite")

        while True:
            nr, nc = r + dr, c + dc
            if not _can_place(out, sprite.patch, nr, nc, bg):
                break
            r, c = nr, nc

        paste(out, sprite.patch, r, c, transparent=bg)
        h, w = map(int, sprite.patch.shape)
        placements[color] = (r, c, r + h - 1, c + w - 1)

    return out, placements


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[str, list[int], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    all_colors = {int(color) for color in np.unique(g) if int(color) != bg}
    assert all_colors, "expected at least one foreground color"

    candidates = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        coords = comp["coords"]
        box = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = box
        colors = [int(g[r, c]) for r, c in coords]
        if len(set(colors)) != len(colors) or set(colors) != all_colors:
            continue

        if r0 == r1 and c0 < c1 and 0 < r0 < height - 1:
            if c0 == 0 and c1 < width - 1:
                ordered = [int(g[r0, c]) for c in range(c1, c0 - 1, -1)]
                candidates.append(("left", ordered, coords))
            elif c1 == width - 1 and c0 > 0:
                ordered = [int(g[r0, c]) for c in range(c0, c1 + 1)]
                candidates.append(("right", ordered, coords))
        elif c0 == c1 and r0 < r1 and 0 < c0 < width - 1:
            if r0 == 0 and r1 < height - 1:
                ordered = [int(g[r, c0]) for r in range(r1, r0 - 1, -1)]
                candidates.append(("up", ordered, coords))
            elif r1 == height - 1 and r0 > 0:
                ordered = [int(g[r, c0]) for r in range(r0, r1 + 1)]
                candidates.append(("down", ordered, coords))

    assert len(candidates) == 1, "expected one edge-attached legend strip"
    side, order, coords = candidates[0]
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[coords[:, 0], coords[:, 1]] = True
    legend_patch = np.zeros_like(g, dtype=int)
    legend_patch[legend_mask] = g[legend_mask]
    return side, order, legend_mask, legend_patch


def _extract_sprites(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> dict[int, Sprite]:
    g = np.asarray(grid, dtype=int).copy()
    g[legend_mask] = bg
    sprites: dict[int, Sprite] = {}
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        assert color not in sprites, "each sprite color must be unique"
        box = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, box)
        sprites[color] = Sprite(color=color, patch=patch, bbox=box)

    assert sprites, "expected at least one sprite"
    return sprites


def _analyze_grid(grid: np.ndarray) -> tuple[int, str, list[int], np.ndarray, np.ndarray, dict[int, Sprite]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    side, order, legend_mask, legend_patch = _parse_legend(g, bg)
    sprites = _extract_sprites(g, bg, legend_mask)
    assert set(order) == set(sprites), "legend colors must match sprite colors"
    assert len(order) == len(sprites), "legend must list each sprite exactly once"
    return bg, side, order, legend_mask, legend_patch, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each sprite toward the edge touched by the legend, following the legend order from its far end to its attached end.

    Input structure:
    - The grid has a black background, several monochromatic 4-connected sprites with distinct colors, and one edge-attached legend strip.
    - The legend is a single row or column whose cells list every sprite color exactly once.

    Rule:
    - Clear the canvas but keep the legend in place.
    - The border touched by the legend sets gravity: a legend attached to the bottom means downward gravity, one attached to the right means rightward gravity, and so on.
    - Process sprites in the legend order read from the free far end toward the edge-attached near end.
    - Each sprite keeps its shape and its coordinate orthogonal to gravity, then slides one step at a time along gravity until the next step would hit the border, the legend, or an already settled sprite.

    Key insight:
    - The puzzle is not ordinary physics: both the gravity direction and the settling order are encoded by the legend, so early sprites become the interlocking ground that later sprites must land on.
    """
    g = np.asarray(grid, dtype=int)
    bg, side, order, _legend_mask, legend_patch, sprites = _analyze_grid(g)
    out, _placements = _simulate(g.shape, legend_patch, sprites, order, side, bg=bg)
    return out


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = map(int, blocked.shape)
    for r, c in cells:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                blocked[nr, nc] = True


def _patch_cells(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [
        (r0 + int(dr), c0 + int(dc))
        for dr, dc in np.argwhere(patch != BG)
    ]


def _choose_band(width: int, legend_col: int, min_width: int) -> tuple[int, int] | None:
    max_width = min(ACTIVE_BAND_WIDTH_RANGE[1], width - 3)
    if max_width < min_width:
        return None

    band_widths = list(range(min_width, max_width + 1))
    random.shuffle(band_widths)
    for band_width in band_widths:
        starts = list(range(1, width - band_width))
        random.shuffle(starts)
        for start in starts:
            end = start + band_width - 1
            if legend_col < start - 1 or legend_col > end + 1:
                return start, end
    return None


def _place_sprites_base(
    height: int,
    width: int,
    colors: list[int],
    patches: list[np.ndarray],
    legend_col: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, Sprite], list[int]] | None:
    top_limit = height // 2
    if top_limit < 4:
        return None

    min_band_width = max(patch.shape[1] for patch in patches) + 1
    band = _choose_band(width, legend_col, min_band_width)
    if band is None:
        return None
    band_start, band_end = band

    legend_order = colors[:]
    random.shuffle(legend_order)
    legend_top = height - len(legend_order)
    if legend_top <= top_limit:
        return None

    grid = np.zeros((height, width), dtype=int)
    legend_patch = np.zeros_like(grid)
    legend_cells = []
    for idx, color in enumerate(legend_order):
        rr = legend_top + idx
        legend_patch[rr, legend_col] = int(color)
        legend_cells.append((rr, legend_col))
    paste(grid, legend_patch, 0, 0, transparent=BG)

    blocked = np.zeros((height, width), dtype=bool)
    _mark_blocked(blocked, legend_cells)

    sprites: dict[int, Sprite] = {}
    placement_order = list(zip(colors, patches))
    random.shuffle(placement_order)
    for color, patch in placement_order:
        h, w = map(int, patch.shape)
        row_max = top_limit - h
        if row_max < 1:
            return None

        starts = list(range(band_start, band_end - w + 2))
        rows = list(range(1, row_max + 1))
        random.shuffle(starts)
        random.shuffle(rows)
        placed = False
        for c0 in starts:
            for r0 in rows:
                cells = _patch_cells(patch, r0, c0)
                if any(blocked[r, c] for r, c in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = int(color)
                _mark_blocked(blocked, cells)
                sprites[int(color)] = Sprite(
                    color=int(color),
                    patch=patch.copy(),
                    bbox=(r0, c0, r0 + h - 1, c0 + w - 1),
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    out, placements = _simulate(grid.shape, legend_patch, sprites, legend_order, side="down", bg=BG)
    deltas = [placements[color][0] - sprites[color].bbox[0] for color in legend_order]
    if min(deltas) <= 0 or sum(delta >= 2 for delta in deltas) < 2:
        return None

    return grid, out, sprites, legend_order


def _physical_order(side: str, sprites: dict[int, Sprite]) -> list[int]:
    if side == "down":
        key = lambda item: (-item[1].bbox[2], item[1].bbox[0], item[1].bbox[1], item[0])
    elif side == "up":
        key = lambda item: (item[1].bbox[0], item[1].bbox[1], item[1].bbox[2], item[0])
    elif side == "left":
        key = lambda item: (item[1].bbox[1], item[1].bbox[0], item[1].bbox[3], item[0])
    else:
        key = lambda item: (-item[1].bbox[3], item[1].bbox[0], item[1].bbox[1], item[0])
    return [color for color, _sprite in sorted(sprites.items(), key=key)]


def _alt_orders(correct: list[int], side: str, sprites: dict[int, Sprite]) -> list[list[int]]:
    alts: list[list[int]] = []
    candidates = [
        list(reversed(correct)),
        _physical_order(side, sprites),
    ]
    near = _physical_order(side, sprites)
    candidates.append(list(reversed(near)))
    if len(correct) >= 2:
        swapped = correct[:]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        candidates.append(swapped)
    if len(correct) >= 3:
        swapped_tail = correct[:]
        swapped_tail[-2], swapped_tail[-1] = swapped_tail[-1], swapped_tail[-2]
        candidates.append(swapped_tail)

    seen = set()
    for cand in candidates:
        key = tuple(cand)
        if cand != correct and key not in seen:
            seen.add(key)
            alts.append(cand)
    return alts


def _semantic_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, side, order, _legend_mask, legend_patch, sprites = _analyze_grid(grid)
    except AssertionError:
        return False

    if order == _physical_order(side, sprites):
        return False

    try:
        actual, _ = _simulate(grid.shape, legend_patch, sprites, order, side, bg=bg)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    different_valid_alt = False
    physical_alt_diff = False
    for alt in _alt_orders(order, side, sprites):
        try:
            alt_out, _ = _simulate(grid.shape, legend_patch, sprites, alt, side, bg=bg)
        except AssertionError:
            continue
        if not np.array_equal(alt_out, expected):
            different_valid_alt = True
            if alt == _physical_order(side, sprites):
                physical_alt_diff = True

    return different_valid_alt and physical_alt_diff


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLORS, count)
    patches = [_shape_to_patch(random.choice(INTERESTING_SHAPES), color) for color in colors]

    legend_col = random.randint(1, width - 2)
    base = _place_sprites_base(height, width, colors, patches, legend_col)
    if base is None:
        return None
    base_grid, base_out, _sprites, _legend_order = base

    k = _next_rotation()
    grid = np.rot90(base_grid, k).copy()
    expected = np.rot90(base_out, k).copy()

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_ok(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(INSTANCE_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid edge-legend drop puzzle")
