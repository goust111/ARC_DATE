# Combo variant of dbff022c (index 19)
# combo_spec: D1/1+D13/0
# Name: 色值排序映射 + 面积决定填充色

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original dbff022c exposes a 2xN / Nx2 edge palette that directly says
#    border-color -> fill-color. Here that palette is removed entirely.
#    Example with three hollow sprites:
#      border colors (also the only non-bg colors in the input): [2, 5, 8]
#      left-to-right placement follows that same hidden rank order
#      hole areas by sprite: [3, 1, 2]
#      sorted areas = [1, 2, 3]
#    For the middle sprite (border 5, area 1):
#      border_rank = 1, area_rank = 0, fill_rank = (1 + 0 + 1) mod 3 = 2
#      so its hole is filled by color 8.
#    The output is therefore no longer readable from a local legend; it needs
#    both the color-order cue and the area cue.
# b) Complexity calibration:
#    Keep exactly three fillable sprites and exactly three area classes {1,2,3}.
#    That is enough to show repeated structure in one sample without creating a
#    large ambiguous lookup table. The mapping is cyclic, so every sample uses
#    multiple different outcomes and neither factor collapses to a constant.
# c) Broken assumptions from the original solver:
#    - Palette search, orientation inference, and palette masking are invalid.
#    - Filling can no longer use a direct border->fill dictionary.
#    - The new solver must first collect all enclosed holes, then derive two
#      latent ranks: border-color rank and area rank.
# d) Placement legality:
#    The generator uses output-first construction with three ring sprites placed
#    on one row with at least one bg column between neighbors. Holes stay away
#    from the outer grid boundary, so each intended hole is a valid enclosed bg
#    component and components never merge.
# e) Visual observability:
#    Pure numeric color ordering would be too hidden, so generation aligns the
#    sprites' left-to-right order with the same color ranking used by the
#    solver. The human-visible arrangement supports the latent rank relation.
# f) Visual cleanliness:
#    Only separated hollow rectangles are used; there is no palette residue, no
#    overlapping markers, and hole sizes 1/2/3 are easy to compare by eye.
# g) Non-degeneracy:
#    Area assignments are shuffled across the three ranked colors, so the same
#    border-rank does not always receive the same fill and the same area does
#    not always map to the same fill. This prevents reduction to a border-only
#    or area-only rule.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color on the outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _collect_holes(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    """Collect enclosed bg components together with their border color and area."""
    holes: list[dict[str, object]] = []
    for pts in components_mask(g == bg, connectivity=4):
        touches_boundary = any(
            r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in pts
        )
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    v = int(g[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border_color = max(sorted(adj_counts), key=lambda col: adj_counts[col])
        holes.append(
            {
                "coords": pts,
                "area": int(len(pts)),
                "border": int(border_color),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow sprite by combining a hidden color order with a hidden size order.

    Input structure:
    - bg is the dominant boundary color.
    - Several separated hollow sprites appear without any explicit palette or legend.
    - Every fillable hole is an enclosed bg region whose surrounding border has one solid non-bg color, and hole sizes fall into a few small visible area classes.

    Rule:
    - Find all enclosed bg holes and record two properties for each one: the surrounding border color and the hole area.
    - Sort the participating non-bg border colors by color value to get border ranks, and sort the observed hole areas from smallest to largest to get area ranks.
    - If a hole has border-rank i and area-rank j among n ranked border colors, fill it with the border color at rank (i + j + 1) mod n.

    Key insight: There is no longer a direct palette lookup. The output color is produced by composing two latent signals at once: color-order position and area-order position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    holes = _collect_holes(g, bg)
    if not holes:
        return g.copy()

    border_colors = sorted({int(hole["border"]) for hole in holes})
    area_values = sorted({int(hole["area"]) for hole in holes})
    border_rank = {color: idx for idx, color in enumerate(border_colors)}
    area_rank = {area: idx for idx, area in enumerate(area_values)}

    out = g.copy()
    n = len(border_colors)
    for hole in holes:
        i = border_rank[int(hole["border"])]
        j = area_rank[int(hole["area"])]
        fill = int(border_colors[(i + j + 1) % n])
        for r, c in hole["coords"]:
            out[int(r), int(c)] = fill
    return out


# --------------- generator (combo) ---------------

_GRID_H_MIN, _GRID_H_MAX = 5, 10
_GRID_W_CHOICES = (14, 15)
_N_SPRITES = 3
_AREA_VALUES = (1, 2, 3)
_MAX_RETRIES = 400


def _ring_with_hole_area(area: int) -> np.ndarray:
    """3 x (area+2) rectangular ring with a 1 x area interior hole."""
    w = int(area) + 2
    mask = np.ones((3, w), dtype=bool)
    mask[1, 1:-1] = False
    return mask


def _enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return enclosed bg components inside a boolean sprite mask."""
    h, w = sprite.shape
    holes = []
    for pts in components_mask(~sprite, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border_color: int,
    fill_color: int | None,
) -> None:
    """Draw one sprite; optionally pre-fill its hole cells."""
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = int(border_color)
    if fill_color is None:
        return
    for pts in _enclosed_holes(mask):
        for rr, cc in pts:
            grid[r0 + int(rr), c0 + int(cc)] = int(fill_color)


def _build_layout(
    rng: random.Random, widths: list[int], H: int, W: int
) -> tuple[int, list[int]] | None:
    """Place three sprites on a single row with 1-cell gaps."""
    total_w = sum(widths) + (_N_SPRITES - 1)
    if total_w > W:
        return None
    row = rng.randint(1, H - 4)
    start_c = rng.randint(0, W - total_w)
    cols = [start_c]
    for w in widths[:-1]:
        cols.append(cols[-1] + w + 1)
    return row, cols


def _expected_output(
    border_colors: list[int], area_by_rank: list[int]
) -> dict[int, int]:
    """Return border_color -> fill_color under the fused cyclic rule."""
    sorted_areas = sorted(area_by_rank)
    area_rank = {area: idx for idx, area in enumerate(sorted_areas)}
    mapping: dict[int, int] = {}
    n = len(border_colors)
    for i, border_color in enumerate(border_colors):
        j = area_rank[area_by_rank[i]]
        mapping[int(border_color)] = int(border_colors[(i + j + 1) % n])
    return mapping


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.choice(_GRID_W_CHOICES)

    border_colors = sorted(rng.sample(list(range(1, 10)), _N_SPRITES))
    area_by_rank = rng.sample(list(_AREA_VALUES), _N_SPRITES)
    sprites = [_ring_with_hole_area(area) for area in area_by_rank]
    widths = [sprite.shape[1] for sprite in sprites]
    layout = _build_layout(rng, widths, H, W)
    if layout is None:
        return None
    row, cols = layout

    output = np.full((H, W), bg, dtype=int)
    mapping = _expected_output(border_colors, area_by_rank)
    placements: list[tuple[int, int, np.ndarray, int, int]] = []
    for border_color, sprite, col, area in zip(
        border_colors, sprites, cols, area_by_rank, strict=False
    ):
        fill_color = mapping[int(border_color)]
        _paint_sprite(output, row, col, sprite, border_color, fill_color)
        placements.append((row, col, sprite, border_color, area))

    inp = output.copy()
    for r0, c0, sprite, _border_color, _area in placements:
        for pts in _enclosed_holes(sprite):
            for rr, cc in pts:
                inp[r0 + int(rr), c0 + int(cc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
