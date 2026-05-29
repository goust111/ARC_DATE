# Combo variant of d59b0160 (index 43)
# combo_spec: D6/0+D8/0
# Name: 邻居颜色填充+背景色不固定
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: infer bg=7, read the top-left legend, find every non-bg 4-connected
#      component, and replace each matching component with a flat bg-colored block.
#    - Fused task: infer bg separately for each example from the outer border, delete every
#      matching component simultaneously, then refill each deleted pixel from the nearest
#      surviving non-bg pixel. If no such survivor exists, fall back to bg.
#    - Small example: a deleted 6x6 rectangle surrounded (with a 1-cell bg gap) by a blue
#      bar above, a yellow bar below, a teal bar left, and a maroon bar right. The old rule
#      yields a uniform bg patch. The new rule yields a four-way nearest-neighbor partition.
#
# b) Complexity calibration:
#    - To make D6 visible, each generated sample contains at least one large deleted object
#      plus four nearby color beacons. The fill is therefore structured and repeatable,
#      rather than collapsing to a single flat color.
#    - The number of moving parts stays small: one legend, one deleted core object, four
#      beacon components, and two additional kept objects.
#
# c) Implicit-assumption audit:
#    - The original solver assumed "bg is fixed orange" and "delete means write bg".
#      Both are invalid now.
#    - It also relied on analyzing all non-bg components together; here the legend area is
#      explicitly excluded so the legend itself can never be mistaken for a deletable object.
#    - A naive nearest-neighbor rule that includes bg donors would degenerate to flat bg
#      whenever objects are separated by a bg moat. The fused rule therefore propagates only
#      from surviving non-bg pixels, with bg used only as a fallback.
#
# d) Placement legality:
#    - The deleted rectangle is placed only where a 2-cell margin exists for the surrounding
#      beacon bars. All other components are placed with at least a 1-cell bg gap.
#
# e) Visual observability:
#    - Across 2-4 examples, a human can see that the same "contains all legend colors"
#      objects disappear, but the vacated area is not blanked out: it inherits nearby colors.
#      The variable bg forces bg inference from context instead of from a fixed color id.
#
# f) Visual cleanliness:
#    - Beacons are thin bars on the four sides of the deleted object, so the output pattern
#      is localized and readable instead of cluttered by many overlapping structures.
#
# g) New solver assumptions guaranteed by the generator:
#    - bg is never 0 and never the legend border color 3, so 0 can still serve as the solid
#      body color inside objects.
#    - At least one deleted component exists and has nearby non-bg donor beacons, ensuring
#      the fused behavior is exercised rather than silently reducing to the original task.

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


GRID_H, GRID_W = 16, 16
LEGEND_SIZE = 4
LEGEND_BORDER = 3
BODY_COLOR = 0
TARGET_POOL = [1, 2, 4, 5, 6, 8, 9]
BG_CHOICES = [1, 2, 4, 5, 6, 7, 8, 9]

DELETED_H = (5, 8)
DELETED_W = (5, 8)
KEPT_H = (3, 6)
KEPT_W = (3, 7)
COLOR_DENSITY_LO = 0.14
COLOR_DENSITY_HI = 0.40
MAX_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    The legend is the top-left square whose bottom edge and right edge are the same color.
    Return (size, border_color, interior_target_colors).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, :sz] == x) and np.all(g[:sz, sz - 1] == x):
            inside = g[: sz - 1, : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    raise AssertionError("no legend border found")


def _nearest_survivor_fill(g: np.ndarray, delete_mask: np.ndarray, bg: int) -> np.ndarray:
    """
    Replace each deleted cell by the nearest surviving non-bg pixel color.
    Ties are broken by donor coordinate order from np.argwhere (row-major).
    """
    out = g.copy()
    if not bool(delete_mask.any()):
        return out

    donor_coords = np.argwhere((~delete_mask) & (g != bg))
    if donor_coords.size == 0:
        out[delete_mask] = bg
        return out

    donor_colors = g[donor_coords[:, 0], donor_coords[:, 1]].astype(int)
    deleted_coords = np.argwhere(delete_mask)

    for r, c in deleted_coords:
        dists = np.abs(donor_coords[:, 0] - r) + np.abs(donor_coords[:, 1] - c)
        out[r, c] = int(donor_colors[int(dists.argmin())])

    return out


def _delete_mask_from_legend(
    g: np.ndarray,
    bg: int,
    legend_sz: int,
    legend_colors: set[int],
) -> np.ndarray:
    """Mask of non-legend components whose color set contains all legend colors."""
    obj_mask = (g != bg)
    obj_mask[:legend_sz, :legend_sz] = False

    delete_mask = np.zeros_like(obj_mask, dtype=bool)
    for pts in components_mask(obj_mask, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            delete_mask[pts[:, 0], pts[:, 1]] = True
    return delete_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete legend-matching components and refill them from nearby surviving colors.

    Input structure:
    - One color acts as the background, but it is not fixed across examples; infer it from the outer frame.
    - The top-left corner contains the same 4x4 legend format as the original task: a green bottom/right border with target colors in the interior.
    - Elsewhere, 4-connected non-bg objects are built from solid color-0 bodies with sprinkled accent colors.

    Rule:
    - Read the target color set from the legend interior.
    - Ignore the legend itself, then find all other non-bg 4-connected components.
    - Delete every component whose color set contains all legend colors, but do it simultaneously.
    - Refill each deleted cell with the color of the nearest surviving non-bg pixel; if no such donor exists, use the inferred background color.

    Key insight:
    - The legend still tells you which components are removable, but the visual result now depends on two extra inferences:
      first infer the current sample's background color, then propagate colors inward from the deleted region's local non-bg neighborhood.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_sz, _legend_border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"
    delete_mask = _delete_mask_from_legend(g, bg, legend_sz, legend_colors)
    return _nearest_survivor_fill(g, delete_mask, bg)


def _available_colors(bg: int) -> list[int]:
    return [c for c in TARGET_POOL if c != bg]


def _place_legend(inp: np.ndarray, bg: int, targets: list[int]) -> None:
    """Draw the fixed-format legend in the top-left 4x4 block."""
    s = LEGEND_SIZE
    inp[:s, :s] = bg
    inp[s - 1, :s] = LEGEND_BORDER
    inp[:s, s - 1] = LEGEND_BORDER

    positions = [(0, 0), (0, 2), (2, 1)]
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions, strict=False):
        inp[r, c] = color


def _component_patch(
    h: int,
    w: int,
    targets: list[int],
    deleted: bool,
    palette: list[int],
) -> np.ndarray:
    """Solid BODY_COLOR rectangle with sprinkled accent colors."""
    patch = np.full((h, w), BODY_COLOR, dtype=int)
    area = h * w
    lo = max(len(targets) if deleted else 1, math.ceil(area * COLOR_DENSITY_LO))
    hi = max(lo, math.floor(area * COLOR_DENSITY_HI))
    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    non_targets = [c for c in palette if c not in targets]
    if deleted:
        colors = list(targets) + [random.choice(palette) for _ in range(n_colored - len(targets))]
    else:
        used_target_count = random.randint(0, len(targets) - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    for (r, c), color in zip(coords, colors, strict=False):
        patch[r, c] = color
    return patch


def _deleted_candidates(inp: np.ndarray, bg: int, h: int, w: int) -> list[tuple[int, int]]:
    """Positions where a deleted rectangle and its four side beacons fit cleanly."""
    occ = (inp != bg)
    candidates: list[tuple[int, int]] = []
    for r0 in range(2, GRID_H - h - 1):
        for c0 in range(2, GRID_W - w - 1):
            rr0 = r0 - 2
            rr1 = r0 + h + 2
            cc0 = c0 - 2
            cc1 = c0 + w + 2
            if occ[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _paste_bar(inp: np.ndarray, bg: int, cells: list[tuple[int, int]], color: int) -> None:
    patch = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c in cells:
        patch[r, c] = color
    paste(inp, patch, 0, 0, transparent=bg)


def _beacon_cells(r0: int, c0: int, h: int, w: int) -> list[list[tuple[int, int]]]:
    """
    Return 4 side bars (top, bottom, left, right), each separated from the rectangle by 1 bg cell.
    """
    top_len = min(3, w - 1)
    bot_len = min(3, w - 1)
    left_len = min(3, h - 1)
    right_len = min(3, h - 1)

    top_start = c0 + (w - top_len) // 2
    bot_start = c0 + (w - bot_len) // 2
    left_start = r0 + (h - left_len) // 2
    right_start = r0 + (h - right_len) // 2

    return [
        [(r0 - 2, c) for c in range(top_start, top_start + top_len)],
        [(r0 + h + 1, c) for c in range(bot_start, bot_start + bot_len)],
        [(r, c0 - 2) for r in range(left_start, left_start + left_len)],
        [(r, c0 + w + 1) for r in range(right_start, right_start + right_len)],
    ]


def _place_deleted_core(inp: np.ndarray, bg: int, targets: list[int], palette: list[int]) -> bool:
    """Place one large deletable component plus four nearby donor beacons."""
    h = random.randint(*DELETED_H)
    w = random.randint(*DELETED_W)
    candidates = _deleted_candidates(inp, bg, h, w)
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    patch = _component_patch(h, w, targets, True, palette)
    paste(inp, patch, r0, c0)

    beacon_colors = random.sample(palette, 4)
    for cells, color in zip(_beacon_cells(r0, c0, h, w), beacon_colors, strict=False):
        _paste_bar(inp, bg, cells, color)
    return True


def _place_kept_components(inp: np.ndarray, bg: int, targets: list[int], palette: list[int]) -> bool:
    """Place two extra non-deletable objects with a 1-cell bg moat."""
    for _ in range(2):
        h = random.randint(*KEPT_H)
        w = random.randint(*KEPT_W)
        patch = _component_patch(h, w, targets, False, palette)
        try:
            r0, c0 = random_free_location(inp, patch, transparent=bg, padding=1, padding_connectivity=8)
        except ValueError:
            return False
        paste(inp, patch, r0, c0)
    return True


def _old_style_output(inp: np.ndarray) -> np.ndarray:
    """Baseline for degeneration checks: same deletion criterion, flat bg refill."""
    g = np.asarray(inp, dtype=int)
    bg = _bg_boundary(g)
    legend_sz, _legend_border, legend_colors = _find_legend(g, bg)
    delete_mask = _delete_mask_from_legend(g, bg, legend_sz, legend_colors)

    out = g.copy()
    out[delete_mask] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    palette = _available_colors(bg)
    targets = random.sample(palette, 3)

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    _place_legend(inp, bg, targets)

    if not _place_deleted_core(inp, bg, targets, palette):
        return None
    if not _place_kept_components(inp, bg, targets, palette):
        return None

    new_out = main(inp)
    old_out = _old_style_output(inp)
    if np.array_equal(new_out, old_out):
        return None

    legend_sz, _legend_border, legend_colors = _find_legend(inp, bg)
    delete_mask = _delete_mask_from_legend(inp, bg, legend_sz, legend_colors)
    if not bool(delete_mask.any()):
        return None
    if not bool(np.any(new_out[delete_mask] != bg)):
        return None
    if len(set(map(int, np.unique(new_out[delete_mask])))) < 3:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused input")
