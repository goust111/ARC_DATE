from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import permutations

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _relative_point(pt: Coord, box: BBox) -> tuple[Fraction, Fraction]:
    r, c = pt
    r0, c0, r1, c1 = box
    assert r1 > r0 and c1 > c0, "reference box must span both axes"
    return Fraction(r - r0, r1 - r0), Fraction(c - c0, c1 - c0)


def _relative_bbox_center(obj_box: BBox, reference_box: BBox) -> tuple[Fraction, Fraction]:
    r0, c0, r1, c1 = obj_box
    rr0, cc0, rr1, cc1 = reference_box
    assert rr1 > rr0 and cc1 > cc0, "object layout must span both axes"
    return (
        Fraction(r0 + r1 - 2 * rr0, 2 * (rr1 - rr0)),
        Fraction(c0 + c1 - 2 * cc0, 2 * (cc1 - cc0)),
    )


def _dist2(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> Fraction:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _unique_assignment(
    src_pos: list[tuple[Fraction, Fraction]],
    dst_pos: list[tuple[Fraction, Fraction]],
) -> tuple[int, ...]:
    assert len(src_pos) == len(dst_pos), "objects and markers must have the same count"

    scored: list[tuple[Fraction, tuple[int, ...]]] = []
    for perm in permutations(range(len(dst_pos))):
        total = sum(_dist2(src_pos[i], dst_pos[perm[i]]) for i in range(len(src_pos)))
        scored.append((total, perm))

    best_score = min(score for score, _perm in scored)
    best = [perm for score, perm in scored if score == best_score]
    assert len(best) == 1, "relative-position matching must be unique"
    return best[0]


def _legend_body(g: np.ndarray, bg: int) -> tuple[int, BBox]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        by_color[int(comp["color"])].append(comp)

    one_component_colors = [color for color, comps in by_color.items() if len(comps) == 1]
    assert one_component_colors, "input must contain a single-component legend body"

    max_area = max(int(by_color[color][0]["area"]) for color in one_component_colors)
    legend_colors = [
        color for color in one_component_colors if int(by_color[color][0]["area"]) == max_area
    ]
    assert len(legend_colors) == 1, "legend body must be the unique largest single component"
    color = legend_colors[0]
    return color, by_color[color][0]["bbox"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each placeholder object by the control pixel at the matching relative position inside the legend.

    Input structure:
    - The background is the dominant color.
    - A large single-color legend body forms a rectangle containing one-pixel
      colored markers.
    - Outside the legend rectangle, all foreground pixels have one placeholder
      color and form the same number of 8-connected objects as there are markers.

    Rule:
    - Treat the bbox of all placeholder objects as a map of their layout.
    - Treat the legend bbox as the corresponding map of control-pixel positions.
    - Match placeholder objects to markers by the unique minimum total distance
      between bbox-center relative positions.
    - Recolor every pixel of each placeholder object with its matched marker
      color, leaving the legend and background unchanged.

    Key insight:
    - The gray block is a positional legend: marker locations, not marker color
      order or object shape, determine which color each outside object receives.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend_color, legend_box = _legend_body(g, bg)
    lr0, lc0, lr1, lc1 = legend_box
    legend_rect = np.zeros(g.shape, dtype=bool)
    legend_rect[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    marker_cells = np.argwhere(legend_rect & (g != bg) & (g != legend_color))
    assert marker_cells.size > 0, "legend must contain control pixels"
    marker_colors = [int(g[r, c]) for r, c in marker_cells]
    assert len(set(marker_colors)) == len(marker_colors), "control colors must be unique"

    outside_fg = (~legend_rect) & (g != bg)
    outside_colors = set(map(int, g[outside_fg]))
    assert len(outside_colors) == 1, "outside foreground must be one placeholder color"
    placeholder_color = outside_colors.pop()
    assert placeholder_color not in set(marker_colors), "placeholder color must be distinct"

    object_coords = components_mask((g == placeholder_color) & (~legend_rect), connectivity=8)
    assert len(object_coords) == len(marker_cells), "each marker must correspond to one object"

    object_box = bbox_coords(np.vstack(object_coords))
    object_boxes = [bbox_coords(coords) for coords in object_coords]
    object_pos = [_relative_bbox_center(obj_box, object_box) for obj_box in object_boxes]
    marker_pos = [_relative_point((int(r), int(c)), legend_box) for r, c in marker_cells]
    assignment = _unique_assignment(object_pos, marker_pos)

    out = g.copy()
    for coords, marker_idx in zip(object_coords, assignment):
        out[coords[:, 0], coords[:, 1]] = marker_colors[marker_idx]
    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste

# Colors.
BG_COLOR, LEGEND_COLOR = 0, 5
COLOR_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

# Size bounds; RNG upper bounds are exclusive unless noted.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (14, 20), (20, 26)
MARKER_COUNT_RANGE = (3, 6)
LEGEND_HEIGHT_MIN, LEGEND_HEIGHT_MAX = 6, 10
LEGEND_WIDTH_RANGE = (6, 9)

# Sprite constraints.
SPRITE_HEIGHT_RANGE, SPRITE_WIDTH_RANGE = (2, 5), (3, 7)
SPRITE_PIXEL_MIN, SPRITE_PIXEL_MAX = 5, 15
SPRITE_FILL_MIN, SPRITE_FILL_MAX = 0.47, 0.78

# Layout constraints.
MARKER_CHEB_GAP, OBJECT_CHEB_GAP = 2, 3
OBJECT_LEGEND_MANHATTAN_GAP = 2
OBJECT_ZONE_MIN_SPAN, OBJECT_ZONE_BORDER = 8, 1
LEGEND_ZONE_GAP, LEGEND_SIDE_OFFSET = 2, 10

# Placement and retry behavior.
PLACEMENT_SCORE_JITTER, GEN_MAX_ATTEMPTS = 0.01, 300


def _gen_rect_distance(r: int, c: int, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    dr = max(r0 - r, 0, r - r1)
    dc = max(c0 - c, 0, c - c1)
    return dr + dc

def _gen_cheb_gap(a: np.ndarray, b: np.ndarray) -> int:
    dr = np.abs(a[:, None, 0] - b[None, :, 0])
    dc = np.abs(a[:, None, 1] - b[None, :, 1])
    return int(np.maximum(dr, dc).min())

def _gen_sprite_mask(rng: np.random.Generator) -> np.ndarray:
    feasible: list[tuple[int, int, int, int]] = []
    for h in range(*SPRITE_HEIGHT_RANGE):
        for w in range(*SPRITE_WIDTH_RANGE):
            lo = max(SPRITE_PIXEL_MIN, int(np.ceil(SPRITE_FILL_MIN * h * w)))
            hi = min(SPRITE_PIXEL_MAX, int(np.floor(SPRITE_FILL_MAX * h * w)), h * w - 1)
            if lo <= hi:
                feasible.append((h, w, lo, hi))

    h, w, lo, hi = feasible[int(rng.integers(len(feasible)))]
    target = int(rng.integers(lo, hi + 1))
    mask = np.zeros((h, w), dtype=bool)

    corners = [((0, 0), (h - 1, w - 1)), ((0, w - 1), (h - 1, 0))]
    if bool(rng.integers(2)):
        start, end = corners[int(rng.integers(2))]
    else:
        end, start = corners[int(rng.integers(2))]

    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        step_r = 0 if r == er else (1 if er > r else -1)
        step_c = 0 if c == ec else (1 if ec > c else -1)
        choices: list[tuple[int, int]] = []
        if step_r and step_c:
            choices.extend([(step_r, step_c), (step_r, 0), (0, step_c)])
        elif step_r:
            choices.append((step_r, 0))
        else:
            choices.append((0, step_c))
        dr, dc = choices[int(rng.integers(len(choices)))]
        r += dr
        c += dc
        mask[r, c] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(rr) + dr, int(cc) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
        nr, nc = list(frontier)[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True

    return mask


def _gen_marker_cells(
    rng: np.random.Generator, legend_box: tuple[int, int, int, int], n: int
) -> list[tuple[int, int]]:
    lr0, lc0, lr1, lc1 = legend_box
    cells = [(r, c) for r in range(lr0, lr1 + 1) for c in range(lc0, lc1 + 1)]
    rng.shuffle(cells)

    selected: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= MARKER_CHEB_GAP for r, c in selected):
            selected.append(cell)
            if len(selected) == n:
                break

    if len(selected) != n:
        return []
    rows = [r for r, _c in selected]
    cols = [c for _r, c in selected]
    if max(rows) == min(rows) or max(cols) == min(cols):
        return []
    return selected

def _gen_regions(
    rng: np.random.Generator, h: int, w: int, lh: int, lw: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    choices: list[str] = []
    if w - lw - LEGEND_ZONE_GAP >= OBJECT_ZONE_MIN_SPAN:
        choices.extend(["left", "right"])
    if h - lh - LEGEND_ZONE_GAP >= OBJECT_ZONE_MIN_SPAN:
        choices.extend(["top", "bottom"])
    if not choices:
        return None

    zone_r1 = h - OBJECT_ZONE_BORDER - 1
    zone_c1 = w - OBJECT_ZONE_BORDER - 1
    side = choices[int(rng.integers(len(choices)))]
    if side == "left":
        max_lc0 = w - lw - LEGEND_SIDE_OFFSET
        if max_lc0 < 0:
            return None
        lr0 = int(rng.integers(0, h - lh + 1))
        lc0 = int(rng.integers(0, max_lc0 + 1))
        legend = (lr0, lc0, lr0 + lh - 1, lc0 + lw - 1)
        zone = (OBJECT_ZONE_BORDER, legend[3] + LEGEND_ZONE_GAP, zone_r1, zone_c1)
    elif side == "right":
        min_lc0 = LEGEND_SIDE_OFFSET
        max_lc0 = w - lw
        if min_lc0 > max_lc0:
            return None
        lr0 = int(rng.integers(0, h - lh + 1))
        lc0 = int(rng.integers(min_lc0, max_lc0 + 1))
        legend = (lr0, lc0, lr0 + lh - 1, lc0 + lw - 1)
        zone = (
            OBJECT_ZONE_BORDER,
            OBJECT_ZONE_BORDER,
            zone_r1,
            legend[1] - LEGEND_ZONE_GAP,
        )
    elif side == "top":
        max_lr0 = h - lh - LEGEND_SIDE_OFFSET
        if max_lr0 < 0:
            return None
        lr0 = int(rng.integers(0, max_lr0 + 1))
        lc0 = int(rng.integers(0, w - lw + 1))
        legend = (lr0, lc0, lr0 + lh - 1, lc0 + lw - 1)
        zone = (legend[2] + LEGEND_ZONE_GAP, OBJECT_ZONE_BORDER, zone_r1, zone_c1)
    else:
        min_lr0 = LEGEND_SIDE_OFFSET
        max_lr0 = h - lh
        if min_lr0 > max_lr0:
            return None
        lr0 = int(rng.integers(min_lr0, max_lr0 + 1))
        lc0 = int(rng.integers(0, w - lw + 1))
        legend = (lr0, lc0, lr0 + lh - 1, lc0 + lw - 1)
        zone = (
            OBJECT_ZONE_BORDER,
            OBJECT_ZONE_BORDER,
            legend[0] - LEGEND_ZONE_GAP,
            zone_c1,
        )

    if (
        zone[2] - zone[0] + 1 < OBJECT_ZONE_MIN_SPAN
        or zone[3] - zone[1] + 1 < OBJECT_ZONE_MIN_SPAN
    ):
        return None
    return legend, zone


def _gen_placement_candidates(
    rng: np.random.Generator,
    mask: np.ndarray,
    rel: tuple[float, float],
    zone: tuple[int, int, int, int],
) -> list[tuple[float, int, int]]:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    sh, sw = mask.shape
    target_r = zr0 + rel[0] * (zh - 1)
    target_c = zc0 + rel[1] * (zw - 1)

    locs: list[tuple[float, int, int]] = []
    for r in range(zr0, zr1 - sh + 2):
        center_r = r + (sh - 1) / 2
        for c in range(zc0, zc1 - sw + 2):
            center_c = c + (sw - 1) / 2
            score = (center_r - target_r) ** 2 + (center_c - target_c) ** 2
            score += float(rng.random()) * PLACEMENT_SCORE_JITTER
            locs.append((score, r, c))
    locs.sort()
    return locs

def _gen_place_sprite(
    rng: np.random.Generator,
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    rel: tuple[float, float],
    zone: tuple[int, int, int, int],
    legend_box: tuple[int, int, int, int],
    placed: list[np.ndarray],
) -> np.ndarray | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    sh, sw = mask.shape
    if sh > zh or sw > zw:
        return None

    sprite_coords = np.argwhere(mask)
    for _score, r, c in _gen_placement_candidates(rng, mask, rel, zone):
        coords = sprite_coords + np.array([r, c])
        if any(
            _gen_rect_distance(int(rr), int(cc), legend_box) < OBJECT_LEGEND_MANHATTAN_GAP
            for rr, cc in coords
        ):
            continue
        if any(_gen_cheb_gap(coords, old) < OBJECT_CHEB_GAP for old in placed):
            continue

        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(color)
        paste(out, patch, r, c, transparent=BG_COLOR, overwrite=False)
        return coords
    return None

def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    n = int(rng.integers(*MARKER_COUNT_RANGE))
    lh = int(rng.integers(LEGEND_HEIGHT_MIN, min(LEGEND_HEIGHT_MAX, h) + 1))
    lw = int(rng.integers(*LEGEND_WIDTH_RANGE))

    regions = _gen_regions(rng, h, w, lh, lw)
    if regions is None:
        return None
    legend_box, zone = regions
    marker_cells = _gen_marker_cells(rng, legend_box, n)
    if not marker_cells:
        return None

    colors = [int(c) for c in rng.choice(COLOR_PALETTE, size=n + 1, replace=False)]
    placeholder = colors[0]
    marker_colors = colors[1:]

    out = np.zeros((h, w), dtype=int)
    lr0, lc0, lr1, lc1 = legend_box
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = LEGEND_COLOR
    for (r, c), color in zip(marker_cells, marker_colors):
        out[r, c] = color

    rels = [
        ((r - lr0) / (lr1 - lr0), (c - lc0) / (lc1 - lc0))
        for r, c in marker_cells
    ]
    masks = [_gen_sprite_mask(rng) for _ in range(n)]

    order = sorted(range(n), key=lambda i: int(masks[i].sum()), reverse=True)
    placed: list[np.ndarray] = []
    for i in order:
        coords = _gen_place_sprite(
            rng, out, masks[i], marker_colors[i], rels[i], zone, legend_box, placed
        )
        if coords is None:
            return None
        placed.append(coords)

    inp = out.copy()
    outside = np.ones(inp.shape, dtype=bool)
    outside[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
    inp[outside & (inp != BG_COLOR)] = placeholder

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_MAX_ATTEMPTS} attempts")
