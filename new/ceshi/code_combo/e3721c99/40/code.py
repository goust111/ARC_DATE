# Combo variant of e3721c99 (index 40)
# combo_spec: D1/0+D2/1+D4/0
# Name: 形状匹配+中空框架+洞边框着色

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, is_contiguous, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The original task keyed each legend by hole count and then repainted every
#      matching grey sprite monochromatically.
#    - In this combo, the workspace sprites are thick hollow frames. The legend
#      also shows thick hollow frames, and matching is performed on the exact
#      cropped frame mask, not on a scalar hole count.
#    - Example: two legend entries can share the same 6x10 outer rectangle while
#      one has a single long cavity and the other has two small cavities. Under
#      the original rule, those could never coexist because hole count would be
#      the only key. Under the fused rule, they map to different colors because
#      their full frame contours differ.
#    - After a grey frame matches a legend, only the frame pixels that touch an
#      enclosed hole become the legend color; the outer band of the frame stays
#      grey. Thus the output is explicitly two-tone and visibly different from
#      both the original full recolor and from leaving the sprite unchanged.
# b) Complexity calibration:
#    - A small fixed template library prevents accidental near-duplicates while
#      still giving several repeated supports per sample.
#    - Each sample deliberately includes three legends:
#      1) a same-outer-bbox family pair with different internal cavities, so
#         exact frame shape matters;
#      2) an extra one-hole legend from another family, so hole count alone is
#         ambiguous.
#    - Matching sprites repeat these templates in the workspace, and at least one
#      distractor shares a legend hole count but not the legend frame mask.
# c) Hidden-assumption review against the original main():
#    - Original `_legend_mapping()` assumed `hole_count -> color`; that breaks as
#      soon as two legend frames share a hole count. The new key is the exact
#      cropped frame mask.
#    - Original grey processing collapsed a sprite to one scalar hole count. That
#      discards the internal frame partitioning, which is now semantically
#      important.
#    - Original `for r, c in pts: out[r, c] = newc` assumed uniform recoloring.
#      The fused transform instead requires per-pixel classification into
#      hole-border versus outer-frame body.
#    - If matching used only the filled silhouette, the D2 hollow-frame change
#      would mostly collapse back into plain rectangle matching. To keep the
#      fusion real, the solver compares the exact hollow frame mask.
# d) Placement legality:
#    - Legends remain in a border-adjacent panel behind a divider.
#    - Workspace sprites are pasted only in unblocked cells, with an 8-neighbor
#      dilation gap so frames neither overlap nor touch.
#    - Templates are pre-validated as 4-connected, hole-bearing frame components.
# e) Visual observability:
#    - A solver can see two same-sized legend rectangles with different internal
#      cavity layouts and different colors, which makes exact frame matching
#      inferable from a few examples.
#    - The output always preserves a grey outer band, so the inner-border-only
#      recoloring is directly visible.
# f) Visual cleanliness:
#    - The template library uses bold orthogonal frames with 2-cell-thick outer
#      bands; legends and sprites are separated by background padding.
#    - The panel remains visually simple while workspace objects show the new
#      double-border effect.
# g) Color-change inferability and anti-degeneration target:
#    - Legend colors are unique and tied only to visible frame shapes.
#    - Every sample contains duplicated legend hole count 1 and at least one
#      distractor whose hole count collides with the legend set, so neither the
#      original hole-count rule nor a filled-silhouette shortcut is sufficient.
# h) New solver assumptions that the generator guarantees:
#    - One interior legend component per legend color.
#    - No rotation or reflection between legend and workspace copies.
#    - Every matching sprite has at least one enclosed hole and at least one grey
#      outer-band pixel not adjacent to any hole.


def _submask_from_points(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _enclosed_hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return enclosed 4-connected background regions inside a cropped boolean mask."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            cells = []
            touches = False
            while st:
                r, c = st.pop()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes.append(cells)
    return holes


def _hole_count(mask: np.ndarray) -> int:
    return len(_enclosed_hole_regions(mask))


def _hole_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return mask pixels that 4-touch at least one enclosed hole cell."""
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for hole in _enclosed_hole_regions(mask):
        for r, c in hole:
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    boundary[rr, cc] = True
    return boundary


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Canonicalize an exact frame mask by cropping away translation."""
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _legend_mapping(
    g: np.ndarray, bg: int, grey: int
) -> dict[tuple[tuple[int, ...], ...], int]:
    """
    Pick one legend component per non-(bg,grey) color:
      - component is at least 2x2,
      - component does not touch the outer grid border.
    Map exact hollow-frame mask -> legend color.
    """
    h, w = g.shape
    mapping: dict[tuple[tuple[int, ...], ...], int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        key = _shape_key(_submask_from_points(candidates[0]))
        assert key not in mapping, "legend frame masks must map to exactly one color"
        mapping[key] = int(c)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey hollow frames to legend frames and color only each matched frame's inner rim.

    Input structure:
    - The background is the most common color, with grey (5) hollow-frame sprites scattered in the workspace.
    - A border-adjacent legend panel contains colored hollow frames behind a divider.
    - Different legend entries may share an outer bounding box or even a hole count, but their exact frame masks differ.

    Rule:
    - Treat each colored legend component as one exact target frame shape, normalized only by translation.
    - For every grey workspace sprite, crop its boolean mask and compare that exact hollow-frame mask to the legends.
    - If a match exists, recolor only the sprite pixels that touch an enclosed hole to the legend color, leaving the rest grey.
      If no exact frame match exists, erase the grey sprite to background.

    Key insight:
    - The decisive feature is the full hollow-frame contour, including how internal cavities partition the frame;
      the output then highlights only the hole-adjacent inner border rather than repainting the entire object.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend frames"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        sub = _submask_from_points(pts)
        key = _shape_key(sub)
        if key not in legend:
            for r, c in pts:
                out[r, c] = bg
            continue
        boundary = _hole_boundary_mask(sub)
        newc = int(legend[key])
        r0, c0, _, _ = bbox_coords(pts)
        for r, c in pts:
            out[r, c] = newc if boundary[r - r0, c - c0] else grey

    return out


# --------------- generator (template hollow-frames) ---------------

BG = 0
DIV = 1
GREY = 5
# Slightly larger workspace preserves the same legend/sprite logic while
# reducing rare placement dead-ends from crowded layouts.
GRID_SIZE = 32
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT = 3
TOTAL_SPRITES_RANGE = (4, 5)
MAX_DISTRACTORS = 1
MAX_EXTRA_PER_LEGEND = 1
GEN_RETRIES = 1200

FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3
PANEL_PADDING = 2
PANEL_EXTRA_THICKNESS = 2
MIN_VISIBLE_CORE = 8

_STRUCT8 = generate_binary_structure(2, 2)


def _frame_from_rects(
    h: int, w: int, holes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    for r0, c0, r1, c1 in holes:
        mask[r0:r1, c0:c1] = False
    return mask


TEMPLATE_MASKS: dict[str, np.ndarray] = {
    "long1_h": _frame_from_rects(6, 10, [(2, 2, 4, 8)]),
    "double_h": _frame_from_rects(6, 10, [(2, 2, 4, 4), (2, 6, 4, 8)]),
    "long1_v": _frame_from_rects(10, 6, [(2, 2, 8, 4)]),
    "double_v": _frame_from_rects(10, 6, [(2, 2, 4, 4), (6, 2, 8, 4)]),
    "single_square_big": _frame_from_rects(8, 8, [(2, 2, 6, 6)]),
    "double_diag": _frame_from_rects(8, 8, [(2, 2, 4, 4), (4, 4, 6, 6)]),
    "single_wide_big": _frame_from_rects(8, 10, [(2, 2, 6, 8)]),
    "triple_compact": _frame_from_rects(8, 10, [(2, 2, 4, 4), (2, 6, 4, 8), (4, 4, 6, 6)]),
}

FAMILIES: dict[str, tuple[str, str]] = {
    "hrect": ("long1_h", "double_h"),
    "vrect": ("long1_v", "double_v"),
    "square": ("single_square_big", "double_diag"),
    "wide": ("single_wide_big", "triple_compact"),
}

TEMPLATE_INFO: dict[str, dict[str, int | str]] = {}
for family, members in FAMILIES.items():
    for name in members:
        TEMPLATE_INFO[name] = {
            "holes": _hole_count(TEMPLATE_MASKS[name]),
            "family": family,
        }

for family, (name_a, name_b) in FAMILIES.items():
    mask_a = TEMPLATE_MASKS[name_a]
    mask_b = TEMPLATE_MASKS[name_b]
    assert mask_a.shape == mask_b.shape, f"family {family} must share an outer bbox"

for name, mask in TEMPLATE_MASKS.items():
    assert is_contiguous(mask, background=0, connectivity=4), f"{name} must be 4-connected"
    hole_count = int(TEMPLATE_INFO[name]["holes"])
    assert hole_count >= 1, f"{name} must have at least one hole"
    boundary = _hole_boundary_mask(mask)
    boundary_area = int(np.sum(boundary))
    area = int(np.sum(mask))
    core_area = area - boundary_area
    assert 0 < boundary_area < area, f"{name} must have a visible inner rim"
    assert core_area >= MIN_VISIBLE_CORE, f"{name} must keep a grey outer band"


def _rand_between(lo: int, hi: int) -> int:
    lo = int(lo)
    hi = int(hi)
    assert lo <= hi
    return lo if lo == hi else random.randint(lo, hi)


def _render_template(name: str, color: int) -> np.ndarray:
    return np.where(TEMPLATE_MASKS[name], int(color), BG).astype(int)


def _template_size(name: str) -> tuple[int, int]:
    return TEMPLATE_MASKS[name].shape


def _legend_patch_size(legend: dict) -> tuple[int, int]:
    return legend["patch"].shape


def _line_span_from_names(names: list[str], arrangement: str) -> int:
    idx = 1 if arrangement == "horizontal" else 0
    return sum(_template_size(name)[idx] for name in names) + len(names) - 1


def _line_span(legends: list[dict], arrangement: str) -> int:
    idx = 1 if arrangement == "horizontal" else 0
    return sum(_legend_patch_size(legend)[idx] for legend in legends) + len(legends) - 1


def _cross_span(legends: list[dict], arrangement: str) -> int:
    idx = 0 if arrangement == "horizontal" else 1
    return max(_legend_patch_size(legend)[idx] for legend in legends)


def _legend_name_triples(arrangement: str, max_span: int) -> list[tuple[str, str, str]]:
    triples = []
    single_templates = {family: members[0] for family, members in FAMILIES.items()}
    for family, (single_name, sibling_name) in FAMILIES.items():
        for other_family, other_single in single_templates.items():
            if other_family == family:
                continue
            names = [single_name, sibling_name, other_single]
            if _line_span_from_names(names, arrangement) <= max_span:
                triples.append(tuple(names))
    return triples


def _choose_legends(arrangement: str, max_span: int) -> list[dict]:
    triples = _legend_name_triples(arrangement, max_span)
    assert triples, f"no legend triples fit {arrangement} span {max_span}"
    names = list(random.choice(triples))
    random.shuffle(names)
    colors = random.sample(LEGEND_COLORS, LEGEND_COUNT)
    legends = []
    for name, color in zip(names, colors):
        legends.append(
            {
                "template": name,
                "family": str(TEMPLATE_INFO[name]["family"]),
                "holes": int(TEMPLATE_INFO[name]["holes"]),
                "color": int(color),
                "patch": _render_template(name, int(color)),
            }
        )
    hole_counts = [legend["holes"] for legend in legends]
    assert len(legends) == LEGEND_COUNT
    assert len(set(hole_counts)) < len(hole_counts), "legend set must repeat at least one hole count"
    return legends


def _place_legend_line(grid: np.ndarray, legends: list[dict], arrangement: str, r: int, c: int) -> None:
    rr, cc = int(r), int(c)
    for legend in legends:
        patch = legend["patch"]
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends, "horizontal")
    cross = _cross_span(legends, "horizontal")
    panel_h = cross + PANEL_PADDING + random.randint(0, PANEL_EXTRA_THICKNESS)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = _rand_between(1, panel_h - cross - 1)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = _rand_between(div_r + 2, GRID_SIZE - cross - 1)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends, "vertical")
    cross = _cross_span(legends, "vertical")
    panel_w = cross + PANEL_PADDING + random.randint(0, PANEL_EXTRA_THICKNESS)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = _rand_between(1, panel_w - cross - 1)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = _rand_between(div_c + 2, GRID_SIZE - cross - 1)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str,
    panel_h: int,
    panel_w: int,
    cross_dim: int,
    along_dim: int,
) -> tuple[int, int, int, int, int, int]:
    n = GRID_SIZE
    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim
    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends, arrangement)
    cross = _cross_span(legends, arrangement)
    along = total + PANEL_PADDING + random.randint(
        0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + PANEL_PADDING))
    )
    cross_panel = cross + PANEL_PADDING + random.randint(0, PANEL_EXTRA_THICKNESS)
    if arrangement == "horizontal":
        panel_h, panel_w = cross_panel, along
        cross_dim, along_dim = cross, total
    else:
        panel_h, panel_w = along, cross_panel
        cross_dim, along_dim = total, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, cross_dim, along_dim
    )

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = _rand_between(r_min, r_max)
    c0 = _rand_between(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_plan(legends: list[dict]) -> list[dict]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors

    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[dict] = []
    for i, legend in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append({"template": legend["template"], "color": legend["color"]})

    legend_names = {legend["template"] for legend in legends}
    legend_holes = {legend["holes"] for legend in legends}
    distractor_pool = [name for name in TEMPLATE_MASKS if name not in legend_names]
    collision_pool = [
        name for name in distractor_pool if int(TEMPLATE_INFO[name]["holes"]) in legend_holes
    ]

    if n_distractors > 0:
        assert collision_pool, "need at least one hole-count-collision distractor"
        plan.append({"template": random.choice(collision_pool), "color": None})
        n_distractors -= 1

    for _ in range(n_distractors):
        plan.append({"template": random.choice(distractor_pool), "color": None})

    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    candidates = [(r, c) for r in range(GRID_SIZE - h + 1) for c in range(GRID_SIZE - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(GRID_SIZE, r + h + 1), min(GRID_SIZE, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _matched_output_patch(sprite: np.ndarray, color: int) -> np.ndarray:
    mask = sprite == GREY
    boundary = _hole_boundary_mask(mask)
    out = sprite.copy()
    out[boundary] = int(color)
    return out


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]
) -> bool:
    items = []
    for item in _sprite_plan(legends):
        sprite = _render_template(item["template"], GREY)
        items.append((sprite.size, sprite, item["color"]))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, color in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = _matched_output_patch(sprite, int(color))
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
