# Combo variant of e3721c99 (index 2)
# combo_spec: D1/0+D4/1
# Name: 形状轮廓匹配+洞内填色

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, is_contiguous, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior trace:
#    - Original task: a grey sprite was reduced to its hole count, matched against a
#      legend shape with the same hole count, then recolored completely.
#    - Combo behavior: first ignore enclosed cavities and compare only the outer
#      contour. If a grey sprite's filled silhouette matches a legend's filled
#      silhouette after translation, the sprite matches even when their hole counts
#      differ. The output then keeps the grey body and paints only that sprite's own
#      enclosed holes with the legend color.
#    - Small example: suppose a pink legend is an L-shaped ring with 2 holes. A grey
#      L-shaped sprite with 1 hole still matches because both fill to the same solid
#      L outline. Old logic would reject it by hole count; combo logic accepts it and
#      colors just the single interior hole pink while leaving the shell grey.
# b) Complexity calibration:
#    - A fixed silhouette library keeps contour matching legible and avoids accidental
#      near-matches from organic noise.
#    - Every legend template used for matching supports at least two positive hole
#      counts, so each legend can be backed by at least one sprite whose hole count
#      differs from the legend's own hole count while still showing a visible fill.
#    - Legend hole counts are often duplicated across different silhouettes so the old
#      hole-count rule cannot explain the examples.
# c) Hidden assumptions from the original main() that no longer hold:
#    - `hole_count -> color` is invalid once different legend silhouettes may share a
#      hole count; the lookup key must become the hole-filled silhouette.
#    - Overwriting every grey pixel is wrong because the transformed attribute is now
#      the interior cavities, not the shell.
#    - A distractor can no longer be defined only by an unlisted hole count; it must
#      use a silhouette absent from the legend set.
# d) Placement legality:
#    - Legends still live in a blocked panel behind a divider.
#    - Workspace sprites keep 1-cell padding and also use non-overlapping bounding
#      boxes, preventing one sprite from sitting inside another sprite's hole and
#      being overwritten by the fill operation.
# e) Visual observability:
#    - Each legend generates at least one repeated match whose hole count differs from
#      the legend patch, making the contour rule visible.
#    - Matches retain grey outlines but gain colored interiors, while distractors
#      vanish entirely, making the fill-vs-recolor distinction visible.
# f) Visual cleanliness:
#    - The grid contains only three meaningful workspace layers: background, grey
#      shell, and one legend color inside the shell. Template spacing and bbox
#      separation keep components visually separate.
# g) Color-change inferability:
#    - Colors are copied directly from the legend that shares the contour. They are
#      not ordered numerically or chosen by position, so the visible structural cue is
#      precisely "same outline -> use this legend color inside the holes".
# h) Quantitative anti-degeneration target:
#    - Each sample should contain matched sprites with positive holes and with hole
#      counts different from their legend's hole count, and at least one duplicated
#      legend hole count across different silhouettes when possible.
# i) New solver assumptions and generator guarantees:
#    - Solver assumes one interior legend component per legend color; generator uses
#      unique colors and border-separated panel placement.
#    - Solver assumes fixed orientation; generator never rotates or flips workspace
#      sprites relative to their legend template.
#    - Solver assumes enclosed holes are empty in the input and safe to paint;
#      generator enforces bbox disjointness and validates every sample with main().


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return enclosed background components inside a cropped boolean mask."""
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
            touches = False
            pts = []
            while st:
                r, c = st.pop()
                pts.append((int(r), int(c)))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes.append(pts)
    return holes


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed background components in a cropped boolean mask."""
    return len(_hole_regions(mask))


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only background regions that are fully enclosed inside the mask bbox."""
    out = mask.copy()
    for hole in _hole_regions(mask):
        for r, c in hole:
            out[r, c] = True
    return out


def _submask_from_points(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Canonicalize a component by its hole-filled silhouette."""
    filled = _fill_internal_holes(mask)
    return tuple(tuple(int(v) for v in row) for row in filled.astype(int))


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[tuple[tuple[int, ...], ...], int]:
    """
    Pick one legend component per non-(bg,grey) color:
      - component is at least 2x2,
      - component does not touch the outer grid border.
    Map outer-silhouette key -> legend color.
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
        assert key not in mapping, "legend silhouettes must map to exactly one color"
        mapping[key] = int(c)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside grey shapes by matching their outer silhouette to a color legend.

    Input structure:
    - bg is the most common color; grey (5) sprites occupy the workspace.
    - Colored legend components sit in a border-adjacent panel behind a divider.
    - Legend shapes and grey sprites may contain different numbers of enclosed holes
      even when they share the same external outline.

    Rule:
    - Each colored legend component defines one outer contour, obtained by ignoring
      its enclosed cavities.
    - For every grey sprite, fill its enclosed holes only for the purpose of contour
      comparison, then match that contour against the legend panel.
    - If a match exists, keep the grey shell unchanged and paint each enclosed hole
      of that sprite with the legend color; otherwise erase the whole grey sprite.

    Key insight:
    - The legend is selected by exact external geometry, but the transferred color is
      applied to the sprite's internal holes rather than to its body.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        key = _shape_key(sub)
        if key not in legend:
            for r, c in pts:
                out[r, c] = bg
            continue
        fillc = int(legend[key])
        for hole in _hole_regions(sub):
            for dr, dc in hole:
                out[r0 + dr, c0 + dc] = fillc

    return out


# --------------- generator (contour-first templates + hole fill output) ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
GEN_RETRIES = 300

FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3
PANEL_PADDING = 2
PANEL_EXTRA_THICKNESS = 2

MAX_VARIANT_HOLES = 2
MIN_HOLE_SEP = 2

_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_STRUCT8 = generate_binary_structure(2, 2)


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


OUTER_TEMPLATES: dict[str, np.ndarray] = {
    "l_left": _mask_from_rows([
        "11110",
        "11110",
        "11110",
        "11110",
        "11111",
        "11111",
    ]),
    "l_right": _mask_from_rows([
        "01111",
        "01111",
        "01111",
        "01111",
        "11111",
        "11111",
    ]),
    "t_cap": _mask_from_rows([
        "111111",
        "111111",
        "001100",
        "001100",
        "001100",
        "001100",
    ]),
    "plus": _mask_from_rows([
        "011110",
        "011110",
        "111111",
        "111111",
        "011110",
        "011110",
    ]),
    "step_left": _mask_from_rows([
        "111100",
        "111100",
        "111111",
        "001111",
        "001111",
        "001111",
    ]),
    "step_right": _mask_from_rows([
        "001111",
        "001111",
        "111111",
        "111100",
        "111100",
        "111100",
    ]),
    "c_shape": _mask_from_rows([
        "111111",
        "111111",
        "110000",
        "110000",
        "111111",
        "111111",
    ]),
    "p_shape": _mask_from_rows([
        "11110",
        "11111",
        "11111",
        "11110",
        "11100",
        "11100",
    ]),
}


def _template_safe_sites(mask: np.ndarray) -> list[tuple[int, int]]:
    core = binary_erosion(mask, structure=_STRUCT4, border_value=0)
    return [tuple(map(int, rc)) for rc in np.argwhere(core)]


def _seed_groups(sites: list[tuple[int, int]], count: int) -> list[tuple[tuple[int, int], ...]]:
    valid = []
    for comb in itertools.combinations(sites, count):
        ok = True
        for (r1, c1), (r2, c2) in itertools.combinations(comb, 2):
            if max(abs(r1 - r2), abs(c1 - c2)) < MIN_HOLE_SEP:
                ok = False
                break
        if ok:
            valid.append(tuple(sorted(comb)))
    return valid


def _enumerate_variants(outer: np.ndarray) -> dict[int, list[np.ndarray]]:
    variants: dict[int, list[np.ndarray]] = {0: [outer.copy()]}
    sites = _template_safe_sites(outer)
    for hole_count in range(1, MAX_VARIANT_HOLES + 1):
        cur = []
        for seeds in _seed_groups(sites, hole_count):
            mask = outer.copy()
            for r, c in seeds:
                mask[r, c] = False
            if not is_contiguous(mask, background=0, connectivity=4):
                continue
            if _hole_count(mask) != hole_count:
                continue
            if not np.array_equal(_fill_internal_holes(mask), outer):
                continue
            cur.append(mask)
        if cur:
            variants[hole_count] = cur
    return variants


TEMPLATE_VARIANTS = {name: _enumerate_variants(mask) for name, mask in OUTER_TEMPLATES.items()}

for _name, _outer in OUTER_TEMPLATES.items():
    assert is_contiguous(_outer, background=0, connectivity=4)
    assert np.array_equal(_fill_internal_holes(_outer), _outer)
    assert 1 in TEMPLATE_VARIANTS[_name], f"template {_name} must support a visible hole variant"


def _template_options(name: str) -> list[int]:
    return sorted(TEMPLATE_VARIANTS[name].keys())


def _positive_template_options(name: str) -> list[int]:
    return [h for h in _template_options(name) if h > 0]


def _legend_viable_templates() -> list[str]:
    return [name for name in OUTER_TEMPLATES if len(_positive_template_options(name)) >= 2]


def _rand_between(lo: int, hi: int) -> int:
    lo = int(lo)
    hi = int(hi)
    assert lo <= hi
    return lo if lo == hi else random.randint(lo, hi)


def _render_template(name: str, color: int, hole_count: int) -> np.ndarray:
    mask = random.choice(TEMPLATE_VARIANTS[name][hole_count])
    return np.where(mask, int(color), BG).astype(int)


def _legend_patch_size(legend: dict) -> tuple[int, int]:
    return legend["patch"].shape


def _line_span(legends: list[dict], arrangement: str) -> int:
    idx = 1 if arrangement == "horizontal" else 0
    return sum(_legend_patch_size(legend)[idx] for legend in legends) + len(legends) - 1


def _cross_span(legends: list[dict], arrangement: str) -> int:
    idx = 0 if arrangement == "horizontal" else 1
    return max(_legend_patch_size(legend)[idx] for legend in legends)


def _choose_legends(arrangement: str, max_span: int) -> list[dict]:
    del arrangement
    viable = _legend_viable_templates()
    for _ in range(200):
        count = random.randint(*LEGEND_COUNT_RANGE)
        names = random.sample(viable, count)
        random.shuffle(names)
        colors = random.sample(LEGEND_COLORS, count)
        legends = []
        for name, color in zip(names, colors):
            opts = _positive_template_options(name)
            legends.append({
                "template": name,
                "color": int(color),
                "legend_holes": random.choice(opts),
            })

        if count >= 2:
            i, j = random.sample(range(count), 2)
            shared = sorted(set(_positive_template_options(names[i])) & set(_positive_template_options(names[j])))
            if shared:
                val = random.choice(shared)
                legends[i]["legend_holes"] = val
                legends[j]["legend_holes"] = val

        for legend in legends:
            legend["patch"] = _render_template(
                legend["template"], legend["color"], legend["legend_holes"]
            )

        if _line_span(legends, "horizontal") <= max_span and _line_span(legends, "vertical") <= max_span:
            return legends
    raise RuntimeError("failed to choose legend set")


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
    along = total + PANEL_PADDING + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + PANEL_PADDING)))
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

    plan = []
    for i, legend in enumerate(legends):
        positive = _positive_template_options(legend["template"])
        different = [h for h in positive if h != legend["legend_holes"]]
        first_holes = random.choice(different)
        plan.append({
            "template": legend["template"],
            "sprite_holes": first_holes,
            "color": legend["color"],
            "legend_holes": legend["legend_holes"],
        })
        for _ in range(extras[i]):
            plan.append({
                "template": legend["template"],
                "sprite_holes": random.choice(positive),
                "color": legend["color"],
                "legend_holes": legend["legend_holes"],
            })

    legend_templates = {legend["template"] for legend in legends}
    distractor_templates = [name for name in OUTER_TEMPLATES if name not in legend_templates]
    for _ in range(n_distractors):
        name = random.choice(distractor_templates)
        plan.append({
            "template": name,
            "sprite_holes": random.choice(_positive_template_options(name)),
            "color": None,
            "legend_holes": None,
        })
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, occupied_bbox: np.ndarray, sprite: np.ndarray
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
        if np.any(occupied_bbox[r : r + h, c : c + w]):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]
) -> bool:
    items = []
    for item in _sprite_plan(legends):
        sprite = _render_template(item["template"], GREY, item["sprite_holes"])
        items.append((sprite.size, sprite, item["color"]))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    occupied_bbox = np.zeros_like(grid, dtype=bool)
    for _, sprite, color in items:
        loc = _random_sprite_location(blocked, occupied, occupied_bbox, sprite)
        if loc is None:
            return False
        r, c = loc
        h, w = sprite.shape
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        occupied_bbox[r : r + h, c : c + w] = True
        if color is not None:
            paste(expected, sprite, r, c, transparent=BG)
            for hole in _hole_regions(sprite == GREY):
                for dr, dc in hole:
                    expected[r + dr, c + dc] = int(color)
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
