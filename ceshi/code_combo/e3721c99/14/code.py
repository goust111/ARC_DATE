# Combo variant of e3721c99 (index 14)
# combo_spec: D1/2+D8/0
# Name: 连通分量分支匹配+8连通洞计数
#
# Design analysis
# - Concrete behavior: the original puzzle mapped legend color by hole count. Here the
#   legend patches are small tree-like shapes whose identity is their branch count.
#   Grey sprites are thicker versions of branchy shapes. Before counting branches we
#   fill holes using 8-connected background, then thin to a 1-pixel skeleton and count
#   skeleton endpoints. A diagonal bg leak can connect an interior pocket to the
#   outside under 8-connectivity but not under 4-connectivity, so the same sprite can
#   change its branch signature depending on the hole rule.
# - Small example: a thick "T" body with a diagonal staircase of removed cells from an
#   inner pocket to the bbox border contains several 4-connected holes, but those cells
#   form one 8-connected leak to the exterior. Under the old logic the pocket would be
#   treated as enclosed; under this combo it stays open, the filled mask differs, and
#   the skeleton endpoint count can change.
# - Complexity calibration: branch counts are limited to {2,3,4,5}. That keeps the
#   mapping human-trackable and leaves room for distractors. The legend panel uses only
#   2-3 entries to avoid visual clutter while still giving repeated evidence across the
#   grey sprites.
# - Implicit assumptions audited from the original main():
#   1) legend extraction still assumes one interior component per legend color; the
#      generator now avoids divider-color legends to keep that unique.
#   2) grey sprites are still 4-connected components; the generator validates this after
#      every shape edit.
#   3) a branch-count -> color mapping must be injective; the generator samples distinct
#      branch counts.
#   4) branch counting now depends on a stable skeleton; the generator thickens only a
#      small family of tree-like templates and rejects anything whose computed branch
#      count does not match the intended target.
# - Placement legality: the divider blocks off the legend panel exactly as before, and
#   sprite placement uses 8-neighborhood padding so components neither overlap nor touch.
# - Visual observability: the legend explicitly shows different arm counts, while the
#   workspace repeats those branch motifs in larger grey blobs. Some blobs contain
#   diagonal bg leaks, so a human can notice that "almost holes" do not behave like
#   sealed holes here.
# - Cleanliness / inferability: colors remain sparse, objects stay separated, and the
#   rule does not depend on numeric color ids. The branch-count signal is the only
#   matching key, with 8-connected hole filling as the topological detail needed to read
#   that signal correctly.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, neighbors8, paste


def _neighbors(connectivity: int):
    return neighbors4 if int(connectivity) == 4 else neighbors8


def _fill_holes(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Return mask with enclosed background components filled."""
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    filled = mask.copy()
    neigh = _neighbors(connectivity)
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp = []
            touches = False
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neigh(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches:
                for r, c in comp:
                    filled[r, c] = True
    return filled


def _hole_count(mask: np.ndarray, connectivity: int) -> int:
    """Count enclosed background components under the chosen bg connectivity."""
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    neigh = _neighbors(connectivity)
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neigh(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning on a binary mask."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove = []
            rows, cols = np.where(img == 1)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if r == 0 or c == 0 or r == img.shape[0] - 1 or c == img.shape[1] - 1:
                    continue
                p2 = img[r - 1, c]
                p3 = img[r - 1, c + 1]
                p4 = img[r, c + 1]
                p5 = img[r + 1, c + 1]
                p6 = img[r + 1, c]
                p7 = img[r + 1, c - 1]
                p8 = img[r, c - 1]
                p9 = img[r - 1, c - 1]
                vals = [p2, p3, p4, p5, p6, p7, p8, p9]
                b = int(sum(vals))
                if b < 2 or b > 6:
                    continue
                seq = vals + [vals[0]]
                a = sum(int(seq[i] == 0 and seq[i + 1] == 1) for i in range(8))
                if a != 1:
                    continue
                if step == 0:
                    if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                        continue
                else:
                    if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                        continue
                to_remove.append((r, c))
            if to_remove:
                changed = True
                for r, c in to_remove:
                    img[r, c] = 0
    return img[1:-1, 1:-1].astype(bool)


def _endpoint_count(skel: np.ndarray) -> int:
    """Count 8-neighbor skeleton endpoints."""
    count = 0
    for r, c in np.argwhere(skel):
        deg = 0
        for rr, cc in neighbors8(int(r), int(c)):
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                deg += 1
        if deg == 1:
            count += 1
    return count


def _branch_count(mask: np.ndarray, hole_connectivity: int = 8) -> int:
    """Fill holes under the chosen rule, skeletonize, then count skeleton endpoints."""
    solid = _fill_holes(mask, connectivity=hole_connectivity)
    skel = _thin(solid)
    return _endpoint_count(skel)


def _component_mask_from_points(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[int(r - r0), int(c - c0)] = True
    return sub


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """Extract one legend component per color and map branch_count -> color."""
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
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
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        bc = _branch_count(_component_mask_from_points(candidates[0]), hole_connectivity=8)
        assert bc not in mapping, f"legend branch count {bc} must map to exactly one color"
        mapping[bc] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching their branch count to a color legend.

    Input structure:
    - The background is the dominant color; grey (5) components live in the workspace.
    - A divider separates a legend panel from the workspace.
    - Each colored legend patch is an interior branchy shape whose arm count is unique.

    Rule:
    - For every legend patch, fill enclosed gaps using 8-connected background, thin the
      result to a skeleton, and record its endpoint count as the legend key.
    - For every grey sprite, apply the same 8-connected fill + skeleton analysis.
    - Recolor the sprite to the legend color with the same branch count; erase sprites
      whose branch count is absent from the legend.

    Key insight: the mapping is topological but no longer based on raw hole count; the
    decisive feature is branch structure after interpreting diagonal bg leaks with
    8-connected hole logic.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        bc = _branch_count(_component_mask_from_points(pts), hole_connectivity=8)
        newc = int(legend[bc]) if bc in legend else bg
        for r, c in pts:
            out[int(r), int(c)] = newc
    return out


# --------------- generator ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 2)
BRANCH_VALUES = [2, 3, 4]
PANEL_SIZE_RANGE = (7, 9)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (5, 7)
MAX_DISTRACTORS = 3
MAX_EXTRA_PER_LEGEND = 1
SPRITE_RETRIES = 90
GEN_RETRIES = 400
SPRITE_MAX_H = 13
SPRITE_MAX_W = 18

_STRUCT8 = generate_binary_structure(2, 2)

_LEGEND_TEMPLATES = {
    2: np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=bool),
    3: np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=bool),
    4: np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=bool),
    5: np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ], dtype=bool),
}

_SENSITIVE_SPRITES = {
    2: np.array([
        [1, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1],
    ], dtype=bool),
    3: np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=bool),
}


def _legend_span(_branch_count_value: int) -> int:
    return 5


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return np.rot90(mask, random.randrange(4))


def _legend_patch(branch_count_value: int, color: int, arrangement: str) -> np.ndarray:
    base = _LEGEND_TEMPLATES[branch_count_value]
    patch = base if arrangement == "horizontal" else np.rot90(base)
    assert _branch_count(patch, hole_connectivity=8) == branch_count_value
    out = np.where(patch, int(color), BG).astype(int)
    return out


def _line_span(legends: list[tuple[int, int]]) -> int:
    return sum(_legend_span(b) for b, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [vals for vals in itertools.combinations(BRANCH_VALUES, count)
              if sum(_legend_span(v) for v in vals) + count - 1 <= max_span]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    values = list(random.choice(combos))
    random.shuffle(values)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(values, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    rr, cc = int(r), int(c)
    for branch_count_value, color in legends:
        patch = _legend_patch(branch_count_value, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 6)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 6)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 6)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 6)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(corner: str, panel_h: int, panel_w: int,
                       arrangement: str, total: int) -> tuple[int, int, int, int, int, int]:
    n = GRID_SIZE
    cross_dim = 5 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 5
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


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    panel_h, panel_w = (cross, along) if arrangement == "horizontal" else (along, cross)
    corner = random.choice(["tl", "tr", "bl", "br"])
    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(corner, panel_h, panel_w, arrangement, total)

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

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    kind = random.choice(["horizontal", "vertical"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    return _layout_full_vertical(grid, blocked)


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0:r1 + 1, c0:c1 + 1]


def _add_random_bumps(mask: np.ndarray, n_bumps: int) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    frontier = []
    for r, c in np.argwhere(out):
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and not out[rr, cc]:
                frontier.append((rr, cc))
    random.shuffle(frontier)
    for r, c in frontier[:n_bumps]:
        out[r, c] = True
    return out


def _carve_diagonal_leak(mask: np.ndarray) -> np.ndarray | None:
    out = mask.copy()
    h, w = out.shape
    centers = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)
               if out[r, c] and out[r - 1, c] and out[r + 1, c] and out[r, c - 1] and out[r, c + 1]]
    random.shuffle(centers)
    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r0, c0 in centers:
        random.shuffle(dirs)
        for dr, dc in dirs:
            rr, cc = r0, c0
            trial = out.copy()
            ok = True
            for _ in range(6):
                if not (0 <= rr < h and 0 <= cc < w) or not trial[rr, cc]:
                    ok = False
                    break
                trial[rr, cc] = False
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    break
                rr += dr
                cc += dc
            if not ok:
                continue
            if not any(trial[0, :]) or not any(trial[-1, :]) or not any(trial[:, 0]) or not any(trial[:, -1]):
                continue
            comps = components_mask(trial, connectivity=4)
            if len(comps) != 1:
                continue
            return trial
    return None


def _make_sprite(branch_count_value: int, require_sensitive: bool) -> np.ndarray | None:
    base = _LEGEND_TEMPLATES[branch_count_value]
    if require_sensitive:
        sensitive = _SENSITIVE_SPRITES.get(branch_count_value)
        if sensitive is None:
            return None
        variants = []
        for flip in (False, True):
            mask = np.fliplr(sensitive) if flip else sensitive
            for rot in range(4):
                trial = np.rot90(mask, rot)
                if _branch_count(trial, 8) == branch_count_value and _branch_count(trial, 4) != branch_count_value:
                    variants.append(trial.copy())
        if not variants:
            return None
        mask = random.choice(variants)
        return np.where(mask, GREY, BG).astype(int)
    for _ in range(SPRITE_RETRIES):
        mask = np.kron(_transform_mask(base), np.ones((2, 2), dtype=bool))
        if random.random() < 0.8:
            mask = binary_dilation(mask, structure=_STRUCT8, iterations=1).astype(bool)
        mask = np.pad(mask, 2)
        mask = _add_random_bumps(mask, random.randint(0, 5))
        if random.random() < 0.8:
            leaked = _carve_diagonal_leak(mask)
            if leaked is not None:
                mask = leaked
        mask = _crop(mask)
        if mask.shape[0] > SPRITE_MAX_H or mask.shape[1] > SPRITE_MAX_W:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        bc8 = _branch_count(mask, hole_connectivity=8)
        if bc8 != branch_count_value:
            continue
        bc4 = _branch_count(mask, hole_connectivity=4)
        if require_sensitive and bc4 == bc8:
            continue
        if np.sum(mask) < 18:
            continue
        return np.where(mask, GREY, BG).astype(int)
    return None


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None, bool]]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    extras = [0] * len(legends)
    extra_budget = min(total - len(legends) - 1, MAX_EXTRA_PER_LEGEND * len(legends))
    for _ in range(max(0, extra_budget)):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        if not choices or random.random() < 0.4:
            break
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None, bool]] = []
    sensitive_choices = [i for i, (branch_count_value, _color) in enumerate(legends)
                         if branch_count_value in _SENSITIVE_SPRITES]
    special_idx = random.choice(sensitive_choices)
    for i, (branch_count_value, color) in enumerate(legends):
        for j in range(1 + extras[i]):
            require_sensitive = i == special_idx and j == 0
            plan.append((branch_count_value, color, require_sensitive))

    remaining = total - len(plan)
    legend_values = {b for b, _ in legends}
    distractor_values = [b for b in BRANCH_VALUES if b not in legend_values]
    assert distractor_values
    n_distractors = min(remaining, random.randint(1, min(MAX_DISTRACTORS, remaining)))
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_values), None, random.random() < 0.5))
    while len(plan) < total:
        b, color = random.choice(legends)
        plan.append((b, color, False))
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
        if np.any(occupied_pad[r:r + h, c:c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    items = []
    for branch_count_value, color, require_sensitive in _sprite_plan(legends):
        sprite = _make_sprite(branch_count_value, require_sensitive=require_sensitive)
        if sprite is None:
            return False
        items.append((int(np.sum(sprite == GREY)), sprite, color))
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
            out_patch = np.where(sprite == GREY, int(color), BG)
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

    # Reject grids where the D8 change is invisible on every grey sprite.
    sensitive = 0
    for pts in components_mask(grid == GREY, connectivity=4):
        mask = _component_mask_from_points(pts)
        if _branch_count(mask, 8) != _branch_count(mask, 4):
            sensitive += 1
    if sensitive == 0:
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
    raise RuntimeError("failed to generate combo variant input")
