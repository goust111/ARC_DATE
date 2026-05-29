# Combo variant of e3721c99 (index 6)
# combo_spec: D2/2+D8/2
# Name: 干扰凹陷精灵+计数凹陷而非洞

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D2/2 + D8/2
# a) Concrete behavior:
#    - The original task colored each grey sprite by matching the number of closed holes to a legend.
#    - In this fused variant, the decisive feature is no longer a closed cavity. It is an open indentation:
#      a 4-connected background notch that reaches the sprite bbox edge, has width 1-2, side walls on both
#      sides, a foreground back wall, and inward depth >= 2.
#    - The legend is changed in the same language. A horizontal legend patch with count 3 now looks like:
#        X.X.X.X
#        X.X.X.X
#        XXXXXXX
#      which has three top-open indentations and zero closed holes.
#    - A grey sprite can simultaneously contain, for example, 2 open indentations and 1 sealed hole.
#      Old logic would read "1"; the fused logic must read "2" and recolor by the 2-notch legend entry.
# b) Complexity calibration:
#    - Counts are kept in a small discrete range 1-4 for legend entries, so 2-4 training examples can
#      repeatedly support the mapping.
#    - Grey sprites deliberately also contain sealed holes chosen independently from the indentation count,
#      so the fusion does not collapse back to the original hole-count task.
#    - Different sprites with the same indentation count use different notch sides and sizes, making it
#      visible that geometry varies while count stays decisive.
# c) Hidden assumptions in the original main():
#    - `_hole_count(sub)` assumed the relevant feature was a non-border-touching background component.
#      D8/2 breaks that assumption completely: all legend notches are border-touching, so old legend
#      parsing collapses to repeated hole_count=0 and becomes unusable.
#    - `_legend_mapping()` assumed the same "count holes" primitive explained both the legend and sprites.
#      In the fused task this remains true only after replacing holes with the new indentation detector.
#    - Recoloring per grey connected component is still valid; D2/2 changes internal geometry, not the
#      fact that each sprite is one connected grey component.
# d) Placement legality:
#    - Divider and legend-panel placement from the source task are preserved.
#    - Sprites are placed only in the workspace, with 1-cell padded non-overlap via 8-neighbor dilation.
#    - Sprite construction uses a filled rectangle base, then carves edge notches and internal holes in
#      non-overlapping slots, so components stay connected and never exceed the canvas bounds.
# e) Visual observability:
#    - Every accepted grid has legend patches whose hole counts are all 0 while their indentation counts
#      are distinct and positive; a solver must therefore notice open notches in the legend.
#    - Grey sprites also contain misleading sealed holes, so "count holes" and "count indentations"
#      disagree on visible objects inside the same sample.
# f) Visual cleanliness:
#    - Each indentation is a single narrow rectangular notch. No extra markers are added.
#    - The legend uses one-sided repetitive notches, while sprites mix sides and sizes; both remain easy
#      to parse at ARC scale without visual clutter.
# g) Color inference:
#    - Colors are still inferred structurally from the legend panel only. No ordering by color id is used.
#    - Multiple sprites map to the same legend count, giving repeated support that count, not exact shape,
#      determines recoloring.
# h) Fusion-specific behavior metrics:
#    - `legend_hole_collision`: all positive-count legend patches have closed-hole count 0, so the old
#      hole-based legend interpretation is non-injective.
#    - `sprite_conflicts`: the number of grey sprites with `indentations != holes` must be positive and
#      is typically all sprites in generated samples.
# i) New solver assumptions and generator guarantees:
#    - The indentation detector recognizes axis-aligned deep notches of width 1-2 and depth >= 2.
#    - `generate_input()` explicitly constructs only this family of open recesses for both legend and
#      grey sprites, and validates every finished grid with the new `main()`.


BG = 0
DIV = 1
GREY = 5

GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
MAX_INDENT_COUNT = 4
MAX_HOLE_COUNT = 4
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (6, 8)
MAX_DISTRACTORS = 5
MAX_EXTRA_PER_LEGEND = 1
SPRITE_RETRIES = 80
GEN_RETRIES = 300

SPRITE_H_RANGE = (7, 9)
SPRITE_W_RANGE = (7, 9)
NOTCH_WIDTH_CHOICES = (1, 2)
NOTCH_DEPTH_CHOICES = (2, 3)
_SIDES = ("top", "bottom", "left", "right")
_STRUCT8 = generate_binary_structure(2, 2)
_SPRITE_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _component_submask(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _edge_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        if not bool(line[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(line[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _hole_count(mask: np.ndarray) -> int:
    """
    Original-task helper: count closed 4-connected holes in a foreground mask.
    """
    sub = np.asarray(mask, dtype=bool)
    h, w = sub.shape
    bg = ~sub
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            stack = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                if r > 0 and bg[r - 1, c] and not vis[r - 1, c]:
                    vis[r - 1, c] = True
                    stack.append((r - 1, c))
                if r + 1 < h and bg[r + 1, c] and not vis[r + 1, c]:
                    vis[r + 1, c] = True
                    stack.append((r + 1, c))
                if c > 0 and bg[r, c - 1] and not vis[r, c - 1]:
                    vis[r, c - 1] = True
                    stack.append((r, c - 1))
                if c + 1 < w and bg[r, c + 1] and not vis[r, c + 1]:
                    vis[r, c + 1] = True
                    stack.append((r, c + 1))
            if not touches:
                holes += 1
    return holes


def _indentation_count(mask: np.ndarray) -> int:
    """
    Count deep open indentations in a tight foreground mask.

    An indentation is a rectangular 4-connected bg notch that:
    - touches exactly one side of the tight bbox,
    - has mouth width 1-2,
    - has depth >= 2,
    - keeps foreground side walls for the whole depth,
    - and is closed by a foreground back wall.
    """
    sub = np.asarray(mask, dtype=bool)
    h, w = sub.shape
    count = 0

    for c0, c1 in _edge_runs(~sub[0]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            if sub[depth, c0 : c1 + 1].any():
                break
            if not bool(sub[depth, c0 - 1]) or not bool(sub[depth, c1 + 1]):
                break
            depth += 1
        if depth >= 2 and depth < h and sub[depth, c0 : c1 + 1].all():
            count += 1

    for c0, c1 in _edge_runs(~sub[h - 1]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            r = h - 1 - depth
            if sub[r, c0 : c1 + 1].any():
                break
            if not bool(sub[r, c0 - 1]) or not bool(sub[r, c1 + 1]):
                break
            depth += 1
        close_r = h - 1 - depth
        if depth >= 2 and close_r >= 0 and sub[close_r, c0 : c1 + 1].all():
            count += 1

    for r0, r1 in _edge_runs(~sub[:, 0]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            if sub[r0 : r1 + 1, depth].any():
                break
            if not bool(sub[r0 - 1, depth]) or not bool(sub[r1 + 1, depth]):
                break
            depth += 1
        if depth >= 2 and depth < w and sub[r0 : r1 + 1, depth].all():
            count += 1

    for r0, r1 in _edge_runs(~sub[:, w - 1]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            c = w - 1 - depth
            if sub[r0 : r1 + 1, c].any():
                break
            if not bool(sub[r0 - 1, c]) or not bool(sub[r1 + 1, c]):
                break
            depth += 1
        close_c = w - 1 - depth
        if depth >= 2 and close_c >= 0 and sub[r0 : r1 + 1, close_c].all():
            count += 1

    return count


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg, grey) color and map indentation_count -> color.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
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
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == c
        ic = _indentation_count(sub)
        assert ic not in mapping, f"legend indentation count {ic} must map to exactly one color"
        mapping[ic] = c
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching open indentation counts to the legend.

    Input structure:
    - The background is the most common color; grey (5) sprites lie in the workspace, separated from a
      colored legend panel by a divider.
    - Legend patches are compact colored components away from the outer border. They no longer encode
      closed holes: each patch shows a positive number of edge-open recesses.
    - Grey sprites may also contain sealed holes, but those holes are distractions rather than the
      matching feature.

    Rule:
    - Read each legend patch as indentation_count -> color, where an indentation is an open notch of
      depth at least 2 that enters from one bbox side and closes inside the shape.
    - For each grey sprite, count those open indentations in its tight bbox and ignore closed holes.
    - Recolor the sprite with the legend color for the same indentation count; erase grey sprites whose
      counts are absent from the legend.

    Key insight:
    - The decisive structure is no longer enclosed topology. It is the number of deep edge recesses, so
      the solver must distinguish open notches from genuine holes and follow the legend written in that
      same open-notch language.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        sub = _component_submask(pts)
        ic = _indentation_count(sub)
        newc = int(legend[ic]) if ic in legend else bg
        for r, c in pts:
            out[r, c] = newc
    return out


def _legend_span(indent_count: int) -> int:
    return 3 if indent_count == 0 else 2 * indent_count + 1


def _legend_patch(indent_count: int, color: int, arrangement: str) -> np.ndarray:
    span = _legend_span(indent_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if indent_count:
            patch[:2, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if indent_count:
            patch[1::2, :2] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    return sum(_legend_span(i) for i, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    del arrangement
    options: list[tuple[int, tuple[int, ...]]] = []
    for count in range(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1):
        for vals in itertools.combinations(range(1, MAX_INDENT_COUNT + 1), count):
            if sum(_legend_span(i) for i in vals) + count - 1 <= max_span:
                options.append((count, vals))
    assert options, "legend settings must admit at least one count combination"
    _, picked = random.choice(options)
    counts = list(picked)
    random.shuffle(counts)
    colors = random.sample(LEGEND_COLORS, len(counts))
    return list(zip(counts, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    rr, cc = int(r), int(c)
    for indent_count, color in legends:
        patch = _legend_patch(indent_count, color, arrangement)
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
        r0 = random.randint(1, panel_h - 4)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
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
        c0 = random.randint(1, panel_w - 4)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    n = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

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
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total
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

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _choose_hole_count(indent_count: int, legend_counts: set[int], matched: bool) -> int:
    candidates = [h for h in range(MAX_HOLE_COUNT + 1) if h != indent_count]
    if matched:
        preferred = [h for h in candidates if h == 0 or h in legend_counts]
    else:
        preferred = [h for h in candidates if h in legend_counts]
    return random.choice(preferred or candidates)


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None, int]]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None, int]] = []
    legend_counts = {i for i, _ in legends}
    for idx, (indent_count, color) in enumerate(legends):
        for _ in range(1 + extras[idx]):
            hole_count = _choose_hole_count(indent_count, legend_counts, matched=True)
            plan.append((indent_count, color, hole_count))

    distractor_counts = [i for i in range(MAX_INDENT_COUNT + 1) if i not in legend_counts]
    positive_distractors = [i for i in distractor_counts if i > 0]
    for _ in range(n_distractors):
        indent_count = random.choice(positive_distractors or distractor_counts)
        hole_count = _choose_hole_count(indent_count, legend_counts, matched=False)
        plan.append((indent_count, None, hole_count))
    random.shuffle(plan)
    return plan


def _reserve(buffer_mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = buffer_mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    buffer_mask[rr0:rr1, cc0:cc1] = True


def _carve_notch(mask: np.ndarray, side: str, start: int, width: int, depth: int) -> None:
    h, w = mask.shape
    if side == "top":
        mask[:depth, start : start + width] = False
    elif side == "bottom":
        mask[h - depth :, start : start + width] = False
    elif side == "left":
        mask[start : start + width, :depth] = False
    else:
        mask[start : start + width, w - depth :] = False


def _valid_notch_starts(side: str, width: int, h: int, w: int) -> list[int]:
    if side in ("top", "bottom"):
        hi = w - width - 2
        return list(range(2, hi + 1)) if hi >= 2 else []
    hi = h - width - 2
    return list(range(2, hi + 1)) if hi >= 2 else []


def _place_holes(mask: np.ndarray, hole_count: int, reserved: np.ndarray) -> bool:
    h, w = mask.shape
    placed = 0
    attempts = 0
    while placed < hole_count and attempts < 400:
        attempts += 1
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        if reserved[r, c] or not mask[r, c]:
            continue
        if not (
            mask[r - 1, c]
            and mask[r + 1, c]
            and mask[r, c - 1]
            and mask[r, c + 1]
        ):
            continue
        rr0 = max(0, r - 1)
        rr1 = min(h, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + 2)
        if not np.all(mask[rr0:rr1, cc0:cc1]):
            continue
        mask[r, c] = False
        _reserve(reserved, r, c, r, c)
        placed += 1
    return placed == hole_count


def _base_dims(indent_count: int, hole_count: int) -> tuple[int, int]:
    min_h = max(SPRITE_H_RANGE[0], 7 + int(hole_count >= 2))
    min_w = max(SPRITE_W_RANGE[0], 7 + int(hole_count >= 3))
    if indent_count >= 3:
        min_h = max(min_h, 8)
        min_w = max(min_w, 8)
    h = random.randint(min_h, SPRITE_H_RANGE[1])
    w = random.randint(min_w, SPRITE_W_RANGE[1])
    return h, w


def _make_sprite(indent_count: int, hole_count: int) -> np.ndarray | None:
    key = (indent_count, hole_count)
    cache = _SPRITE_CACHE.setdefault(key, [])
    if cache and random.random() < 0.8:
        return random.choice(cache).copy()

    for _ in range(SPRITE_RETRIES):
        h, w = _base_dims(indent_count, hole_count)
        mask = np.ones((h, w), dtype=bool)
        reserved = np.zeros((h, w), dtype=bool)

        sides = random.sample(_SIDES, indent_count)
        ok = True
        for side in sides:
            width = random.choice(NOTCH_WIDTH_CHOICES)
            starts = _valid_notch_starts(side, width, h, w)
            if not starts:
                ok = False
                break
            start = random.choice(starts)
            depth = random.choice(NOTCH_DEPTH_CHOICES)
            _carve_notch(mask, side, start, width, depth)
            if side == "top":
                _reserve(reserved, 0, start, depth - 1, start + width - 1)
            elif side == "bottom":
                _reserve(reserved, h - depth, start, h - 1, start + width - 1)
            elif side == "left":
                _reserve(reserved, start, 0, start + width - 1, depth - 1)
            else:
                _reserve(reserved, start, w - depth, start + width - 1, w - 1)
        if not ok:
            continue

        if not _place_holes(mask, hole_count, reserved):
            continue
        if _indentation_count(mask) != indent_count:
            continue
        if _hole_count(mask) != hole_count:
            continue

        sprite = np.where(mask, GREY, BG).astype(int)
        if len(cache) < 12:
            cache.append(sprite.copy())
        return sprite
    return None


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    items = []
    for indent_count, color, hole_count in _sprite_plan(legends):
        sprite = _make_sprite(indent_count, hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, color))
    items.sort(key=lambda x: x[0], reverse=True)
    base_grid = grid.copy()
    base_expected = expected.copy()

    for _ in range(24):
        work_grid = base_grid.copy()
        work_expected = base_expected.copy()
        occupied = np.zeros_like(grid)
        trial_items = items[:]
        random.shuffle(trial_items)
        trial_items.sort(key=lambda x: x[0], reverse=True)
        success = True
        for _, sprite, color in trial_items:
            loc = _random_sprite_location(blocked, occupied, sprite)
            if loc is None:
                success = False
                break
            r, c = loc
            paste(work_grid, sprite, r, c, transparent=BG)
            paste(occupied, sprite, r, c, transparent=BG)
            if color is not None:
                out_patch = np.where(sprite == GREY, int(color), BG)
                paste(work_expected, out_patch, r, c, transparent=BG)
        if success:
            grid[:, :] = work_grid
            expected[:, :] = work_expected
            return True
    return False


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_indent_counts: list[int] = []
    legend_hole_counts: list[int] = []
    for color in [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, GREY)]:
        comps = components_mask(g == color, connectivity=4)
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
            legend_indent_counts.append(_indentation_count(sub))
            legend_hole_counts.append(_hole_count(sub))
    sprite_indent_counts: list[int] = []
    sprite_hole_counts: list[int] = []
    for pts in components_mask(g == GREY, connectivity=4):
        sub = _component_submask(pts)
        sprite_indent_counts.append(_indentation_count(sub))
        sprite_hole_counts.append(_hole_count(sub))
    return {
        "legend_indent_counts": legend_indent_counts,
        "legend_hole_counts": legend_hole_counts,
        "legend_hole_collision": len(set(legend_hole_counts)) < len(legend_hole_counts),
        "sprite_indent_counts": sprite_indent_counts,
        "sprite_hole_counts": sprite_hole_counts,
        "sprite_conflicts": sum(int(i != h) for i, h in zip(sprite_indent_counts, sprite_hole_counts)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    expected = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    metrics = _behavior_metrics(grid)
    if not metrics["legend_hole_collision"]:
        return None
    if int(metrics["sprite_conflicts"]) < max(2, len(metrics["sprite_indent_counts"]) // 2):
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
    raise RuntimeError("failed to generate e3721c99 combo D2/2+D8/2 input")
