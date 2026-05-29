# Variant of e3721c99
# Dimension: D10 - 图例形状的视觉复杂度 (meta)
# Change: 图例形状为小型彩色精灵，自身带有明确可数的洞... -> 图例形状为嵌套结构（洞中有洞）

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete appearance and behavior trace
#    - Grey workspace sprites stay close to the original task: each is a single connected grey blob
#      whose ordinary hole count is the feature that determines its output color.
#    - The change is only on the legend side. Every legend patch is now a connected monochrome shape
#      made of 1-3 "nested chambers". Each chamber has:
#        1. one outer cavity, and
#        2. one smaller cavity inside a same-color island attached to the outer frame by a thin stem.
#      So a legend with k chambers has total enclosed background components = 2*k, but the number of
#      top-level cavities visible from the outside of the legend body is only k.
#    - Small trace:
#        old logic on legend: count all holes -> chamber counts become 2, 4, 6
#        new logic on legend: count only cavities not contained inside another cavity -> 1, 2, 3
#        grey sprites are still counted by ordinary total holes -> a 2-hole grey blob matches the legend
#        whose patch has two chambers, not the one whose patch has four enclosed background components.
#    - This makes the variant visible: humans see "hole inside a hole" in the legend, and must infer
#      that the mapping uses the outer chambers, not every enclosed void at every depth.
#
# b) Complexity calibration
#    - Legend values are limited to 1-3. That keeps the nested motif readable and allows repeated support
#      across 2-3 legend entries in one sample.
#    - Zero-hole legends are removed on purpose. A zero-hole legend would not show the new nested behavior,
#      so it would weaken D10 and allow degeneration back to the original task family.
#    - Sprite counts are trimmed slightly because the legend panel is taller / wider than before.
#
# c) Old implicit assumptions that break here
#    - Original _legend_mapping assumed "legend value = total hole count of a monochrome component".
#      That fails immediately once a legend chamber contains its own inner hole.
#    - Original generator encoded the legend using tiny 3-cell-thick bars with simple perforations.
#      Those patches cannot show nested topology, so the panel geometry must be enlarged.
#    - Original solver used the same hole counter for legends and grey sprites. In this variant the two
#      measurements diverge: legends need top-level-hole counting, sprites still use ordinary hole counting.
#
# d) Placement legality
#    - Each legend color appears exactly once as one interior component, never on the border.
#    - Divider color stays fixed and legend colors exclude it, so touching the divider cannot merge a legend
#      with the border structure.
#    - Grey sprites are generated and placed only in the workspace, with a 1-cell moat via dilated occupancy.
#
# e) Visual observability
#    - Every legend patch is forced to have nested cavities, so the new rule is not optional noise.
#    - Because the output still recolors ordinary grey sprites, the only plausible explanation is that one
#      must read the legend's chamber count in a hierarchical way.
#
# f) Visual cleanliness
#    - Panels use only 2 rows / 2 columns of patches with fixed 1-cell gaps.
#    - Legend colors are few and distinct; divider is kept separate; sprites never overlap or touch.
#
# g) Color-change inferability
#    - Colors are not ordered or transformed by id. They merely label legend entries.
#    - The structural cue is the repeated pairing: all grey sprites with N ordinary holes receive the color
#      of the legend whose nested patch has N outer chambers.
#
# h) New main() assumptions and generator guarantees
#    - main() assumes each legend cavity tree is simple enough that nested inner holes are strictly contained
#      in the bounding box of their parent outer cavity; generate_input() builds exactly that motif.
#    - main() assumes grey sprites do not require hierarchical counting; generate_input() still uses the
#      original blob generator, which only creates flat hole sets.


# ---------------- topology helpers ----------------

def _hole_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bboxes of enclosed background components inside a binary mask."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            pts = []
            touches = False
            while st:
                r, c = st.pop()
                pts.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                arr = np.asarray(pts, dtype=int)
                boxes.append(bbox_coords(arr))
    return boxes


def _hole_count(mask: np.ndarray) -> int:
    """Ordinary hole count: every enclosed background component contributes 1."""
    return len(_hole_boxes(mask))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer != inner
        and outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _top_level_hole_count(mask: np.ndarray) -> int:
    """
    Count only depth-1 cavities.

    For the generated nested legends, every inner hole sits inside the bbox of its parent chamber cavity,
    while sibling chambers have disjoint bboxes. Top-level holes are therefore exactly the enclosed
    background components whose bbox is not contained in another enclosed background component bbox.
    """
    boxes = _hole_boxes(mask)
    return sum(
        1
        for i, box in enumerate(boxes)
        if not any(_strictly_contains(other, box) for j, other in enumerate(boxes) if i != j)
    )


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg,grey) color and map top-level-hole-count -> color.
    """
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
            if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        hc = _top_level_hole_count(sub)
        assert hc not in mapping, f"legend top-level hole count {hc} must map to one color"
        mapping[hc] = int(color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching them to nested legend shapes.

    Input structure:
    - The background is the most common color, and grey (5) blobs are scattered in the workspace.
    - A divider marks off a legend panel containing large colored components that visibly contain
      nested cavities: each outer chamber also contains a smaller hole inside an attached island.

    Rule:
    - Each colored legend defines a mapping from its number of top-level chambers to its color.
    - For every grey sprite, count its ordinary enclosed holes inside its own bounding box.
    - Recolor the grey sprite with the color of the legend whose chamber count matches that hole count;
      erase grey distractors whose counts are absent from the legend.

    Key insight:
    - The legend is no longer read by counting every visible void. The correct signature is the number
      of outer cavities, while the extra inner holes are nested decoration that must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "grid must contain at least one nested legend component"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        newc = int(legend[hc]) if hc in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 3)
LEGEND_VALUES = (1, 2, 3)
FULL_PANEL_THICKNESS = 17
L_PANEL_WIDE = (17, 21)   # (height, width) for horizontal legend packing
L_PANEL_TALL = (21, 17)   # (height, width) for vertical legend packing

# -- sprites --
TOTAL_SPRITES_RANGE = (7, 9)
MAX_DISTRACTORS = 5
MAX_EXTRA_PER_LEGEND = 2
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 22)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 11
SPRITE_MAX_W = 16
SPRITE_MAX_ASPECT = 2.5
SPRITE_HOLE_VALUES = (0, 1, 2, 3, 4)

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


# ---------- legend helpers ----------

def _legend_patch(top_holes: int, color: int, arrangement: str) -> np.ndarray:
    """
    Build a connected nested legend.

    Horizontal form: 7 x (6*k+1), made of k shared-wall chambers.
    Each chamber contributes one outer cavity and one inner cavity, so total holes = 2*k.
    """
    k = int(top_holes)
    patch = np.zeros((7, 6 * k + 1), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    for wall in range(0, 6 * k + 1, 6):
        patch[:, wall] = color
    for i in range(k):
        base = 6 * i
        # Stem connecting the inner ring to the outer frame.
        patch[1, base + 3] = color
        # 3x3 ring around the nested inner hole.
        patch[2:5, base + 2 : base + 5] = color
        patch[3, base + 3] = BG
    if arrangement == "vertical":
        patch = patch.T
    mask = patch == color
    assert _hole_count(mask) == 2 * k
    assert _top_level_hole_count(mask) == k
    return patch


def _choose_legends(count: int) -> list[tuple[int, int]]:
    holes = random.sample(list(LEGEND_VALUES), count)
    colors = random.sample(LEGEND_COLORS, count)
    random.shuffle(holes)
    return list(zip(holes, colors))


def _place_horizontal_rows(
    grid: np.ndarray, legends: list[tuple[int, int]], panel_r0: int, panel_c0: int, panel_w: int
) -> None:
    entries = [(holes, color, _legend_patch(holes, color, "horizontal")) for holes, color in legends]
    row_a = panel_r0 + 1
    row_b = panel_r0 + 9
    if len(entries) == 2:
        total_w = entries[0][2].shape[1] + 1 + entries[1][2].shape[1]
        if total_w <= panel_w - 2 and random.random() < 0.5:
            row = random.choice([row_a, row_b])
            order = entries[:]
            random.shuffle(order)
            start = panel_c0 + random.randint(1, panel_w - 1 - total_w)
            paste(grid, order[0][2], row, start)
            paste(grid, order[1][2], row, start + order[0][2].shape[1] + 1)
            return
        rows = [row_a, row_b]
        random.shuffle(rows)
        for row, entry in zip(rows, entries):
            patch = entry[2]
            start = panel_c0 + random.randint(1, panel_w - 1 - patch.shape[1])
            paste(grid, patch, row, start)
        return

    entries.sort(key=lambda item: item[2].shape[1], reverse=True)
    single_row, double_row = random.choice([(row_a, row_b), (row_b, row_a)])
    single = entries[0][2]
    start = panel_c0 + random.randint(1, panel_w - 1 - single.shape[1])
    paste(grid, single, single_row, start)
    pair = entries[1:]
    random.shuffle(pair)
    total_w = pair[0][2].shape[1] + 1 + pair[1][2].shape[1]
    start = panel_c0 + random.randint(1, panel_w - 1 - total_w)
    paste(grid, pair[0][2], double_row, start)
    paste(grid, pair[1][2], double_row, start + pair[0][2].shape[1] + 1)


def _place_vertical_cols(
    grid: np.ndarray, legends: list[tuple[int, int]], panel_r0: int, panel_c0: int, panel_h: int
) -> None:
    entries = [(holes, color, _legend_patch(holes, color, "vertical")) for holes, color in legends]
    col_a = panel_c0 + 1
    col_b = panel_c0 + 9
    if len(entries) == 2:
        total_h = entries[0][2].shape[0] + 1 + entries[1][2].shape[0]
        if total_h <= panel_h - 2 and random.random() < 0.5:
            col = random.choice([col_a, col_b])
            order = entries[:]
            random.shuffle(order)
            start = panel_r0 + random.randint(1, panel_h - 1 - total_h)
            paste(grid, order[0][2], start, col)
            paste(grid, order[1][2], start + order[0][2].shape[0] + 1, col)
            return
        cols = [col_a, col_b]
        random.shuffle(cols)
        for col, entry in zip(cols, entries):
            patch = entry[2]
            start = panel_r0 + random.randint(1, panel_h - 1 - patch.shape[0])
            paste(grid, patch, start, col)
        return

    entries.sort(key=lambda item: item[2].shape[0], reverse=True)
    single_col, double_col = random.choice([(col_a, col_b), (col_b, col_a)])
    single = entries[0][2]
    start = panel_r0 + random.randint(1, panel_h - 1 - single.shape[0])
    paste(grid, single, start, single_col)
    pair = entries[1:]
    random.shuffle(pair)
    total_h = pair[0][2].shape[0] + 1 + pair[1][2].shape[0]
    start = panel_r0 + random.randint(1, panel_h - 1 - total_h)
    paste(grid, pair[0][2], start, double_col)
    paste(grid, pair[1][2], start + pair[0][2].shape[0] + 1, double_col)


# ---------- layout helpers ----------

def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends(random.randint(*LEGEND_COUNT_RANGE))
    panel_h = FULL_PANEL_THICKNESS
    top = random.choice([True, False])
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        _place_horizontal_rows(grid, legends, 0, 0, GRID_SIZE)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        _place_horizontal_rows(grid, legends, div_r + 1, 0, GRID_SIZE)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends(random.randint(*LEGEND_COUNT_RANGE))
    panel_w = FULL_PANEL_THICKNESS
    left = random.choice([True, False])
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        _place_vertical_cols(grid, legends, 0, 0, GRID_SIZE)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        _place_vertical_cols(grid, legends, 0, div_c + 1, GRID_SIZE)
    return legends


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(2)
    if arrangement == "horizontal":
        panel_h, panel_w = L_PANEL_WIDE
    else:
        panel_h, panel_w = L_PANEL_TALL
    corner = random.choice(["tl", "tr", "bl", "br"])
    n = GRID_SIZE

    if corner == "tl":
        div_r, div_c = panel_h, panel_w
        grid[div_r, : div_c + 1] = DIV
        grid[: div_r + 1, div_c] = DIV
        blocked[:div_r, :div_c] = 9
        blocked[div_r, : div_c + 1] = 9
        blocked[: div_r + 1, div_c] = 9
        panel_r0, panel_c0 = 0, 0
    elif corner == "tr":
        div_r, div_c = panel_h, n - 1 - panel_w
        grid[div_r, div_c:] = DIV
        grid[: div_r + 1, div_c] = DIV
        blocked[:div_r, div_c + 1 :] = 9
        blocked[div_r, div_c:] = 9
        blocked[: div_r + 1, div_c] = 9
        panel_r0, panel_c0 = 0, div_c + 1
    elif corner == "bl":
        div_r, div_c = n - 1 - panel_h, panel_w
        grid[div_r, : div_c + 1] = DIV
        grid[div_r:, div_c] = DIV
        blocked[div_r + 1 :, :div_c] = 9
        blocked[div_r, : div_c + 1] = 9
        blocked[div_r:, div_c] = 9
        panel_r0, panel_c0 = div_r + 1, 0
    else:
        div_r, div_c = n - 1 - panel_h, n - 1 - panel_w
        grid[div_r, div_c:] = DIV
        grid[div_r:, div_c] = DIV
        blocked[div_r + 1 :, div_c + 1 :] = 9
        blocked[div_r, div_c:] = 9
        blocked[div_r:, div_c] = 9
        panel_r0, panel_c0 = div_r + 1, div_c + 1

    if arrangement == "horizontal":
        _place_horizontal_rows(grid, legends, panel_r0, panel_c0, panel_w)
    else:
        _place_vertical_cols(grid, legends, panel_r0, panel_c0, panel_h)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


# ---------- sprite generation ----------

def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
    margin = HOLE_SEED_MARGIN
    max_hole = (
        max(2, area // (hole_count + 2))
        if hole_count <= 2
        else max(1, area // (hole_count * 2 + 1))
    )
    forbidden: set[tuple[int, int]] = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max(1, max_hole))
        while len(blob) < target:
            cands = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any(
                        (nnr, nnc) in forbidden and (nnr, nnc) not in blob
                        for nnr, nnc in neighbors4(nr, nc)
                    ):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            p = random.choice(cands)
            forbidden.add(p)
            blob.add(p)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    if (center, center) not in forbidden:
        return center, center
    for d in range(1, csz):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(
    csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros((csz, csz), dtype=bool)
    sr, sc = start
    mask[sr, sc] = True
    frontier = list(neighbors4(sr, sc))
    filled = 1
    while filled < area and frontier:
        idx = random.randrange(len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        r, c = frontier.pop()
        if not (0 <= r < csz and 0 <= c < csz) or mask[r, c] or (r, c) in forbidden:
            continue
        mask[r, c] = True
        filled += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc]:
                frontier.append((nr, nc))
    return mask


def _seal_and_connect(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> np.ndarray:
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc] and (nr, nc) not in forbidden:
                    mask[nr, nc] = True
                    sealed = True
        if not sealed:
            break
    if np.any(mask):
        labeled, n_comp = label_mask(mask)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda label: int(np.sum(labeled == label)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    for r, c in forbidden:
        rmin, cmin = min(rmin, r), min(cmin, c)
        rmax, cmax = max(rmax, r), max(cmax, c)
    h, w = rmax - rmin + 1, cmax - cmin + 1
    if h > SPRITE_MAX_H or w > SPRITE_MAX_W or max(h, w) > SPRITE_MAX_ASPECT * min(h, w):
        return None
    cropped = mask[rmin : rmax + 1, cmin : cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

        forbidden: set[tuple[int, int]] = set()
        if hole_count > 0:
            seeds = _place_hole_seeds(hole_count, center, csz, area)
            if seeds is None:
                continue
            forbidden = _grow_hole_blobs(seeds, hole_count, area, csz)

        start = _find_start(center, csz, forbidden)
        mask = _grow_mask(csz, area, start, forbidden)
        mask = _seal_and_connect(mask, forbidden)

        result = _crop_and_validate(mask, forbidden, hole_count)
        if result is not None:
            return result
    return None


# ---------- sprite planning & placement ----------

def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d
        for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None]] = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in SPRITE_HOLE_VALUES if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
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
    for hole_count, color in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, color))
    items.sort(key=lambda item: item[0], reverse=True)

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


# ---------- top-level generation ----------

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
    raise RuntimeError("failed to generate e3721c99 D10 input")
