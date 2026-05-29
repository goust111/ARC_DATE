# Combo variant of e3721c99 (index 38)
# combo_spec: D2/1+D10/2
# Name: 中空框架精灵+嵌套洞图例

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete appearance and behavior trace
#    - Original task: grey workspace objects are dense blobs with 0-5 ordinary holes, and legend patches are
#      tiny same-color bars whose total hole count directly gives the recolor mapping.
#    - Combo task: the workspace objects become one-pixel-thick hollow "window frames". A 3-hole sprite is not
#      a fat blob with three cavities; it is a thin outer rectangle whose interior is split by two thin divider
#      bars into three enclosed chambers. The legend side changes independently: each colored legend is now a
#      large nested structure where every visible chamber contains a same-color island with another hole inside.
#    - Small trace:
#        old legend reading: total holes in legend {1,2,3}
#        combo legend reading: total holes in legend {2,4,6}, but top-level chamber counts {1,2,3}
#        old sprite reading: dense 2-hole blob -> 2
#        combo sprite reading: thin 2-chamber frame -> still ordinary-hole count 2
#      The solver must therefore count frame cavities on the sprite side, yet ignore depth-2 holes on the
#      legend side. Matching still happens by chamber count, but the two sides demand different topology logic.
#
# b) Complexity calibration
#    - Legend values are restricted to 1-3 so the nested motif stays readable inside a 30x30 grid and each
#      sample can show repeated support for the same mapping.
#    - Grey frame sprites use hole counts 1-4. Values 1-3 can match legends; value 4 is a natural distractor.
#    - Sprite count is kept moderate because both the panel and the frame bboxes are larger than in the source.
#
# c) Old implicit assumptions that break here
#    - Original _legend_mapping assumed "legend hole count = ordinary enclosed-background count". That breaks
#      because each legend chamber now contains a nested inner hole that should not change the mapping value.
#    - Original sprite generator assumed thick organic blobs. That would visually weaken D2/1, so the generator
#      must instead build thin frame networks with no 2x2 filled blocks.
#    - Original solver used the same hole counter for legends and sprites. The combo keeps ordinary counting for
#      grey sprites but switches legends to top-level-hole counting.
#
# d) Placement legality
#    - Legends are placed wholly inside a reserved panel that is separated by a divider; legend colors exclude
#      the divider color, so border contact cannot merge them with the divider.
#    - Grey sprites are placed only in the workspace, with a 1-cell moat enforced by 8-neighborhood dilation.
#    - Frame generators choose sizes that fit the available workspace and stay well within the original bbox
#      limits, so random placement remains feasible.
#
# e) Visual observability
#    - D2/1 is visible because every grey component is sparse and skeletal: a human sees frames/chambers instead
#      of solid blobs.
#    - D10/2 is visible because every legend contains "hole inside a hole" chambers. Across examples, matching
#      outputs show that the correct legend value is the count of outer chambers, not the total number of voids.
#
# f) Visual cleanliness
#    - Legend panels use one or two rows/columns with fixed gaps; sprites never overlap and never touch the
#      divider or each other.
#    - The frame sprites are monochrome, rectilinear, and separated by background, so their chambers remain easy
#      to distinguish even when several are present.
#
# g) Color-change inferability
#    - Colors are just labels for legend entries; they are not ordered numerically.
#    - Each legend hole value appears on at least one grey sprite, usually more than one, so humans can infer
#      that "same chamber count -> same recolor" from repeated examples rather than from a one-off coincidence.
#
# h) New main() assumptions and generator guarantees
#    - _top_level_hole_count assumes nested legend holes have bbox containment: inner holes lie strictly inside
#      their parent outer cavities. The legend generator constructs exactly that motif.
#    - main() assumes each grey component can be counted inside its own bbox without interference from neighbors.
#      The generator keeps every grey sprite as a separate 4-connected component with a background moat.
#    - main() assumes legend colors appear as exactly one interior component per color. The generator uses unique
#      legend colors and never reuses those colors elsewhere in the input.


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

    The generated legends are chamber trees with one inner hole strictly inside each outer cavity. That means
    top-level chambers are exactly the enclosed background components whose bbox is not contained in another
    enclosed background component bbox.
    """
    boxes = _hole_boxes(mask)
    return sum(
        1
        for i, box in enumerate(boxes)
        if not any(_strictly_contains(other, box) for j, other in enumerate(boxes) if i != j)
    )


def _has_full_2x2(mask: np.ndarray) -> bool:
    """Thin frame sprites should never contain a filled 2x2 block."""
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """Pick one interior legend component per color and map top-level-hole-count -> color."""
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
    Recolor grey hollow-frame sprites by matching them to nested legend shapes.

    Input structure:
    - The background is the most common color, and grey (5) workspace objects are thin connected frame networks.
    - Each grey sprite is a hollow rectangular frame whose interior is split into 1-4 enclosed chambers by thin
      divider bars, so the relevant sprite feature is the number of enclosed frame compartments.
    - A separated legend panel contains larger colored components with nested cavities: each visible outer chamber
      also contains a smaller same-color island that encloses a second, deeper hole.

    Rule:
    - Each colored legend maps its number of top-level chambers to its color; the deeper nested holes do not
      create new legend values.
    - For every grey frame sprite, count ordinary enclosed holes inside its own bounding box.
    - Recolor the sprite to the legend color with the same chamber count, and erase grey distractors whose counts
      are not represented in the legend.

    Key insight:
    - Both sides require hole reasoning, but not the same hole rule. Grey sprites are matched by all enclosed
      frame chambers, while legends must be read hierarchically so only the outer chambers count.
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

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 3)
LEGEND_VALUES = (1, 2, 3)
FULL_PANEL_THICKNESS = 17
L_PANEL_WIDE = (17, 21)
L_PANEL_TALL = (21, 17)

# -- grey frame sprites --
TOTAL_SPRITES_RANGE = (6, 8)
MAX_DISTRACTORS = 3
MAX_EXTRA_PER_LEGEND = 2
SPRITE_RETRIES = 20
FRAME_HOLE_VALUES = (1, 2, 3, 4)
FRAME_INNER_SPAN = (2, 3)
FRAME_CROSS_SPAN = (2, 4)
FRAME_MAX_H = 11
FRAME_MAX_W = 16

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


# ---------- legend helpers ----------

def _legend_patch(top_holes: int, color: int, arrangement: str) -> np.ndarray:
    """
    Build a connected nested legend.

    Horizontal form: 7 x (6*k+1), made of k chambers. Every chamber contributes one outer cavity and one
    deeper cavity inside an attached island, so total enclosed holes = 2*k but top-level chambers = k.
    """
    k = int(top_holes)
    patch = np.zeros((7, 6 * k + 1), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    for wall in range(0, 6 * k + 1, 6):
        patch[:, wall] = color
    for i in range(k):
        base = 6 * i
        patch[1, base + 3] = color
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


# ---------- frame sprite generation ----------

def _frame_patch(hole_count: int) -> np.ndarray:
    """Build a thin hollow frame sprite with exactly `hole_count` chambers."""
    if hole_count < 1:
        raise ValueError("frame sprites require at least one hole")
    horizontal = random.choice([True, False])
    chamber_spans = [random.randint(*FRAME_INNER_SPAN) for _ in range(hole_count)]
    cross_span = random.randint(*FRAME_CROSS_SPAN)

    if horizontal:
        inner_h = cross_span
        inner_w = sum(chamber_spans) + (hole_count - 1)
        h, w = inner_h + 2, inner_w + 2
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        col = 1
        for width in chamber_spans[:-1]:
            col += width
            patch[:, col] = GREY
            col += 1
    else:
        inner_w = cross_span
        inner_h = sum(chamber_spans) + (hole_count - 1)
        h, w = inner_h + 2, inner_w + 2
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        row = 1
        for height in chamber_spans[:-1]:
            row += height
            patch[row, :] = GREY
            row += 1

    mask = patch == GREY
    assert _hole_count(mask) == hole_count
    assert not _has_full_2x2(mask)
    assert h <= FRAME_MAX_H and w <= FRAME_MAX_W
    return patch


def _make_sprite(hole_count: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        sprite = _frame_patch(hole_count)
        if random.random() < 0.35:
            sprite = np.flipud(sprite)
        if random.random() < 0.35:
            sprite = np.fliplr(sprite)
        mask = sprite == GREY
        if _hole_count(mask) != hole_count:
            continue
        if _has_full_2x2(mask):
            continue
        return sprite
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
    distractor_holes = [h for h in FRAME_HOLE_VALUES if h not in legend_holes]
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
        items.append((int(np.sum(sprite == GREY)), sprite, color))
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


def _grey_components_are_frame_like(grid: np.ndarray) -> bool:
    for pts in components_mask(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        if _hole_count(sub) < 1:
            return False
        if _has_full_2x2(sub):
            return False
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
    if not _grey_components_are_frame_like(grid):
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
