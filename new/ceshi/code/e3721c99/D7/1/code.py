# Variant of e3721c99
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出与输入同尺寸，图例和精灵均保留在原位... → 仅输出着色后的精灵（去除背景和图例）
#
# Design analysis
# - Concrete behavior:
#   The input scene stays close to the original task: a divider protects a legend panel,
#   colored legend shapes encode hole-count -> color, and grey sprites sit in the workspace.
#   The variant changes only the output stage: matched sprites are recolored, tight-cropped,
#   grouped by their scene rows, and packed into a compact grid. Legend, divider, distractors,
#   and the rest of the background disappear from the output.
# - Small trace:
#   Original logic would turn a 2-hole grey sprite red in place and keep the full 30x30 scene.
#   Here that same sprite becomes a cropped red patch. If another matched sprite lies on the
#   same input row band, both patches share one packed output row with a 1-cell gap.
# - Complexity calibration:
#   The topology rule is unchanged, so the hard part is still "count holes and match legend".
#   The new difficulty is reading the output-format change. To make the row packing visible,
#   the generator biases matched sprites into 2-3 separated horizontal bands.
# - Hidden assumptions checked against the old main():
#   `_legend_mapping(...)` and hole counting still work because the input scene definition is
#   unchanged. The old `out = g.copy()` assumption no longer holds because output shape now
#   depends only on matched objects. The old per-cell recolor loop is replaced by component
#   extraction + crop + pack, so legends/dividers are never copied into the result.
# - Placement legality:
#   Sprites are still generated as non-overlapping connected blobs. Matched sprites are placed
#   inside separated row bands so their bounding boxes recover clean row groups; distractors are
#   placed afterward with the original collision checks. Because each packed row is tighter than
#   its source row span, the packed output remains within ARC's size limits.
# - Visual observability:
#   Across examples, humans can see that only legend-matching sprites survive, get recolored by
#   hole count, and reappear in a compact row-wise layout. No ordering uses hidden values such as
#   color ids; rows come from visible vertical grouping, and within-row order comes from x-position.

import itertools
import random

import numpy as np
from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste
from arc_utils import label_mask
from scipy.ndimage import binary_dilation, generate_binary_structure


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one 'legend' component per non-(bg,grey) color:
      - component is at least 2x2,
      - component does not touch the outer grid border (filters dividers).
    Map hole_count -> legend_color.
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
        sub = (g[r0 : r1 + 1, c0 : c1 + 1] == c)
        hc = _hole_count(sub)
        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = c
    return mapping


def _component_bbox_mask(pts: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    """Return bbox plus a tight boolean mask for one component."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return int(r0), int(c0), int(r1), int(c1), sub


def _matched_sprite_items(g: np.ndarray, legend: dict[int, int], grey: int, bg: int) -> list[dict]:
    """Extract matched grey sprites as recolored tight patches plus their scene bboxes."""
    items: list[dict] = []
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1, sub = _component_bbox_mask(pts)
        hc = _hole_count(sub)
        if hc not in legend:
            continue
        color = int(legend[hc])
        patch = np.where(sub, color, bg).astype(int)
        items.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "patch": patch,
            }
        )
    return items


def _group_rows(items: list[dict]) -> list[list[dict]]:
    """Group matched sprites into visible scene rows using bbox overlap/touch."""
    ordered = sorted(items, key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    rows: list[list[dict]] = []
    current: list[dict] = []
    current_bottom: int | None = None
    for item in ordered:
        if current and item["r0"] > int(current_bottom) + 1:
            rows.append(sorted(current, key=lambda x: (x["c0"], x["r0"])))
            current = [item]
            current_bottom = int(item["r1"])
        else:
            current.append(item)
            current_bottom = int(item["r1"]) if current_bottom is None else max(int(current_bottom), int(item["r1"]))
    if current:
        rows.append(sorted(current, key=lambda x: (x["c0"], x["r0"])))
    return rows


def _pack_rows(rows: list[list[dict]], bg: int) -> np.ndarray:
    """Pack cropped patches into a compact grid with 1-cell gaps and no outer border."""
    assert rows, "packed output needs at least one matched row"
    row_heights = [max(int(item["patch"].shape[0]) for item in row) for row in rows]
    row_widths = [
        sum(int(item["patch"].shape[1]) for item in row) + (len(row) - 1)
        for row in rows
    ]
    out_h = sum(row_heights) + len(rows) - 1
    out_w = max(row_widths)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    rr = 0
    for row, row_h in zip(rows, row_heights):
        cc = 0
        for item in row:
            patch = item["patch"]
            paste(out, patch, rr, cc, transparent=bg)
            cc += int(patch.shape[1]) + 1
        rr += int(row_h) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor legend-matching grey sprites and pack only those objects into a compact output.

    Input structure:
    - bg is the dominant color; grey (5) sprites are scattered in the workspace.
    - Colored legend shapes sit behind a divider and encode distinct hole counts by topology.
    - Some grey sprites match legend hole counts; others are distractors.

    Rule:
    - Read the legend as a mapping from hole count to color.
    - For each grey sprite, count enclosed holes inside its tight bounding box.
    - Keep only sprites whose hole count appears in the legend, recolor each one to the mapped
      legend color, crop it tightly, then pack the kept sprites row by row using their original
      top-to-bottom / left-to-right arrangement. Discard legend, divider, background, and distractors.

    Key insight: The topology cue still chooses the color, but the answer is no longer the full
    scene; it is a compact object-level collage built from the matched components only.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    items = _matched_sprite_items(g, legend, grey, bg)
    assert items, "grid must contain at least one grey sprite whose hole count appears in the legend"
    rows = _group_rows(items)
    return _pack_rows(rows, bg)


# --------------- generator (holes-first organic blob) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)        # number of legend entries
MAX_HOLE_COUNT = 6                 # hole values drawn from range(MAX_HOLE_COUNT)
PANEL_SIZE_RANGE = (5, 8)          # panel thickness (cross-axis)
FULL_LAYOUT_MAX_SPAN = 28          # max legend line span for full-row/col layouts
L_LAYOUT_MAX_SPAN = 24             # max legend line span for L-shaped layouts
L_LAYOUT_EXTRA_PADDING = 3         # max random padding along the L arm

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)      # total sprites (matches + distractors)
MAX_DISTRACTORS = 7                # max distractor sprites
MAX_EXTRA_PER_LEGEND = 2           # max extra match sprites per legend entry
SPRITE_RETRIES = 30                # retries per sprite generation attempt
SPRITE_AREA_0 = (5, 25)            # (min, max) area for 0-hole sprites
SPRITE_AREA_LOW_MIN = 8            # min area for 1-2 hole sprites (max = 25+10*h)
# 3+ hole sprites: min = 10+2*h, max = 20+4*h
CANVAS_SCALE = 1.2                 # sqrt(area) multiplier for canvas size
CANVAS_MIN = 5                     # minimum canvas dimension
CANVAS_PAD = 2                     # padding added to scaled sqrt for canvas
HOLE_SEED_RADIUS_FACTOR = 0.45     # radius = factor * sqrt(area)
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2              # minimum Chebyshev distance between hole seeds
HOLE_SEED_MARGIN = 2               # margin from canvas edge for hole seeds
SEAL_ITERS = 5                     # max iterations for sealing forbidden cells
SPRITE_MAX_H = 12                  # max sprite bbox height
SPRITE_MAX_W = 18                  # max sprite bbox width
SPRITE_MAX_ASPECT = 2.5            # max aspect ratio (max_dim / min_dim)

# -- packed-row placement --
MATCHED_ROW_COUNT_RANGE = (2, 3)   # matched sprites are biased into 2-3 visible bands
MATCHED_ROW_JITTER = 2             # top-row jitter inside one band
MATCHED_ROW_GAP = 2                # minimum vertical gap between matched bands

# -- generation --
GEN_RETRIES = 300                  # outer retries for generate_input()

# 8-connectivity structuring element for dilation
_STRUCT8 = generate_binary_structure(2, 2)


# ---------- legend helpers ----------

def _legend_span(hole_count: int) -> int:
    """Width (or height) of a single legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan (horizontal) or Spanx3 (vertical) legend patch."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_count:
            patch[1::2, 1] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 legends with distinct hole counts that fit within max_span."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


# ---------- layout helpers ----------

def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row horizontal divider with legend panel above or below."""
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
    """Place a full-column vertical divider with legend panel left or right."""
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
    """Compute divider pivot and legend placement bounds for an L-corner."""
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
    else:  # br
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with legend panel in one corner."""
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

    n = GRID_SIZE
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
    else:  # br
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
    """Randomly choose and apply a divider layout (horizontal/vertical/L)."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


# ---------- sprite generation ----------

def _sprite_area(hole_count: int) -> int:
    """Sample a random sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find well-separated seed positions for holes inside the canvas."""
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
    """Grow irregular blob regions around each hole seed. Returns the full forbidden set."""
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
    """Find the nearest non-forbidden cell to the canvas center."""
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
    """Grow a random connected blob of target area, avoiding forbidden cells."""
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
    """Seal gaps around forbidden cells and keep only the largest connected component."""
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
            best = max(range(1, n_comp + 1), key=lambda l: int(np.sum(labeled == l)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to tight bbox (including forbidden cells), validate shape and holes."""
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
    """Generate a random amorphous grey sprite with the specified number of holes."""
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
    """Build a shuffled list of (hole_count, color_or_None) for all sprites to generate."""
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

    plan = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask (1-cell padding in all directions)."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    sprite: np.ndarray,
    row_min: int | None = None,
    row_max: int | None = None,
) -> tuple[int, int] | None:
    """Find a random valid placement for sprite, optionally inside a row band."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    r_lo = 0 if row_min is None else max(0, int(row_min))
    r_hi = n - h if row_max is None else min(n - h, int(row_max))
    if r_hi < r_lo:
        return None
    candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(n - w + 1)]
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


def _assign_matched_rows(items: list[tuple[np.ndarray, int]], row_count: int) -> list[list[tuple[np.ndarray, int]]]:
    """Distribute matched sprites into a small number of balanced row groups."""
    ordered = items[:]
    random.shuffle(ordered)
    ordered.sort(key=lambda x: (x[0].shape[1], x[0].shape[0], x[0].size), reverse=True)
    rows: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(row_count)]
    widths = [0] * row_count
    heights = [0] * row_count
    for idx, item in enumerate(ordered):
        if idx < row_count:
            target = idx
        else:
            target = min(
                range(row_count),
                key=lambda k: (widths[k], heights[k], len(rows[k])),
            )
        sprite = item[0]
        rows[target].append(item)
        widths[target] += int(sprite.shape[1]) + (1 if len(rows[target]) > 1 else 0)
        heights[target] = max(heights[target], int(sprite.shape[0]))
    return rows


def _row_tops(row_heights: list[int]) -> list[int] | None:
    """Choose vertically separated band starts for matched rows."""
    need = sum(int(h) + MATCHED_ROW_JITTER for h in row_heights) + MATCHED_ROW_GAP * (len(row_heights) - 1)
    if need > GRID_SIZE:
        return None
    slack = GRID_SIZE - need
    extras = [0] * (len(row_heights) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1
    tops = []
    cur = extras[0]
    for i, h in enumerate(row_heights):
        tops.append(int(cur))
        cur += int(h) + MATCHED_ROW_JITTER
        if i < len(row_heights) - 1:
            cur += MATCHED_ROW_GAP + extras[i + 1]
    return tops


def _place_matched_sprites(
    grid: np.ndarray,
    blocked: np.ndarray,
    occupied: np.ndarray,
    items: list[tuple[np.ndarray, int]],
) -> bool:
    """Place matched sprites into 2-3 visible horizontal bands."""
    min_rows = min(MATCHED_ROW_COUNT_RANGE[0], len(items))
    max_rows = min(MATCHED_ROW_COUNT_RANGE[1], len(items))
    options = list(range(max_rows, min_rows - 1, -1))
    random.shuffle(options)

    for row_count in options:
        rows = _assign_matched_rows(items, row_count)
        row_heights = [max(int(sprite.shape[0]) for sprite, _ in row) for row in rows]
        tops = _row_tops(row_heights)
        if tops is None:
            continue

        trial_grid = grid.copy()
        trial_occ = occupied.copy()
        ok = True
        for top, row in zip(tops, rows):
            band_max = top + MATCHED_ROW_JITTER
            row_items = row[:]
            random.shuffle(row_items)
            row_items.sort(key=lambda x: x[0].size, reverse=True)
            for sprite, _color in row_items:
                loc = _random_sprite_location(
                    blocked, trial_occ, sprite, row_min=top, row_max=band_max
                )
                if loc is None:
                    ok = False
                    break
                r, c = loc
                paste(trial_grid, sprite, r, c, transparent=BG)
                paste(trial_occ, sprite, r, c, transparent=BG)
            if not ok:
                break
        if ok:
            grid[:, :] = trial_grid
            occupied[:, :] = trial_occ
            return True
    return False


def _place_sprites(grid: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]) -> bool:
    """Generate and place all sprites (matches + distractors) onto the grid."""
    matched: list[tuple[np.ndarray, int]] = []
    distractors: list[np.ndarray] = []
    for hole_count, color in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        if color is None:
            distractors.append(sprite)
        else:
            matched.append((sprite, int(color)))

    occupied = np.zeros_like(grid)
    if not _place_matched_sprites(grid, blocked, occupied, matched):
        return False

    distractors.sort(key=lambda sprite: sprite.size, reverse=True)
    for sprite in distractors:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
    return True


# ---------- top-level generation ----------

def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate via oracle."""
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

    legends = _add_layout(grid, blocked)
    if not _place_sprites(grid, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    out = main(grid)
    if out.shape == grid.shape:
        return None
    if out.shape[0] > GRID_SIZE or out.shape[1] > GRID_SIZE:
        return None

    legend = _legend_mapping(grid, BG, GREY)
    items = _matched_sprite_items(grid, legend, GREY, BG)
    rows = _group_rows(items)
    if len(rows) < 2:
        return None
    if not np.array_equal(out, _pack_rows(rows, BG)):
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
    raise RuntimeError("failed to generate e3721c99 D7 input")
