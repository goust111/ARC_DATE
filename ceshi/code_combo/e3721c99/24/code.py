# Combo variant of e3721c99 (index 24)
# combo_spec: D5/2+D14/1
# Name: 颜色对图例+图例颜色对映射

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task used colored legend shapes as a direct hole-count lookup: count holes on a
#    grey sprite, then copy the color of the legend shape with the same topology. In this combo, the
#    legend becomes a compact 2-column table made of four rows. Each row is [target_color,
#    marker_color], so the right cell is the color painted on a sprite's hole rim. The four rows are
#    split into two close row-pairs, separated by one blank row. A sprite is solved in two stages:
#      - stage 1: find the marker color touching its enclosed hole(s) and use that to select a row;
#      - stage 2: read the row's group mate and use hole-count parity to decide whether to keep the
#        row's own target color (odd hole count) or switch to the partner row's target color (even).
#    Example: suppose one close pair is [red, blue] / [green, yellow]. A blue-rim sprite with one
#    hole becomes red, but a blue-rim sprite with two holes becomes green. Under D5/2 alone both
#    blue-rim sprites would become red; under the original task both would ignore rim colors and
#    instead match purely by topology.
# b) Complexity calibration:
#    The task still stays in the original 30x30 divider+panel+workspace format, with four legend
#    rows and 9-11 sprites. The new rule uses only two repeated control axes that are both visible:
#    marker color and hole-count parity. Every sample repeats the same odd hole count across all four
#    markers and the same even hole count across all four markers, so humans can compare "same marker,
#    different parity" and "same parity, different marker" without facing too many free variables.
# c) Implicit assumption review:
#    The old solver assumed each legend color appeared as one sizeable legend component and every
#    relevant sprite cell was grey. Neither survives. The combo solver must instead:
#      - find one unique 5x2 legend table rather than large colored blobs;
#      - treat marker-colored rim cells as part of each workspace sprite component;
#      - count sprite holes only after extracting the full mixed-color component;
#      - interpret row pairing from spacing, because the final output can come from the neighboring
#        row instead of from the row that directly matches the marker.
#    New assumptions introduced here are audited below and enforced by generation.
# d) Placement legality:
#    The generator keeps the original blocked legend panel and workspace separation, places the
#    5x2 legend table fully inside the panel with a one-cell inset, and places sprites only in the
#    workspace with one-cell padded non-overlap. Marker cells are written onto sprite rims before
#    placement, so occupancy checks see the true final input footprint.
# e) Visual observability:
#    Group membership is readable from the stacked table itself: rows 1-2 are adjacent, rows 3-4 are
#    adjacent, and the blank row between them marks the two groups. Every marker appears on at least
#    one odd-hole sprite and one even-hole sprite, so the viewer can see that the same right-hand
#    legend color does not always produce the same output. Simultaneously, the odd-hole sprites for
#    different markers share the same topology, showing that marker color also matters.
# f) Visual cleanliness:
#    The legend is a small table isolated in the panel; sprites remain amorphous single components in
#    the workspace; only hole rims carry colored accents. There are no overlaid symbols or hidden
#    ordering cues based on numeric color ids. One fill/marker color overlap may exist globally due
#    palette limits, but never within the same row, so the table stays readable.
# g) Change inferability:
#    The left column always represents possible output colors and the right column always represents
#    possible rim markers. The extra reasoning is structural rather than arbitrary: close rows form a
#    pair, and parity decides whether a marker keeps its own row's left color or flips to its pair
#    mate. This gives the color change a stable visual cue instead of an opaque many-to-many remap.
#
# Solver assumption audit after the change:
#   - _find_legend_groups assumes there is exactly one clean 5x2 table whose colored rows follow the
#     pattern [entry, entry, blank, entry, entry]; generate_input() creates exactly one such table.
#   - _find_legend_groups also assumes each legend row has two distinct non-(bg,div,grey) colors and
#     the table is side-bounded by bg/div; the generator enforces these constraints explicitly.
#   - _component_marker assumes every sprite has at least one enclosed hole and all hole-adjacent
#     non-grey cells share one marker color; the generator only emits hole-bearing sprites and colors
#     every hole rim uniformly.
#   - main() assumes every sprite marker belongs to at most one legend row; generator uses four
#     unique marker colors in the legend.
#   - main() assumes row pairing is recovered by vertical proximity rather than by numeric color ids;
#     generator always uses two adjacent rows per group with exactly one blank row between groups.


BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

LEGEND_GROUP_COUNT = 2
ROWS_PER_GROUP = 2
LEGEND_ROW_COUNT = LEGEND_GROUP_COUNT * ROWS_PER_GROUP
GROUP_GAP_ROWS = 1
LEGEND_PATCH_H = LEGEND_ROW_COUNT + GROUP_GAP_ROWS
LEGEND_PATCH_W = 2

PANEL_SIZE_RANGE = (7, 8)

TOTAL_SPRITES_RANGE = (9, 11)
MAX_DISTRACTORS = 3
MAX_EXTRA_PER_MARKER = 2
MAX_SPRITE_HOLES = 5
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 25)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 12
SPRITE_MAX_W = 18
SPRITE_MAX_ASPECT = 2.5

GEN_RETRIES = 600

_STRUCT8 = generate_binary_structure(2, 2)


def _hole_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Return enclosed background regions inside a binary mask using 4-connectivity."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
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
                holes.append(np.array(pts, dtype=int))
    return holes


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed background components inside a binary mask."""
    return len(_hole_regions(mask))


def _table_entry_rows(top: int) -> list[int]:
    """Row indices for the four legend entries inside a 5x2 table."""
    return [top, top + 1, top + 3, top + 4]


def _find_legend_groups(
    g: np.ndarray, bg: int, grey: int, div: int
) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    """
    Find the unique 5x2 legend table:
      rows 0,1 = group A
      row 2    = blank separator
      rows 3,4 = group B
    Each non-blank row is [target_color, marker_color].
    """
    h, w = g.shape

    def good(v: int) -> bool:
        return int(v) not in (bg, grey, div)

    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r0 in range(h - LEGEND_PATCH_H + 1):
        for c0 in range(w - LEGEND_PATCH_W + 1):
            rows = _table_entry_rows(r0)
            if any(
                not (
                    good(g[r, c0])
                    and good(g[r, c0 + 1])
                    and int(g[r, c0]) != int(g[r, c0 + 1])
                )
                for r in rows
            ):
                continue
            if any(int(g[r0 + 2, c]) != bg for c in range(c0, c0 + 2)):
                continue
            if any(
                c0 > 0 and int(g[r, c0 - 1]) not in (bg, div) or
                c0 + 2 < w and int(g[r, c0 + 2]) not in (bg, div)
                for r in range(r0, r0 + LEGEND_PATCH_H)
            ):
                continue
            if r0 > 0 and any(int(g[r0 - 1, c]) not in (bg, div) for c in range(c0, c0 + 2)):
                continue
            if r0 + LEGEND_PATCH_H < h and any(
                int(g[r0 + LEGEND_PATCH_H, c]) not in (bg, div) for c in range(c0, c0 + 2)
            ):
                continue
            rows_data = [(int(g[r, c0]), int(g[r, c0 + 1])) for r in rows]
            markers = [marker for _, marker in rows_data]
            if len(set(markers)) != len(markers):
                continue
            candidates.append((r0, c0, rows_data))

    assert len(candidates) == 1, "legend must be a unique grouped 2-column table"
    r0, c0, rows_data = candidates[0]
    groups = [rows_data[:2], rows_data[2:]]
    legend_mask = np.zeros_like(g, dtype=bool)
    for r in _table_entry_rows(r0):
        legend_mask[r, c0] = True
        legend_mask[r, c0 + 1] = True
    return groups, legend_mask


def _marker_color_map(groups: list[list[tuple[int, int]]]) -> dict[int, tuple[int, int]]:
    """
    Build marker_color -> (odd_output_color, even_output_color).
    Odd hole counts keep the row's own left color; even hole counts flip to the partner row.
    """
    mapping: dict[int, tuple[int, int]] = {}
    for group in groups:
        (fill0, marker0), (fill1, marker1) = group
        mapping[int(marker0)] = (int(fill0), int(fill1))
        mapping[int(marker1)] = (int(fill1), int(fill0))
    return mapping


def _component_marker(subgrid: np.ndarray, comp_mask: np.ndarray, grey: int) -> int | None:
    """Read the unique non-grey color touching the component's enclosed holes."""
    holes = _hole_regions(comp_mask)
    if not holes:
        return None
    markers: set[int] = set()
    h, w = comp_mask.shape
    for hole in holes:
        for r, c in hole:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and comp_mask[nr, nc]:
                    color = int(subgrid[nr, nc])
                    if color != grey:
                        markers.add(color)
    if len(markers) == 1:
        return next(iter(markers))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hole-marked sprites by combining row lookup with within-pair switching.

    Input structure:
    - The grid still has a divider-colored panel and a larger workspace. Inside the panel sits one
      compact 2-column legend table with four colored rows arranged as two adjacent row-pairs,
      separated by a blank row.
    - In each legend row, the left cell is a candidate output color and the right cell is the rim
      marker color that can appear around enclosed holes on workspace sprites.
    - Workspace sprites are mostly grey (5), but the cells bordering each hole are recolored to one
      marker color from the legend's right column.

    Rule:
    - Read the legend as two groups of two rows. A sprite first matches a row by the marker color
      touching its enclosed holes.
    - Count the sprite's holes. If that count is odd, keep the matched row's left color; if it is
      even, switch to the other row's left color within the same 2-row group.
    - Recolor the entire sprite to that resolved color. Sprites whose marker color is not listed in
      the legend, or whose holes do not expose one unique marker, are erased to bg.

    Key insight: The right-column color no longer gives the final answer by itself; it only chooses a
    local legend pair, and the sprite's topology decides whether to stay on that row or jump to its
    grouped partner.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY
    div = DIV

    groups, legend_mask = _find_legend_groups(g, bg, grey, div)
    mapping = _marker_color_map(groups)

    out = g.copy()
    sprite_mask = (g != bg) & (g != div) & (~legend_mask)
    for pts in components_mask(sprite_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub_mask[r - r0, c - c0] = True
        marker = _component_marker(g[r0 : r1 + 1, c0 : c1 + 1], sub_mask, grey)
        if marker not in mapping:
            newc = bg
        else:
            odd_color, even_color = mapping[marker]
            newc = odd_color if _hole_count(sub_mask) % 2 == 1 else even_color
        for r, c in pts:
            out[r, c] = int(newc)

    return out


def _legend_patch(groups: list[list[tuple[int, int]]]) -> np.ndarray:
    """Build the fixed 5x2 legend table."""
    patch = np.zeros((LEGEND_PATCH_H, LEGEND_PATCH_W), dtype=int)
    rows = [groups[0][0], groups[0][1], None, groups[1][0], groups[1][1]]
    for r, row in enumerate(rows):
        if row is None:
            continue
        fill, marker = row
        patch[r, 0] = int(fill)
        patch[r, 1] = int(marker)
    return patch


def _choose_legend_groups() -> list[list[tuple[int, int]]]:
    """
    Pick four unique markers and four unique fills with exactly one cross-role overlap.
    The overlap is kept off the same row and off the same group as its marker use.
    """
    markers = random.sample(LEGEND_COLORS, LEGEND_ROW_COUNT)
    overlap = random.choice(markers)
    fills = [overlap] + [c for c in LEGEND_COLORS if c not in markers]
    random.shuffle(fills)

    marker_rows = list(range(LEGEND_ROW_COUNT))
    overlap_marker_row = next(i for i, marker in enumerate(markers) if marker == overlap)
    fill_perms = list(itertools.permutations(fills))
    random.shuffle(fill_perms)
    for perm in fill_perms:
        if any(perm[i] == markers[i] for i in marker_rows):
            continue
        overlap_fill_row = perm.index(overlap)
        if overlap_fill_row // 2 == overlap_marker_row // 2:
            continue
        rows = [(int(perm[i]), int(markers[i])) for i in marker_rows]
        return [rows[:2], rows[2:]]
    raise AssertionError("failed to assign legend fills")


def _place_legend_stack(
    grid: np.ndarray, groups: list[list[tuple[int, int]]], panel_bbox: tuple[int, int, int, int]
) -> None:
    """Place the grouped legend table inside the panel bbox."""
    rmin, cmin, rmax, cmax = panel_bbox
    patch = _legend_patch(groups)
    h, w = patch.shape
    assert rmax - rmin + 1 >= h and cmax - cmin + 1 >= w, "legend panel must fit the table"
    r0 = random.randint(rmin, rmax - h + 1)
    c0 = random.randint(cmin, cmax - w + 1)
    paste(grid, patch, r0, c0)


def _layout_full_horizontal(
    grid: np.ndarray, blocked: np.ndarray
) -> tuple[int, int, int, int]:
    """Place a full-row divider with the legend panel above or below it."""
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        return 1, 1, div_r - 2, GRID_SIZE - 2
    div_r = GRID_SIZE - 1 - panel_h
    grid[div_r, :] = DIV
    blocked[div_r:, :] = 9
    return div_r + 2, 1, GRID_SIZE - 2, GRID_SIZE - 2


def _layout_full_vertical(
    grid: np.ndarray, blocked: np.ndarray
) -> tuple[int, int, int, int]:
    """Place a full-column divider with the legend panel on the left or right."""
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        return 1, 1, GRID_SIZE - 2, div_c - 2
    div_c = GRID_SIZE - 1 - panel_w
    grid[:, div_c] = DIV
    blocked[:, div_c:] = 9
    return 1, div_c + 2, GRID_SIZE - 2, GRID_SIZE - 2


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> tuple[int, int, int, int]:
    """Place an L-shaped divider with a corner panel."""
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    corner = random.choice(["tl", "tr", "bl", "br"])

    if corner == "tl":
        dr, dc = panel_h, panel_w
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
        return 1, 1, dr - 2, dc - 2
    if corner == "tr":
        dr, dc = panel_h, GRID_SIZE - 1 - panel_w
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
        return 1, dc + 2, dr - 2, GRID_SIZE - 2
    if corner == "bl":
        dr, dc = GRID_SIZE - 1 - panel_h, panel_w
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
        return dr + 2, 1, GRID_SIZE - 2, dc - 2
    dr, dc = GRID_SIZE - 1 - panel_h, GRID_SIZE - 1 - panel_w
    grid[dr, dc:] = DIV
    grid[dr:, dc] = DIV
    blocked[dr + 1 :, dc + 1 :] = 9
    blocked[dr, dc:] = 9
    blocked[dr:, dc] = 9
    return dr + 2, dc + 2, GRID_SIZE - 2, GRID_SIZE - 2


def _add_layout(
    grid: np.ndarray, blocked: np.ndarray, groups: list[list[tuple[int, int]]]
) -> None:
    """Draw one divider layout and place the legend table in the panel."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        panel_bbox = _layout_full_horizontal(grid, blocked)
    elif kind == "vertical":
        panel_bbox = _layout_full_vertical(grid, blocked)
    else:
        panel_bbox = _layout_l(grid, blocked)
    _place_legend_stack(grid, groups, panel_bbox)


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
    """Grow irregular blob regions around each hole seed."""
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
            best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to tight bbox, validate size/aspect, and emit a grey sprite."""
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


def _mark_hole_rims(sprite: np.ndarray, marker: int) -> np.ndarray | None:
    """Recolor all hole-adjacent sprite cells with the marker color."""
    out = sprite.copy()
    mask = sprite == GREY
    holes = _hole_regions(mask)
    if not holes:
        return None
    rim = np.zeros_like(mask, dtype=bool)
    h, w = sprite.shape
    for hole in holes:
        for r, c in hole:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    rim[nr, nc] = True
    if not np.any(rim):
        return None
    out[rim] = int(marker)
    return out


def _expected_color_for_marker(
    groups: list[list[tuple[int, int]]], marker: int, hole_count: int
) -> int:
    """Resolve the combo rule for one sprite specification."""
    odd_color, even_color = _marker_color_map(groups)[int(marker)]
    return odd_color if hole_count % 2 == 1 else even_color


def _sprite_plan(groups: list[list[tuple[int, int]]]) -> list[tuple[int, int, int | None]]:
    """
    Build a shuffled list of (hole_count, marker_color, output_or_None) items.
    Every legend marker gets one odd-hole and one even-hole example to expose the pair rule.
    """
    total = random.randint(*TOTAL_SPRITES_RANGE)
    flat = [row for group in groups for row in group]
    markers = [marker for _, marker in flat]
    base_count = 2 * len(markers)
    remaining = total - base_count
    max_distractors = min(MAX_DISTRACTORS, remaining)
    n_distractors = random.randint(1, max_distractors) if max_distractors > 0 else 0
    extra_matches = remaining - n_distractors

    odd_holes = random.choice([1, 3, 5])
    even_holes = random.choice([2, 4])

    plan: list[tuple[int, int, int | None]] = []
    extras = {int(marker): 0 for marker in markers}
    for fill, marker in flat:
        del fill
        plan.append((odd_holes, int(marker), _expected_color_for_marker(groups, marker, odd_holes)))
        plan.append((even_holes, int(marker), _expected_color_for_marker(groups, marker, even_holes)))

    for _ in range(extra_matches):
        choices = [m for m in markers if extras[int(m)] < MAX_EXTRA_PER_MARKER]
        marker = int(random.choice(choices))
        extras[marker] += 1
        hole_count = random.randint(1, MAX_SPRITE_HOLES)
        plan.append((hole_count, marker, _expected_color_for_marker(groups, marker, hole_count)))

    distractor_markers = [c for c in LEGEND_COLORS if c not in markers]
    for _ in range(n_distractors):
        plan.append((random.randint(1, MAX_SPRITE_HOLES), int(random.choice(distractor_markers)), None))

    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for a sprite, avoiding blocked and occupied areas."""
    h, w = sprite.shape
    sprite_mask = sprite != BG
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


def _place_sprites(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    groups: list[list[tuple[int, int]]],
) -> bool:
    """Generate and place all sprites (matches + distractors) onto the grid."""
    items = []
    for hole_count, marker, fill in _sprite_plan(groups):
        base = _make_sprite(hole_count)
        if base is None:
            return False
        sprite = _mark_hole_rims(base, marker)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, fill))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, fill in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if fill is not None:
            out_patch = np.where(sprite != BG, int(fill), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _fusion_signal(grid: np.ndarray) -> tuple[int, int]:
    """
    Quantify non-degenerate combo behavior on one input.
    Returns:
      - switched_matches: matched sprites whose output differs from direct marker->own-fill lookup
      - multi_output_markers: marker colors that appear on both odd and even matched sprites and
        therefore lead to two different outputs
    """
    g = np.asarray(grid, dtype=int)
    groups, legend_mask = _find_legend_groups(g, BG, GREY, DIV)
    direct = {marker: fill for group in groups for fill, marker in group}
    combo = _marker_color_map(groups)

    switched = 0
    marker_outputs: dict[int, set[int]] = {}
    sprite_mask = (g != BG) & (g != DIV) & (~legend_mask)
    for pts in components_mask(sprite_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub_mask[r - r0, c - c0] = True
        marker = _component_marker(g[r0 : r1 + 1, c0 : c1 + 1], sub_mask, GREY)
        if marker not in combo:
            continue
        odd_color, even_color = combo[marker]
        outc = odd_color if _hole_count(sub_mask) % 2 == 1 else even_color
        if outc != direct[marker]:
            switched += 1
        marker_outputs.setdefault(marker, set()).add(int(outc))

    multi_output = sum(len(colors) >= 2 for colors in marker_outputs.values())
    return switched, multi_output


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, and validate via the solver."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    groups = _choose_legend_groups()
    _add_layout(grid, blocked, groups)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, groups):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum((grid != BG) & (grid != DIV))):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    switched, multi_output = _fusion_signal(grid)
    if switched < LEGEND_ROW_COUNT or multi_output < LEGEND_ROW_COUNT:
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
    raise RuntimeError("failed to generate e3721c99 combo variant input")
